import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.interpolate import interp1d
from sklearn.decomposition import FastICA, NMF
import warnings

warnings.filterwarnings('ignore')


class TimeSignalEnhancement:
    """
    Comprehensive implementation of time-domain signal enhancement and recovery techniques
    """

    def __init__(self):
        self.fs = 1000  # Sampling frequency
        self.t = np.linspace(0, 2, 2000)  # Time vector

    def generate_test_signals(self):
        """Generate test signals for demonstration"""
        # Clean signal: mixture of sinusoids
        clean = np.sin(2 * np.pi * 50 * self.t) + 0.5 * np.sin(2 * np.pi * 120 * self.t) + 0.3 * np.sin(
            2 * np.pi * 200 * self.t)

        # Add noise
        noise = np.random.normal(0, 0.3, len(self.t))
        noisy = clean + noise

        # Convolved signal (simulating room impulse response)
        h = np.array([1, 0.5, 0.25, 0.125])  # Simple impulse response
        convolved = np.convolve(clean, h, mode='same')

        return clean, noisy, convolved, noise

    # =====================================================
    # 1. NOISE SUPPRESSION TECHNIQUES
    # =====================================================

    # ------------------------------------------------------------------
    # 1. Spectral subtraction (fixed STFT tail, perfect-reconstruction,
    #    and gain computed on magnitude spectrum)
    # ------------------------------------------------------------------
    def spectral_subtraction_time_domain(self, noisy_signal, noise_estimate,
                                         alpha=2.0, beta=0.01):
        window_size = 256
        hop_size = window_size // 2
        window = np.hanning(window_size)

        noise_fft = np.fft.fft(noise_estimate[:window_size] * window)
        noise_power = np.abs(noise_fft) ** 2

        enhanced = np.zeros_like(noisy_signal, dtype=float)
        window_sum = np.zeros_like(noisy_signal, dtype=float)

        for i in range(0, len(noisy_signal) - window_size + 1, hop_size):
            frame = noisy_signal[i:i + window_size] * window
            frame_fft = np.fft.fft(frame)
            frame_power = np.abs(frame_fft) ** 2

            gain = 1 - alpha * (noise_power / (frame_power + 1e-10))
            gain = np.sqrt(np.maximum(gain, beta))  # magnitude gain

            enhanced_fft = frame_fft * gain
            enhanced_frame = np.real(np.fft.ifft(enhanced_fft)) * window

            enhanced[i:i + window_size] += enhanced_frame
            window_sum[i:i + window_size] += window ** 2  # OLA normaliser

        window_sum[window_sum == 0] = 1.0  # avoid /0
        enhanced /= window_sum
        return enhanced

    # ------------------------------------------------------------------
    # 2. Blind deconvolution (ℓ2 normalisation and convergence check)
    # ------------------------------------------------------------------
    def blind_deconvolution(self, convolved_signal, max_iter=100):
        N = len(convolved_signal)
        x_est = convolved_signal.copy()
        h_est = np.array([1.0, 0.1, 0.01], dtype=float)
        λ = 0.01  # regulariser

        for it in range(max_iter):
            H = np.fft.fft(h_est, N)
            Y = np.fft.fft(convolved_signal)

            X = Y * np.conj(H) / (np.abs(H) ** 2 + λ)
            x_new = np.real(np.fft.ifft(X))

            X_fft = np.fft.fft(x_new)
            h_new = Y * np.conj(X_fft) / (np.abs(X_fft) ** 2 + λ)
            h_new = np.real(np.fft.ifft(h_new))[:len(h_est)]
            h_new /= (np.linalg.norm(h_new) + 1e-12)  # ℓ2 norm

            if it % 5 == 0:
                reconv = np.convolve(x_new, h_new, mode='same')
                if np.linalg.norm(reconv - convolved_signal) < 1e-6:
                    break

            x_est, h_est = x_new, h_new

        return x_est, h_est

    def homomorphic_filtering(self, signal_in, cutoff_freq=100):
        """
        Homomorphic filtering for multiplicative noise removal
        """
        # Take logarithm (handle negative values)
        log_signal = np.log(np.abs(signal_in) + 1e-10)

        # Design low-pass filter
        nyquist = self.fs / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='low')

        # Filter in log domain
        filtered_log = signal.filtfilt(b, a, log_signal)

        # Convert back to linear domain
        enhanced = np.exp(filtered_log) * np.sign(signal_in)

        return enhanced

    # ------------------------------------------------------------------
    # 3. MAP estimation (unchanged math, but cleaned variable scope)
    # ------------------------------------------------------------------
    def map_estimation_enhancement(self, noisy_signal, noise_var=0.1,
                                   signal_var=1.0):
        nperseg = min(256, len(noisy_signal) // 4)
        f, Pxx = signal.welch(noisy_signal, self.fs, nperseg=nperseg)
        H = Pxx / (Pxx + noise_var / signal_var)

        N = len(noisy_signal)
        freqs = np.fft.fftfreq(N, 1 / self.fs)
        H_interp = np.interp(np.abs(freqs), f, H)

        enhanced_fft = np.fft.fft(noisy_signal) * H_interp
        return np.real(np.fft.ifft(enhanced_fft))

    # ------------------------------------------------------------------
    # 4. Wavelet denoising (√2-normalised filters + zero-phase reconstr.)
    # ------------------------------------------------------------------
    def wavelet_denoising_time_domain(self, noisy_signal, threshold=0.1):
        h0 = np.array([0.4830, 0.8365, 0.2241, -0.1294]) / np.sqrt(2.0)
        h1 = np.array([-0.1294, -0.2241, 0.8365, -0.4830]) / np.sqrt(2.0)

        def dwt_step(x):
            cA = signal.lfilter(h0, 1, x)[::2]
            cD = signal.lfilter(h1, 1, x)[::2]
            return cA, cD

        def idwt_step(cA, cD):
            cA_up = np.zeros(2 * len(cA));
            cA_up[::2] = cA
            cD_up = np.zeros(2 * len(cD));
            cD_up[::2] = cD
            g0, g1 = h0[::-1], h1[::-1]
            x_rec = signal.filtfilt(g0, 1, cA_up) + signal.filtfilt(g1, 1, cD_up)
            return x_rec[:len(cA_up)]

        cA1, cD1 = dwt_step(noisy_signal)
        cA2, cD2 = dwt_step(cA1)
        cD1_thresh = np.where(np.abs(cD1) > threshold, cD1, 0)
        cD2_thresh = np.where(np.abs(cD2) > threshold, cD2, 0)
        cA1_rec = idwt_step(cA2, cD2_thresh)
        enhanced = idwt_step(cA1_rec[:len(cD1_thresh)], cD1_thresh)
        return enhanced[:len(noisy_signal)]

    # =====================================================
    # 2. BLIND SIGNAL SEPARATION
    # =====================================================

    def generate_mixed_signals(self):
        """Generate mixed signals for BSS demonstration"""
        # Source signals
        s1 = np.sin(2 * np.pi * 50 * self.t) + np.sin(2 * np.pi * 120 * self.t)
        s2 = signal.sawtooth(2 * np.pi * 20 * self.t)
        s3 = np.random.normal(0, 0.5, len(self.t))

        sources = np.array([s1, s2, s3])

        # Mixing matrix
        A = np.array([[0.8, 0.3, 0.1],
                      [0.2, 0.9, 0.2],
                      [0.1, 0.2, 0.7]])

        # Mixed signals
        mixed = A @ sources

        return sources, mixed, A

    # ------------------------------------------------------------------
    # 11. ICA (future-proof whiten parameter)
    # ------------------------------------------------------------------
    def ica_time_domain(self, mixed_signals, max_iter=1000):
        ica = FastICA(n_components=mixed_signals.shape[0],
                      max_iter=max_iter, tol=1e-6,
                      whiten='unit-variance', random_state=0)
        separated = ica.fit_transform(mixed_signals.T).T
        mixing_matrix = ica.mixing_
        return separated, mixing_matrix

    def matching_pursuit(self, signal_in, dictionary, max_iter=50, threshold=0.01):
        """
        Matching Pursuit algorithm for sparse decomposition
        """
        # Simple dictionary of sinusoids and wavelets
        residual = signal_in.copy()
        coefficients = []
        indices = []

        for iteration in range(max_iter):
            # Find best matching atom
            correlations = np.abs([np.dot(residual, atom) for atom in dictionary])
            best_idx = np.argmax(correlations)
            best_coeff = np.dot(residual, dictionary[best_idx])

            # Check stopping criterion
            if np.abs(best_coeff) < threshold:
                break

            # Update residual
            residual -= best_coeff * dictionary[best_idx]

            # Store results
            coefficients.append(best_coeff)
            indices.append(best_idx)

        return coefficients, indices, residual

    def nmf_separation(self, magnitude_spectrogram, n_components=3, max_iter=200):
        """
        Non-negative Matrix Factorization for signal separation
        """
        # Apply NMF
        nmf = NMF(n_components=n_components, max_iter=max_iter, random_state=42)
        W = nmf.fit_transform(magnitude_spectrogram)  # Basis functions
        H = nmf.components_  # Activations

        return W, H

    def time_domain_cnn_separation(self, mixed_signal):
        """
        Simplified time-domain CNN approach (conceptual implementation)
        """
        # This is a conceptual implementation
        # In practice, would use deep learning frameworks

        # Simple linear transformation as placeholder
        # Would be replaced with trained CNN weights
        kernel_size = 16
        n_filters = 32

        # Simulate CNN processing
        processed = np.convolve(mixed_signal, np.random.normal(0, 0.1, kernel_size), mode='same')

        return processed

    # =====================================================
    # 3. SIGNAL RECONSTRUCTION AND INTERPOLATION
    # =====================================================

    # ------------------------------------------------------------------
    # 5. Zero-order hold (robust to unsorted indices)
    # ------------------------------------------------------------------
    def zero_order_hold(self, original_signal, missing_indices):
        reconstructed = original_signal.copy()
        for idx in np.sort(missing_indices):
            reconstructed[idx] = reconstructed[idx - 1] if idx > 0 else 0.0
        return reconstructed

    def linear_interpolation(self, original_signal, missing_indices):
        """
        Linear interpolation for missing samples
        """
        reconstructed = original_signal.copy()
        valid_indices = np.setdiff1d(np.arange(len(original_signal)), missing_indices)

        # Use scipy's interpolation
        f = interp1d(valid_indices, original_signal[valid_indices],
                     kind='linear', bounds_error=False, fill_value='extrapolate')

        reconstructed[missing_indices] = f(missing_indices)

        return reconstructed

    # ------------------------------------------------------------------
    # 6. Polynomial interpolation (guard against over-fitting)
    # ------------------------------------------------------------------
    def polynomial_interpolation(self, original_signal, missing_indices,
                                 degree=3):
        reconstructed = original_signal.copy()
        valid_indices = np.setdiff1d(np.arange(len(original_signal)),
                                     missing_indices)
        degree = min(degree, len(valid_indices) - 1)
        coeffs = np.polyfit(valid_indices,
                            original_signal[valid_indices], degree)
        poly = np.poly1d(coeffs)
        reconstructed[missing_indices] = poly(missing_indices)
        return reconstructed

    # ------------------------------------------------------------------
    # 7. Sinc interpolation (simpler, faster with np.sinc)
    # ------------------------------------------------------------------
    def sinc_interpolation(self, original_signal, missing_indices,
                           bandwidth=0.4):
        reconstructed = original_signal.copy()
        valid_indices = np.setdiff1d(np.arange(len(original_signal)),
                                     missing_indices)

        for m in missing_indices:
            t_diff = m - valid_indices
            kernel = np.sinc(bandwidth * t_diff)
            reconstructed[m] = np.sum(original_signal[valid_indices] * kernel)
        return reconstructed

    def ar_reconstruction(self, signal_with_gaps, missing_indices, order=10):
        """
        Autoregressive model reconstruction
        """
        reconstructed = signal_with_gaps.copy()
        valid_indices = np.setdiff1d(np.arange(len(signal_with_gaps)), missing_indices)

        # Estimate AR parameters from valid samples
        valid_signal = signal_with_gaps[valid_indices]

        # Simple AR estimation (Yule-Walker)
        if len(valid_signal) > order:
            ar_coeffs = self._estimate_ar_coefficients(valid_signal, order)

            # Reconstruct missing samples
            for idx in missing_indices:
                if idx >= order:
                    prediction = 0
                    for i in range(order):
                        if idx - i - 1 not in missing_indices:
                            prediction += ar_coeffs[i] * reconstructed[idx - i - 1]
                    reconstructed[idx] = prediction

        return reconstructed

    # ------------------------------------------------------------------
    # 8. AR coefficient estimation (mean-removed data)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 8-bis. AR coefficient estimation (safe, positive-lag autocorr)
    # ------------------------------------------------------------------
    def _estimate_ar_coefficients(self, signal_data, order):
        """
        Yule-Walker AR-parameter estimation with proper lag handling.
        """
        signal_data = signal_data - np.mean(signal_data)  # de-mean
        N = len(signal_data)
        max_lag = min(order, N - 1)  # safety

        # full autocorrelation then keep non-negative lags only
        autocorr_full = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr_full[N - 1: N + max_lag]  # lag 0 … max_lag

        if autocorr[0] == 0:  # degenerate case
            return np.zeros(order)

        autocorr /= autocorr[0]  # normalise

        # build Toeplitz R and r for Yule-Walker
        R = np.array([[autocorr[abs(i - j)] for j in range(max_lag)]
                      for i in range(max_lag)], dtype=float)
        r = autocorr[1:max_lag + 1]

        try:
            a_hat = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            a_hat = np.zeros(max_lag)

        # pad with zeros if max_lag < order
        if max_lag < order:
            a_hat = np.pad(a_hat, (0, order - max_lag))

        return a_hat

    # ------------------------------------------------------------------
    # 9. Compressed sensing (optimal step + sparsify once)
    # ------------------------------------------------------------------
    def compressed_sensing_recovery(self, measurements, measurement_matrix,
                                    sparsity_level=10, max_iter=100):
        x = np.zeros(measurement_matrix.shape[1])
        L = np.linalg.norm(measurement_matrix, 2) ** 2
        step_size = 1.0 / L
        threshold = 0.1

        for _ in range(max_iter):
            residual = measurements - measurement_matrix @ x
            gradient = measurement_matrix.T @ residual
            x = x + step_size * gradient
            x = np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

        keep_idx = np.argsort(np.abs(x))[-sparsity_level:]
        x_sparse = np.zeros_like(x)
        x_sparse[keep_idx] = x[keep_idx]
        return x_sparse

    # ------------------------------------------------------------------
    # 10. CLEAN algorithm (remove edge-guard)
    # ------------------------------------------------------------------
    def clean_algorithm(self, dirty_signal, psf, gain=0.1,
                        threshold=0.01, max_iter=100):
        clean_components = np.zeros_like(dirty_signal)
        residual = dirty_signal.copy()

        for _ in range(max_iter):
            peak_idx = np.argmax(np.abs(residual))
            peak_value = residual[peak_idx]
            if np.abs(peak_value) < threshold:
                break

            clean_components[peak_idx] += gain * peak_value
            residual[peak_idx:peak_idx + len(psf)] -= gain * peak_value * psf

        reconstructed = clean_components + residual
        return reconstructed, clean_components

    def plot_noise_suppression_methods(self):
        """Plot all noise suppression techniques separately"""
        clean, noisy, convolved, noise = self.generate_test_signals()

        # Figure 1: Spectral Subtraction
        fig1 = plt.figure(figsize=(15, 10))
        enhanced_spectral = self.spectral_subtraction_time_domain(noisy, noise)

        plt.subplot(2, 2, 1)
        plt.plot(self.t[:800], clean[:800], 'g-', label='Clean Signal', linewidth=2)
        plt.plot(self.t[:800], noisy[:800], 'r-', label='Noisy Signal', alpha=0.7)
        plt.title('Original vs Noisy Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.t[:800], clean[:800], 'g-', label='Clean Signal', linewidth=2)
        plt.plot(self.t[:800], enhanced_spectral[:800], 'b-', label='Enhanced Signal', linewidth=2)
        plt.title('Spectral Subtraction Result')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        freqs = np.fft.fftfreq(len(clean), 1 / self.fs)[:len(clean) // 2]
        clean_fft = np.abs(np.fft.fft(clean))[:len(clean) // 2]
        noisy_fft = np.abs(np.fft.fft(noisy))[:len(clean) // 2]
        enhanced_fft = np.abs(np.fft.fft(enhanced_spectral))[:len(clean) // 2]

        plt.loglog(freqs, clean_fft, 'g-', label='Clean', linewidth=2)
        plt.loglog(freqs, noisy_fft, 'r-', label='Noisy', alpha=0.7)
        plt.loglog(freqs, enhanced_fft, 'b-', label='Enhanced', linewidth=2)
        plt.title('Frequency Domain Comparison')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        f, t_spec, Sxx_noisy = signal.spectrogram(noisy, self.fs, nperseg=128)
        plt.imshow(10 * np.log10(np.abs(Sxx_noisy) + 1e-10), aspect='auto', origin='lower', extent=[0, 2, 0, 500])
        plt.title('Noisy Signal Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Power (dB)')

        plt.suptitle('Spectral Subtraction Method', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 2: Blind Deconvolution
        fig2 = plt.figure(figsize=(15, 10))
        deconvolved, h_est = self.blind_deconvolution(convolved)

        plt.subplot(2, 2, 1)
        plt.plot(self.t[:800], clean[:800], 'g-', label='Original Signal', linewidth=2)
        plt.plot(self.t[:800], convolved[:800], 'r-', label='Convolved Signal', alpha=0.7)
        plt.title('Original vs Convolved Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.t[:800], clean[:800], 'g-', label='Original Signal', linewidth=2)
        plt.plot(self.t[:800], deconvolved[:800], 'b-', label='Deconvolved Signal', linewidth=2)
        plt.title('Blind Deconvolution Result')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.stem(range(len(h_est)), h_est, basefmt='b-')
        plt.title('Estimated Impulse Response')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        error = np.abs(deconvolved - clean)
        plt.plot(self.t[:800], error[:800], 'r-', linewidth=2)
        plt.title('Reconstruction Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error')
        plt.grid(True)

        plt.suptitle('Blind Deconvolution Method', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 3: Homomorphic Filtering
        fig3 = plt.figure(figsize=(15, 10))
        homomorphic = self.homomorphic_filtering(noisy)

        plt.subplot(2, 2, 1)
        plt.plot(self.t[:800], clean[:800], 'g-', label='Clean Signal', linewidth=2)
        plt.plot(self.t[:800], noisy[:800], 'r-', label='Noisy Signal', alpha=0.7)
        plt.title('Original Signals')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.t[:800], clean[:800], 'g-', label='Clean Signal', linewidth=2)
        plt.plot(self.t[:800], homomorphic[:800], 'b-', label='Homomorphic Filtered', linewidth=2)
        plt.title('Homomorphic Filtering Result')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        log_signal = np.log(np.abs(noisy) + 1e-10)
        plt.plot(self.t[:800], log_signal[:800], 'r-', label='Log Domain Signal', alpha=0.7)
        plt.title('Log Domain Processing')
        plt.xlabel('Time (s)')
        plt.ylabel('Log Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        snr_improvement = 10 * np.log10(np.var(noisy - clean) / (np.var(homomorphic - clean) + 1e-10))
        plt.bar(['Original', 'Homomorphic'], [0, max(0, snr_improvement)], color=['red', 'blue'])
        plt.title(f'SNR Improvement: {snr_improvement:.2f} dB')
        plt.ylabel('SNR Improvement (dB)')
        plt.grid(True)

        plt.suptitle('Homomorphic Filtering Method', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 4: MAP Estimation
        fig4 = plt.figure(figsize=(15, 10))
        map_enhanced = self.map_estimation_enhancement(noisy)

        plt.subplot(2, 2, 1)
        plt.plot(self.t[:800], clean[:800], 'g-', label='Clean Signal', linewidth=2)
        plt.plot(self.t[:800], noisy[:800], 'r-', label='Noisy Signal', alpha=0.7)
        plt.title('Input Signals')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.t[:800], clean[:800], 'g-', label='Clean Signal', linewidth=2)
        plt.plot(self.t[:800], map_enhanced[:800], 'b-', label='MAP Enhanced', linewidth=2)
        plt.title('MAP Estimation Result')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        f, Pxx_noisy = signal.welch(noisy, self.fs, nperseg=256)
        f, Pxx_clean = signal.welch(clean, self.fs, nperseg=256)
        f, Pxx_enhanced = signal.welch(map_enhanced, self.fs, nperseg=256)

        plt.loglog(f, Pxx_clean, 'g-', label='Clean PSD', linewidth=2)
        plt.loglog(f, Pxx_noisy, 'r-', label='Noisy PSD', alpha=0.7)
        plt.loglog(f, Pxx_enhanced, 'b-', label='Enhanced PSD', linewidth=2)
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        # Wiener filter response
        H = Pxx_noisy / (Pxx_noisy + 0.1)
        plt.semilogx(f, H, 'b-', linewidth=2)
        plt.title('MAP Filter Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Filter Gain')
        plt.grid(True)

        plt.suptitle('Maximum A Posteriori (MAP) Estimation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 5: Wavelet Denoising
        fig5 = plt.figure(figsize=(15, 10))
        wavelet_denoised = self.wavelet_denoising_time_domain(noisy)

        plt.subplot(2, 2, 1)
        plt.plot(self.t[:800], clean[:800], 'g-', label='Clean Signal', linewidth=2)
        plt.plot(self.t[:800], noisy[:800], 'r-', label='Noisy Signal', alpha=0.7)
        plt.title('Input Signals')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.t[:800], clean[:800], 'g-', label='Clean Signal', linewidth=2)
        plt.plot(self.t[:800], wavelet_denoised[:800], 'b-', label='Wavelet Denoised', linewidth=2)
        plt.title('Wavelet Denoising Result')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        # Simplified wavelet coefficients visualization
        h0 = np.array([0.4830, 0.8365, 0.2241, -0.1294])
        h1 = np.array([-0.1294, -0.2241, 0.8365, -0.4830])
        cD1 = signal.lfilter(h1, 1, noisy)[::2]
        plt.plot(cD1[:200], 'r-', alpha=0.7, label='Detail Coefficients')
        plt.title('Wavelet Detail Coefficients')
        plt.xlabel('Coefficient Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 4)
        methods = ['Original', 'Spectral Sub.', 'Blind Deconv.', 'Homomorphic', 'MAP', 'Wavelet']
        enhanced_signals = [noisy, enhanced_spectral, deconvolved, homomorphic, map_enhanced, wavelet_denoised]

        mse_values = []
        for enhanced in enhanced_signals:
            mse = np.mean((enhanced - clean) ** 2)
            mse_values.append(mse)

        plt.bar(methods, mse_values, color=['red', 'blue', 'green', 'orange', 'purple', 'brown'])
        plt.title('Mean Squared Error Comparison')
        plt.ylabel('MSE')
        plt.xticks(rotation=45)
        plt.yscale('log')
        plt.grid(True)

        plt.suptitle('Wavelet Denoising Method', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return {
            'enhanced_spectral': enhanced_spectral,
            'deconvolved': deconvolved,
            'homomorphic': homomorphic,
            'map_enhanced': map_enhanced,
            'wavelet_denoised': wavelet_denoised
        }

    def plot_blind_signal_separation(self):
        """Plot blind signal separation methods separately"""
        sources, mixed, A = self.generate_mixed_signals()

        # Figure 6: ICA Separation
        fig6 = plt.figure(figsize=(15, 12))
        separated, mixing_matrix = self.ica_time_domain(mixed)

        plt.subplot(3, 3, 1)
        plt.plot(self.t[:500], sources[0][:500], 'g-', linewidth=2)
        plt.title('Original Source 1')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(3, 3, 2)
        plt.plot(self.t[:500], sources[1][:500], 'g-', linewidth=2)
        plt.title('Original Source 2')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(3, 3, 3)
        plt.plot(self.t[:500], sources[2][:500], 'g-', linewidth=2)
        plt.title('Original Source 3')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(3, 3, 4)
        plt.plot(self.t[:500], mixed[0][:500], 'r-', linewidth=2)
        plt.title('Mixed Signal 1')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(3, 3, 5)
        plt.plot(self.t[:500], mixed[1][:500], 'r-', linewidth=2)
        plt.title('Mixed Signal 2')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(3, 3, 6)
        plt.plot(self.t[:500], mixed[2][:500], 'r-', linewidth=2)
        plt.title('Mixed Signal 3')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(3, 3, 7)
        plt.plot(self.t[:500], separated[0][:500], 'b-', linewidth=2)
        plt.title('Separated Signal 1')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(3, 3, 8)
        plt.plot(self.t[:500], separated[1][:500], 'b-', linewidth=2)
        plt.title('Separated Signal 2')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(3, 3, 9)
        plt.plot(self.t[:500], separated[2][:500], 'b-', linewidth=2)
        plt.title('Separated Signal 3')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.suptitle('Independent Component Analysis (ICA)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 7: Matching Pursuit
        fig7 = plt.figure(figsize=(15, 10))

        # Create dictionary
        dictionary = []
        for freq in [10, 25, 50, 100, 200]:
            dictionary.append(np.sin(2 * np.pi * freq * self.t))
            dictionary.append(np.cos(2 * np.pi * freq * self.t))

        clean, _, _, _ = self.generate_test_signals()
        coeffs, indices, residual = self.matching_pursuit(clean, dictionary)

        # Reconstruct signal
        reconstructed_mp = np.zeros_like(clean)
        for coeff, idx in zip(coeffs, indices):
            reconstructed_mp += coeff * dictionary[idx]

        plt.subplot(2, 2, 1)
        plt.plot(self.t[:800], clean[:800], 'g-', label='Original Signal', linewidth=2)
        plt.title('Original Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.t[:800], reconstructed_mp[:800], 'b-', label='MP Reconstruction', linewidth=2)
        plt.title('Matching Pursuit Reconstruction')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.stem(range(len(coeffs)), coeffs, basefmt='b-')
        plt.title('Sparse Coefficients')
        plt.xlabel('Coefficient Index')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(self.t[:800], residual[:800], 'r-', linewidth=2)
        plt.title('Residual Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.suptitle('Matching Pursuit Sparse Decomposition', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 8: NMF Separation
        fig8 = plt.figure(figsize=(15, 10))

        f, t_spec, Sxx = signal.spectrogram(mixed[0], self.fs, nperseg=64)
        W, H = self.nmf_separation(np.abs(Sxx), n_components=3)

        plt.subplot(2, 3, 1)
        plt.imshow(np.log(np.abs(Sxx) + 1e-10), aspect='auto', origin='lower')
        plt.title('Original Spectrogram')
        plt.xlabel('Time Frame')
        plt.ylabel('Frequency Bin')
        plt.colorbar()

        plt.subplot(2, 3, 2)
        plt.plot(W[:, 0], 'b-', linewidth=2)
        plt.title('NMF Basis Function 1')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 3, 3)
        plt.plot(W[:, 1], 'r-', linewidth=2)
        plt.title('NMF Basis Function 2')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 3, 4)
        plt.plot(W[:, 2], 'g-', linewidth=2)
        plt.title('NMF Basis Function 3')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 3, 5)
        plt.imshow(H, aspect='auto', origin='lower')
        plt.title('NMF Activation Matrix')
        plt.xlabel('Time Frame')
        plt.ylabel('Component')
        plt.colorbar()

        plt.subplot(2, 3, 6)
        reconstructed_nmf = W @ H
        plt.imshow(np.log(reconstructed_nmf + 1e-10), aspect='auto', origin='lower')
        plt.title('NMF Reconstructed Spectrogram')
        plt.xlabel('Time Frame')
        plt.ylabel('Frequency Bin')
        plt.colorbar()

        plt.suptitle('Non-negative Matrix Factorization (NMF)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return {
            'sources': sources,
            'mixed': mixed,
            'separated': separated,
            'mp_reconstruction': reconstructed_mp,
            'nmf_W': W,
            'nmf_H': H
        }

    def plot_signal_reconstruction(self):
        """Plot signal reconstruction and interpolation methods separately"""
        clean, _, _, _ = self.generate_test_signals()

        # Create signal with missing samples
        test_signal = clean.copy()
        missing_indices = np.random.choice(len(test_signal), size=80, replace=False)
        missing_indices = np.sort(missing_indices)
        signal_with_gaps = test_signal.copy()
        signal_with_gaps[missing_indices] = 0

        # Figure 9: Zero-Order Hold
        fig9 = plt.figure(figsize=(15, 10))
        zoh_recon = self.zero_order_hold(signal_with_gaps, missing_indices)

        plt.subplot(2, 2, 1)
        plt.plot(self.t[:800], test_signal[:800], 'g-', label='Original Signal', linewidth=2)
        valid_mask = np.ones(len(test_signal), dtype=bool)
        valid_mask[missing_indices] = False
        plt.plot(self.t[:800][valid_mask[:800]], signal_with_gaps[:800][valid_mask[:800]], 'ro',
                 markersize=3, label='Available Samples')
        plt.title('Signal with Missing Samples')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.t[:800], test_signal[:800], 'g-', label='Original Signal', linewidth=2, alpha=0.7)
        plt.plot(self.t[:800], zoh_recon[:800], 'b-', label='ZOH Reconstruction', linewidth=2)
        plt.title('Zero-Order Hold Interpolation')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        error_zoh = np.abs(zoh_recon - test_signal)
        plt.plot(self.t[:800], error_zoh[:800], 'r-', linewidth=2)
        plt.title('Reconstruction Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        # Zoom in on a section to show step behavior
        zoom_start, zoom_end = 200, 300
        plt.plot(self.t[zoom_start:zoom_end], test_signal[zoom_start:zoom_end], 'g-',
                 label='Original', linewidth=2)
        plt.plot(self.t[zoom_start:zoom_end], zoh_recon[zoom_start:zoom_end], 'b-',
                 label='ZOH', linewidth=2)
        plt.title('Detailed View (Step Behavior)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.suptitle('Zero-Order Hold Interpolation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 10: Linear Interpolation
        fig10 = plt.figure(figsize=(15, 10))
        linear_recon = self.linear_interpolation(signal_with_gaps, missing_indices)

        plt.subplot(2, 2, 1)
        plt.plot(self.t[:800], test_signal[:800], 'g-', label='Original Signal', linewidth=2)
        plt.plot(self.t[:800][valid_mask[:800]], signal_with_gaps[:800][valid_mask[:800]], 'ro',
                 markersize=3, label='Available Samples')
        plt.title('Signal with Missing Samples')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.t[:800], test_signal[:800], 'g-', label='Original Signal', linewidth=2, alpha=0.7)
        plt.plot(self.t[:800], linear_recon[:800], 'b-', label='Linear Interpolation', linewidth=2)
        plt.title('Linear Interpolation Result')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        error_linear = np.abs(linear_recon - test_signal)
        plt.plot(self.t[:800], error_linear[:800], 'r-', linewidth=2)
        plt.title('Reconstruction Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        # Zoom in to show linear behavior
        plt.plot(self.t[zoom_start:zoom_end], test_signal[zoom_start:zoom_end], 'g-',
                 label='Original', linewidth=2)
        plt.plot(self.t[zoom_start:zoom_end], linear_recon[zoom_start:zoom_end], 'b-',
                 label='Linear', linewidth=2)
        plt.title('Detailed View (Linear Behavior)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.suptitle('Linear Interpolation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 11: Polynomial Interpolation
        fig11 = plt.figure(figsize=(15, 10))
        poly_recon = self.polynomial_interpolation(signal_with_gaps, missing_indices[:40], degree=5)

        plt.subplot(2, 2, 1)
        plt.plot(self.t[:800], test_signal[:800], 'g-', label='Original Signal', linewidth=2)
        reduced_missing = missing_indices[:40]
        reduced_valid_mask = np.ones(len(test_signal), dtype=bool)
        reduced_valid_mask[reduced_missing] = False
        plt.plot(self.t[:800][reduced_valid_mask[:800]], test_signal[:800][reduced_valid_mask[:800]], 'ro',
                 markersize=3, label='Available Samples')
        plt.title('Signal with Reduced Missing Samples')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.t[:800], test_signal[:800], 'g-', label='Original Signal', linewidth=2, alpha=0.7)
        plt.plot(self.t[:800], poly_recon[:800], 'b-', label='Polynomial Interpolation', linewidth=2)
        plt.title('Polynomial Interpolation Result')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        error_poly = np.abs(poly_recon - test_signal)
        plt.plot(self.t[:800], error_poly[:800], 'r-', linewidth=2)
        plt.title('Reconstruction Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        # Show potential oscillation issues
        plt.plot(self.t[zoom_start:zoom_end], test_signal[zoom_start:zoom_end], 'g-',
                 label='Original', linewidth=2)
        plt.plot(self.t[zoom_start:zoom_end], poly_recon[zoom_start:zoom_end], 'b-',
                 label='Polynomial', linewidth=2)
        plt.title('Detailed View (Potential Oscillations)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.suptitle('Polynomial Interpolation (Degree 5)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 12: Sinc Interpolation
        fig12 = plt.figure(figsize=(15, 10))
        sinc_recon = self.sinc_interpolation(signal_with_gaps, missing_indices[:30])

        plt.subplot(2, 2, 1)
        plt.plot(self.t[:800], test_signal[:800], 'g-', label='Original Signal', linewidth=2)
        sinc_missing = missing_indices[:30]
        sinc_valid_mask = np.ones(len(test_signal), dtype=bool)
        sinc_valid_mask[sinc_missing] = False
        plt.plot(self.t[:800][sinc_valid_mask[:800]], test_signal[:800][sinc_valid_mask[:800]], 'ro',
                 markersize=3, label='Available Samples')
        plt.title('Signal with Missing Samples')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.t[:800], test_signal[:800], 'g-', label='Original Signal', linewidth=2, alpha=0.7)
        plt.plot(self.t[:800], sinc_recon[:800], 'b-', label='Sinc Interpolation', linewidth=2)
        plt.title('Sinc Interpolation Result')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        # Show sinc function
        x = np.linspace(-10, 10, 200)
        sinc_func = np.sinc(x)
        plt.plot(x, sinc_func, 'b-', linewidth=2)
        plt.title('Sinc Function Kernel')
        plt.xlabel('Sample Offset')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        error_sinc = np.abs(sinc_recon - test_signal)
        plt.plot(self.t[:800], error_sinc[:800], 'r-', linewidth=2)
        plt.title('Reconstruction Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error')
        plt.grid(True)

        plt.suptitle('Sinc Interpolation (Band-Limited)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 13: AR Model Reconstruction
        fig13 = plt.figure(figsize=(15, 10))
        ar_recon = self.ar_reconstruction(signal_with_gaps, missing_indices[:50])

        plt.subplot(2, 2, 1)
        plt.plot(self.t[:800], test_signal[:800], 'g-', label='Original Signal', linewidth=2)
        ar_missing = missing_indices[:50]
        ar_valid_mask = np.ones(len(test_signal), dtype=bool)
        ar_valid_mask[ar_missing] = False
        plt.plot(self.t[:800][ar_valid_mask[:800]], test_signal[:800][ar_valid_mask[:800]], 'ro',
                 markersize=3, label='Available Samples')
        plt.title('Signal with Missing Samples')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.t[:800], test_signal[:800], 'g-', label='Original Signal', linewidth=2, alpha=0.7)
        plt.plot(self.t[:800], ar_recon[:800], 'b-', label='AR Reconstruction', linewidth=2)
        plt.title('AR Model Reconstruction Result')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        # Show AR coefficients
        valid_signal = test_signal[ar_valid_mask]
        if len(valid_signal) > 10:
            ar_coeffs = self._estimate_ar_coefficients(valid_signal, 10)
            plt.stem(range(len(ar_coeffs)), ar_coeffs, basefmt='b-')
            plt.title('Estimated AR Coefficients')
            plt.xlabel('Coefficient Index')
            plt.ylabel('Value')
            plt.grid(True)

        plt.subplot(2, 2, 4)
        error_ar = np.abs(ar_recon - test_signal)
        plt.plot(self.t[:800], error_ar[:800], 'r-', linewidth=2)
        plt.title('Reconstruction Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Absolute Error')
        plt.grid(True)

        plt.suptitle('Autoregressive (AR) Model Reconstruction', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 14: Compressed Sensing
        fig14 = plt.figure(figsize=(15, 10))

        N = 256
        M = 128  # Measurements
        measurement_matrix = np.random.randn(M, N) / np.sqrt(M)
        sparse_signal = np.zeros(N)
        sparse_signal[[50, 100, 150, 200]] = [1, -0.8, 0.6, -0.4]  # Sparse signal
        measurements = measurement_matrix @ sparse_signal

        recovered = self.compressed_sensing_recovery(measurements, measurement_matrix)

        plt.subplot(2, 2, 1)
        plt.stem(range(N), sparse_signal, linefmt='g-', markerfmt='go', basefmt='g-')
        plt.title('Original Sparse Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(measurements, 'r-', linewidth=2)
        plt.title(f'Compressed Measurements (M={M})')
        plt.xlabel('Measurement Index')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.stem(range(N), recovered, linefmt='b-', markerfmt='bs', basefmt='b-')
        plt.title('Compressed Sensing Recovery')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        markerline1, stemlines1, baseline1 = plt.stem(range(N), sparse_signal, linefmt='g-', markerfmt='go',
                                                      basefmt='g-', label='Original')
        markerline1.set_alpha(0.7)
        stemlines1.set_alpha(0.7)

        markerline2, stemlines2, baseline2 = plt.stem(range(N), recovered, linefmt='b-', markerfmt='bs', basefmt='b-',
                                                      label='Recovered')
        markerline2.set_alpha(0.7)
        stemlines2.set_alpha(0.7)
        plt.title('Comparison: Original vs Recovered')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.suptitle('Compressed Sensing Recovery', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 15: CLEAN Algorithm
        fig15 = plt.figure(figsize=(15, 10))

        psf = np.exp(-np.arange(20) ** 2 / 10)  # Gaussian PSF
        psf = psf / np.sum(psf)
        dirty = np.convolve(sparse_signal[:100], psf, mode='same')

        reconstructed_clean, components = self.clean_algorithm(dirty, psf)

        plt.subplot(2, 2, 1)
        plt.plot(sparse_signal[:100], 'g-', linewidth=2, label='True Signal')
        plt.title('True Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(dirty, 'r-', linewidth=2, label='Dirty Signal (Convolved)')
        plt.title('Dirty Signal (PSF Convolved)')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.stem(range(len(components)), components, linefmt='b-', markerfmt='bo', basefmt='b-')
        plt.title('CLEAN Components')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(sparse_signal[:100], 'g-', linewidth=2, alpha=0.7, label='True Signal')
        plt.plot(dirty, 'r-', linewidth=2, alpha=0.7, label='Dirty Signal')
        plt.plot(reconstructed_clean, 'b-', linewidth=2, label='CLEAN Reconstruction')
        plt.title('CLEAN Algorithm Result')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.suptitle('CLEAN Algorithm for Deconvolution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 16: Performance Comparison
        fig16 = plt.figure(figsize=(15, 10))

        interp_methods = ['ZOH', 'Linear', 'Polynomial', 'Sinc', 'AR']
        recon_signals = [zoh_recon, linear_recon, poly_recon, sinc_recon, ar_recon]

        mse_values = []
        for recon in recon_signals:
            mse = np.mean((recon - test_signal) ** 2)
            mse_values.append(mse)

        plt.subplot(2, 2, 1)
        plt.bar(interp_methods, mse_values, color=['red', 'blue', 'green', 'orange', 'purple'])
        plt.title('Mean Squared Error Comparison')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.grid(True)

        # Computational complexity (relative)
        complexity_scores = [1, 2, 5, 8, 6]  # Relative complexity
        plt.subplot(2, 2, 2)
        plt.bar(interp_methods, complexity_scores, color=['red', 'blue', 'green', 'orange', 'purple'])
        plt.title('Computational Complexity')
        plt.ylabel('Relative Complexity Score')
        plt.grid(True)

        # Frequency response comparison
        plt.subplot(2, 2, 3)
        for i, (method, recon) in enumerate(zip(interp_methods, recon_signals)):
            freqs = np.fft.fftfreq(len(recon), 1 / self.fs)[:len(recon) // 2]
            recon_fft = np.abs(np.fft.fft(recon))[:len(recon) // 2]
            plt.loglog(freqs, recon_fft, label=method, alpha=0.8)

        # Original signal
        clean_fft = np.abs(np.fft.fft(test_signal))[:len(test_signal) // 2]
        plt.loglog(freqs, clean_fft, 'k-', linewidth=3, label='Original', alpha=0.7)

        plt.title('Frequency Domain Comparison')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)

        # Summary table
        plt.subplot(2, 2, 4)
        plt.axis('off')

        # Create performance summary
        summary_text = f"""
        Performance Summary:

        Best MSE: {interp_methods[np.argmin(mse_values)]} ({mse_values[np.argmin(mse_values)]:.2e})
        Fastest: {interp_methods[np.argmin(complexity_scores)]}
        Most Accurate: {interp_methods[np.argmin(mse_values)]}

        Method Characteristics:
        • ZOH: Simple, introduces steps
        • Linear: Smooth, limited bandwidth
        • Polynomial: High-order, may oscillate
        • Sinc: Ideal reconstruction, band-limited
        • AR: Model-based, good for predictable signals
        """

        plt.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))

        plt.suptitle('Interpolation Methods Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return {
            'zoh_recon': zoh_recon,
            'linear_recon': linear_recon,
            'poly_recon': poly_recon,
            'sinc_recon': sinc_recon,
            'ar_recon': ar_recon,
            'cs_recovery': recovered,
            'clean_reconstruction': reconstructed_clean,
            'mse_values': mse_values
        }

    def demonstrate_all_methods(self):
        """
        Run all demonstrations with separate figures for each category
        """
        print("=== Time-Domain Signal Enhancement and Recovery Simulation ===\n")

        print("1. Demonstrating Noise Suppression Techniques...")
        noise_results = self.plot_noise_suppression_methods()
        print("   ✓ Spectral Subtraction")
        print("   ✓ Blind Deconvolution")
        print("   ✓ Homomorphic Filtering")
        print("   ✓ MAP Estimation")
        print("   ✓ Wavelet Denoising")

        print("\n2. Demonstrating Blind Signal Separation...")
        bss_results = self.plot_blind_signal_separation()
        print("   ✓ Independent Component Analysis (ICA)")
        print("   ✓ Matching Pursuit")
        print("   ✓ Non-negative Matrix Factorization (NMF)")

        print("\n3. Demonstrating Signal Reconstruction and Interpolation...")
        reconstruction_results = self.plot_signal_reconstruction()
        print("   ✓ Zero-Order Hold")
        print("   ✓ Linear Interpolation")
        print("   ✓ Polynomial Interpolation")
        print("   ✓ Sinc Interpolation")
        print("   ✓ AR Model Reconstruction")
        print("   ✓ Compressed Sensing")
        print("   ✓ CLEAN Algorithm")

        # Create final summary figure
        self.plot_final_summary(noise_results, bss_results, reconstruction_results)

        print("\n=== Simulation Complete ===")
        print("All methods have been successfully implemented and demonstrated!")

        return {
            'noise_suppression': noise_results,
            'blind_separation': bss_results,
            'reconstruction': reconstruction_results
        }

    def plot_final_summary(self, noise_results, bss_results, reconstruction_results):
        """Create a comprehensive summary figure"""

        fig_summary = plt.figure(figsize=(20, 12))

        # Overall performance metrics
        plt.subplot(2, 4, 1)
        noise_methods = ['Spectral Sub.', 'Blind Deconv.', 'Homomorphic', 'MAP', 'Wavelet']

        # Calculate SNR improvements for noise suppression methods
        clean, noisy, _, _ = self.generate_test_signals()
        snr_improvements = []

        for method_name, enhanced in zip(noise_methods, [
            noise_results['enhanced_spectral'],
            noise_results['deconvolved'],
            noise_results['homomorphic'],
            noise_results['map_enhanced'],
            noise_results['wavelet_denoised']
        ]):
            noise_power_original = np.var(noisy - clean)
            noise_power_enhanced = np.var(enhanced - clean)
            snr_improvement = 10 * np.log10(noise_power_original / (noise_power_enhanced + 1e-10))
            snr_improvements.append(max(0, snr_improvement))

        plt.bar(noise_methods, snr_improvements, color=['red', 'blue', 'green', 'orange', 'purple'])
        plt.title('Noise Suppression: SNR Improvement')
        plt.ylabel('SNR Improvement (dB)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Interpolation MSE comparison
        plt.subplot(2, 4, 2)
        interp_methods = ['ZOH', 'Linear', 'Polynomial', 'Sinc', 'AR']
        plt.bar(interp_methods, reconstruction_results['mse_values'],
                color=['red', 'blue', 'green', 'orange', 'purple'])
        plt.title('Interpolation: Mean Squared Error')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Computational complexity comparison
        plt.subplot(2, 4, 3)
        all_methods = noise_methods + interp_methods
        complexity_scores = [3, 8, 4, 6, 7, 1, 2, 5, 8, 6]  # Relative complexity

        colors = ['lightcoral'] * 5 + ['lightblue'] * 5
        plt.bar(all_methods, complexity_scores, color=colors)
        plt.title('Computational Complexity')
        plt.ylabel('Relative Complexity Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Method characteristics radar chart (simplified)
        plt.subplot(2, 4, 4)
        categories = ['Accuracy', 'Speed', 'Robustness', 'Simplicity']

        # Example scores for different method types
        noise_suppression_scores = [7, 5, 6, 4]
        interpolation_scores = [8, 8, 7, 8]
        separation_scores = [6, 4, 5, 3]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        noise_suppression_scores += noise_suppression_scores[:1]
        interpolation_scores += interpolation_scores[:1]
        separation_scores += separation_scores[:1]

        plt.plot(angles, noise_suppression_scores, 'o-', linewidth=2, label='Noise Suppression')
        plt.plot(angles, interpolation_scores, 's-', linewidth=2, label='Interpolation')
        plt.plot(angles, separation_scores, '^-', linewidth=2, label='Signal Separation')

        plt.fill(angles, noise_suppression_scores, alpha=0.25)
        plt.fill(angles, interpolation_scores, alpha=0.25)
        plt.fill(angles, separation_scores, alpha=0.25)

        plt.xticks(angles[:-1], categories)
        plt.ylim(0, 10)
        plt.title('Method Category Performance')
        plt.legend()
        plt.grid(True)

        # Time-frequency analysis comparison
        plt.subplot(2, 4, 5)
        f, t_spec, Sxx_clean = signal.spectrogram(clean, self.fs, nperseg=128)
        plt.imshow(10 * np.log10(np.abs(Sxx_clean) + 1e-10), aspect='auto', origin='lower',
                   extent=[0, 2, 0, 500])
        plt.title('Clean Signal Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Power (dB)')

        plt.subplot(2, 4, 6)
        f, t_spec, Sxx_enhanced = signal.spectrogram(noise_results['enhanced_spectral'], self.fs, nperseg=128)
        plt.imshow(10 * np.log10(np.abs(Sxx_enhanced) + 1e-10), aspect='auto', origin='lower',
                   extent=[0, 2, 0, 500])
        plt.title('Enhanced Signal Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(label='Power (dB)')

        # Signal separation quality
        plt.subplot(2, 4, 7)
        sources, mixed, A = self.generate_mixed_signals()
        separated, _ = self.ica_time_domain(mixed)

        # Calculate separation quality (simplified correlation measure)
        correlation_matrix = np.abs(np.corrcoef(sources))
        separation_matrix = np.abs(np.corrcoef(separated))

        plt.imshow(correlation_matrix, cmap='Blues', vmin=0, vmax=1)
        plt.title('Original Sources Correlation')
        plt.xlabel('Source Index')
        plt.ylabel('Source Index')
        plt.colorbar()

        # Algorithm timeline/evolution
        plt.subplot(2, 4, 8)
        methods_timeline = {
            'Spectral Subtraction': 1979,
            'Wiener Filtering': 1942,
            'Homomorphic': 1965,
            'Wavelet Denoising': 1988,
            'ICA': 1994,
            'Compressed Sensing': 2006,
            'Deep Learning': 2012
        }

        years = list(methods_timeline.values())
        methods = list(methods_timeline.keys())

        plt.scatter(years, range(len(years)), s=100, c=range(len(years)), cmap='viridis')
        for i, (method, year) in enumerate(methods_timeline.items()):
            plt.annotate(method, (year, i), xytext=(5, 0), textcoords='offset points',
                         fontsize=8, va='center')

        plt.xlabel('Year')
        plt.title('Algorithm Development Timeline')
        plt.yticks([])
        plt.grid(True, alpha=0.3)

        plt.suptitle('Comprehensive Summary: Time-Domain Signal Enhancement & Recovery',
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

        # Create detailed performance table
        self.create_performance_table(snr_improvements, reconstruction_results['mse_values'])

    def create_performance_table(self, snr_improvements, mse_values):
        """Create a detailed performance comparison table"""

        fig_table = plt.figure(figsize=(16, 10))

        # Noise suppression methods table
        plt.subplot(2, 2, 1)
        plt.axis('tight')
        plt.axis('off')

        noise_methods = ['Spectral Subtraction', 'Blind Deconvolution', 'Homomorphic', 'MAP Estimation',
                         'Wavelet Denoising']

        table_data = []
        for i, method in enumerate(noise_methods):
            complexity = ['Low', 'High', 'Medium', 'Medium', 'High'][i]
            robustness = ['Medium', 'Low', 'High', 'High', 'High'][i]
            snr_val = f"{snr_improvements[i]:.2f} dB"
            table_data.append([method, snr_val, complexity, robustness])

        table1 = plt.table(cellText=table_data,
                           colLabels=['Method', 'SNR Improvement', 'Complexity', 'Robustness'],
                           cellLoc='center',
                           loc='center',
                           colColours=['lightblue'] * 4)
        table1.auto_set_font_size(False)
        table1.set_fontsize(9)
        table1.scale(1, 2)
        plt.title('Noise Suppression Methods Performance', fontweight='bold', pad=20)

        # Interpolation methods table
        plt.subplot(2, 2, 2)
        plt.axis('tight')
        plt.axis('off')

        interp_methods = ['Zero-Order Hold', 'Linear Interpolation', 'Polynomial', 'Sinc Interpolation', 'AR Model']

        table_data2 = []
        for i, method in enumerate(interp_methods):
            complexity = ['Very Low', 'Low', 'High', 'Very High', 'High'][i]
            accuracy = ['Low', 'Medium', 'High*', 'Very High', 'Medium'][i]
            mse_val = f"{mse_values[i]:.2e}"
            table_data2.append([method, mse_val, complexity, accuracy])

        table2 = plt.table(cellText=table_data2,
                           colLabels=['Method', 'MSE', 'Complexity', 'Accuracy'],
                           cellLoc='center',
                           loc='center',
                           colColours=['lightgreen'] * 4)
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1, 2)
        plt.title('Interpolation Methods Performance', fontweight='bold', pad=20)

        # Applications and use cases
        plt.subplot(2, 2, 3)
        plt.axis('tight')
        plt.axis('off')

        applications = [
            ['Audio Processing', 'Spectral Sub., Wavelet', 'Real-time noise reduction'],
            ['Medical Imaging', 'Blind Deconv., MAP', 'Image deblurring'],
            ['Communications', 'Homomorphic, ICA', 'Signal separation'],
            ['Radar/Sonar', 'CLEAN, Compressed Sensing', 'High-resolution imaging'],
            ['Speech Processing', 'AR Model, NMF', 'Voice enhancement'],
            ['Seismic Analysis', 'Wavelet, Matching Pursuit', 'Signal decomposition']
        ]

        table3 = plt.table(cellText=applications,
                           colLabels=['Application Domain', 'Preferred Methods', 'Primary Use'],
                           cellLoc='center',
                           loc='center',
                           colColours=['lightyellow'] * 3)
        table3.auto_set_font_size(False)
        table3.set_fontsize(9)
        table3.scale(1, 2)
        plt.title('Application Domains and Methods', fontweight='bold', pad=20)

        # Method selection guidelines
        plt.subplot(2, 2, 4)
        plt.axis('tight')
        plt.axis('off')

        guidelines_text = """
        METHOD SELECTION GUIDELINES

        For Noise Suppression:
        • Real-time: Spectral Subtraction
        • High quality: Wavelet Denoising
        • Unknown noise: MAP Estimation
        • Multiplicative noise: Homomorphic

        For Signal Reconstruction:
        • Simple/fast: Zero-Order Hold
        • Smooth signals: Linear Interpolation  
        • Band-limited: Sinc Interpolation
        • Predictable patterns: AR Model
        • Sparse signals: Compressed Sensing

        For Signal Separation:
        • Independent sources: ICA
        • Sparse representation: Matching Pursuit
        • Non-negative data: NMF
        • Complex mixing: Deep Learning

        Performance Trade-offs:
        • Accuracy ↔ Computational Cost
        • Robustness ↔ Specialization
        • Real-time ↔ Quality
        """

        plt.text(0.05, 0.95, guidelines_text, fontsize=10, verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))

        plt.suptitle('Performance Analysis and Selection Guidelines',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


# Main execution
if __name__ == "__main__":
    enhancer = TimeSignalEnhancement()
    results = enhancer.demonstrate_all_methods()
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print("Total methods implemented: 15")
    print("Figures generated: 16")
    print("Categories covered: 3")
    print("  - Noise Suppression: 5 methods")
    print("  - Blind Signal Separation: 4 methods")
    print("  - Signal Reconstruction: 7 methods")
    print("=" * 60)
