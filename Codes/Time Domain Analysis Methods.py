import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from scipy.fftpack import hilbert
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import librosa


class TimeDomainAnalysis:
    def __init__(self, signal_data, sampling_rate):
        self.signal = signal_data
        self.fs = sampling_rate

    def statistical_characteristics(self):
        mean_value = np.mean(self.signal)

        variance = np.var(self.signal)
        power = np.mean(self.signal ** 2)

        skewness = stats.skew(self.signal)
        kurtosis = stats.kurtosis(self.signal)

        hist, bin_edges = np.histogram(self.signal, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
        kde.fit(self.signal.reshape(-1, 1))
        x_grid = np.linspace(min(self.signal), max(self.signal), 1000).reshape(-1, 1)
        kde_estimate = np.exp(kde.score_samples(x_grid))

        hist_prob = hist / np.sum(hist)
        entropy = -np.sum(hist_prob * np.log2(hist_prob + 1e-10))

        return {
            "mean": mean_value,
            "variance": variance,
            "power": power,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "histogram": (bin_centers, hist),
            "kde": (x_grid.flatten(), kde_estimate),
            "entropy": entropy
        }

    def instantaneous_characteristics(self):
        zcr = np.sum(np.abs(np.diff(np.sign(self.signal)))) / (2 * len(self.signal))

        analytic_signal = hilbert(self.signal)
        hilbert_envelope = np.abs(analytic_signal)

        rectified = np.abs(self.signal)
        b, a = signal.butter(3, 0.05)
        rectified_smoothed = signal.filtfilt(b, a, rectified)

        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.fs

        frame_length = int(0.025 * self.fs)
        hop_length = int(0.010 * self.fs)
        energy_contour = []

        for i in range(0, len(self.signal) - frame_length, hop_length):
            frame = self.signal[i:i + frame_length]
            energy = np.sum(frame ** 2)
            energy_contour.append(energy)

        return {
            "zcr": zcr,
            "hilbert_envelope": hilbert_envelope,
            "rectified_envelope": rectified_smoothed,
            "instantaneous_frequency": instantaneous_frequency,
            "energy_contour": np.array(energy_contour)
        }

    def linear_prediction_analysis(self, order=10):
        autocorr = np.correlate(self.signal, self.signal, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]

        lpc_coeffs = self._levinson_durbin(autocorr[:order + 1], order)

        N = len(self.signal)
        X = np.zeros((N - order, order))
        for i in range(order):
            X[:, i] = self.signal[order - i - 1:N - i - 1]
        y = self.signal[order:]
        cov_coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

        return {
            "lpc_coeffs": lpc_coeffs,
            "cov_coeffs": cov_coeffs
        }

    def _levinson_durbin(self, r, order):
        a = np.zeros(order + 1)
        e = r[0]
        a[0] = 1.0

        for i in range(1, order + 1):
            k = 0.0
            for j in range(1, i):
                k -= a[j] * r[i - j]
            k = (r[i] + k) / e

            a_new = np.zeros(order + 1)
            a_new[0] = 1.0
            for j in range(1, i):
                a_new[j] = a[j] + k * a[i - j]
            a_new[i] = k
            a = a_new

            e = e * (1 - k * k)

        return a[1:]

    def endpoint_detection(self, threshold_factor=0.3, min_zcr=0.1):
        frame_length = int(0.025 * self.fs)
        hop_length = int(0.010 * self.fs)
        energy = np.array([np.sum(self.signal[i:i + frame_length] ** 2)
                           for i in range(0, len(self.signal) - frame_length, hop_length)])

        zcr = np.array([np.sum(np.abs(np.diff(np.sign(self.signal[i:i + frame_length])))) / (2 * frame_length)
                        for i in range(0, len(self.signal) - frame_length, hop_length)])

        energy_threshold = threshold_factor * np.max(energy)
        energy_endpoints = np.where(energy > energy_threshold)[0]

        zcr_endpoints = np.where(zcr > min_zcr)[0]
        combined_endpoints = np.intersect1d(energy_endpoints, zcr_endpoints)

        noise_floor = np.mean(energy[:10])
        adaptive_threshold = noise_floor * 2
        adaptive_endpoints = np.where(energy > adaptive_threshold)[0]

        return {
            "energy": energy,
            "zcr": zcr,
            "energy_endpoints": energy_endpoints,
            "combined_endpoints": combined_endpoints,
            "adaptive_endpoints": adaptive_endpoints
        }

    def time_domain_features(self):
        frame_length = int(0.025 * self.fs)
        hop_length = int(0.010 * self.fs)
        energy = np.array([np.sum(self.signal[i:i + frame_length] ** 2)
                           for i in range(0, len(self.signal) - frame_length, hop_length)])

        max_lag = int(0.02 * self.fs)
        amdf = np.zeros(max_lag)
        for lag in range(1, max_lag):
            amdf[lag] = np.mean(np.abs(self.signal[:-lag] - self.signal[lag:]))
        amdf[0] = 0

        teo = self.signal[1:-1] ** 2 - self.signal[:-2] * self.signal[2:]

        return {
            "energy": energy,
            "amdf": amdf,
            "teo": teo
        }


def plot_original_signal(signal_data, fs):
    time = np.arange(len(signal_data)) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(time, signal_data)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_statistical_characteristics(signal_data, fs, stats_results):
    time = np.arange(len(signal_data)) / fs

    # Plot 1: Statistical Distribution
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    bin_centers, hist = stats_results["histogram"]
    x_grid, kde = stats_results["kde"]
    plt.bar(bin_centers, hist, alpha=0.5, width=(bin_centers[1] - bin_centers[0]))
    plt.plot(x_grid, kde, 'r-', linewidth=2)
    plt.title('Probability Distribution')
    plt.xlabel('Amplitude')
    plt.ylabel('Probability')
    plt.grid(True)

    # Print statistical metrics
    plt.figtext(0.15, 0.85,
                f"Mean: {stats_results['mean']:.4f}\n"
                f"Variance: {stats_results['variance']:.4f}\n"
                f"Power: {stats_results['power']:.4f}\n"
                f"Skewness: {stats_results['skewness']:.4f}\n"
                f"Kurtosis: {stats_results['kurtosis']:.4f}\n"
                f"Entropy: {stats_results['entropy']:.4f}",
                bbox=dict(facecolor='white', alpha=0.8))

    # Plot 2: Signal with mean
    plt.subplot(2, 1, 2)
    plt.plot(time, signal_data, 'b-', label='Signal')
    plt.axhline(y=stats_results['mean'], color='r', linestyle='-', label=f"Mean: {stats_results['mean']:.4f}")
    plt.title('Signal with Mean Value')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_instantaneous_characteristics(signal_data, fs, instant_results):
    time = np.arange(len(signal_data)) / fs

    # Figure 1: Envelopes
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time, signal_data, 'b-', alpha=0.5, label='Signal')
    plt.plot(time, instant_results["hilbert_envelope"], 'r-', linewidth=2, label='Hilbert Envelope')
    plt.plot(time, instant_results["rectified_envelope"], 'g-', linewidth=2, label='Rectified-Smoothed Envelope')
    plt.title('Signal Envelopes')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    inst_freq = instant_results["instantaneous_frequency"]
    plt.plot(time[:-1], inst_freq)
    plt.title(f'Instantaneous Frequency (Zero-Crossing Rate: {instant_results["zcr"]:.4f})')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim([0, min(2000, fs / 2)])  # Limit to reasonable frequency range
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Figure 2: Energy Contour
    plt.figure(figsize=(10, 4))
    energy_contour = instant_results["energy_contour"]
    frame_time = np.arange(len(energy_contour)) * 0.01  # 10ms hop
    plt.plot(frame_time, energy_contour)
    plt.title('Energy Contour')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_linear_prediction(signal_data, fs, lpc_results):
    plt.figure(figsize=(12, 8))

    # Plot 1: LPC coefficients
    plt.subplot(2, 1, 1)
    lpc_coeffs = lpc_results["lpc_coeffs"]
    cov_coeffs = lpc_results["cov_coeffs"]
    lpc_indices = np.arange(1, len(lpc_coeffs) + 1)

    plt.stem(lpc_indices, lpc_coeffs, 'b', markerfmt='bo', label='Autocorrelation Method')
    plt.stem(lpc_indices, cov_coeffs, 'r', markerfmt='ro', label='Covariance Method')
    plt.title('LPC Coefficients')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Plot 2: LPC Spectrum
    plt.subplot(2, 1, 2)

    # Calculate spectra
    w, h_autocorr = signal.freqz([1], np.concatenate(([1], lpc_coeffs)), worN=2000)
    w, h_cov = signal.freqz([1], np.concatenate(([1], cov_coeffs)), worN=2000)

    # Plot spectrum
    freq = w * fs / (2 * np.pi)
    plt.plot(freq, 20 * np.log10(np.abs(h_autocorr)), 'b-', label='Autocorrelation Method')
    plt.plot(freq, 20 * np.log10(np.abs(h_cov)), 'r-', label='Covariance Method')
    plt.title('LPC Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.xlim([0, fs / 2])

    plt.tight_layout()
    plt.show()


def plot_endpoint_detection(signal_data, fs, endpoint_results):
    time = np.arange(len(signal_data)) / fs

    plt.figure(figsize=(12, 10))

    # Plot 1: Signal
    plt.subplot(3, 1, 1)
    plt.plot(time, signal_data)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot 2: Energy with Endpoints
    plt.subplot(3, 1, 2)
    energy = endpoint_results["energy"]
    frame_time = np.arange(len(energy)) * 0.01  # 10ms hop
    energy_endpoints = endpoint_results["energy_endpoints"]
    adaptive_endpoints = endpoint_results["adaptive_endpoints"]

    plt.plot(frame_time, energy, 'b-', label='Energy')
    plt.scatter(frame_time[energy_endpoints], energy[energy_endpoints], c='r', s=30, label='Fixed Threshold')
    plt.scatter(frame_time[adaptive_endpoints], energy[adaptive_endpoints], c='g', s=20, alpha=0.5,
                label='Adaptive Threshold')
    plt.title('Endpoint Detection - Energy Method')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)

    # Plot 3: Zero Crossing Rate with Combined Endpoints
    plt.subplot(3, 1, 3)
    zcr = endpoint_results["zcr"]
    combined_endpoints = endpoint_results["combined_endpoints"]

    plt.plot(frame_time, zcr, 'b-', label='Zero Crossing Rate')
    plt.scatter(frame_time[combined_endpoints], zcr[combined_endpoints], c='r', s=30, label='Combined Detection')
    plt.title('Endpoint Detection - ZCR Method')
    plt.xlabel('Time (s)')
    plt.ylabel('ZCR')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_time_domain_features(signal_data, fs, feature_results):
    time = np.arange(len(signal_data)) / fs

    plt.figure(figsize=(12, 8))

    # Plot 1: AMDF
    plt.subplot(2, 1, 1)
    amdf = feature_results["amdf"]
    lag_time = np.arange(len(amdf)) / fs * 1000  # Convert to ms
    plt.plot(lag_time, amdf)
    # Find potential pitch period (minimum after initial values)
    if len(amdf) > 10:
        pitch_lag = np.argmin(amdf[10:]) + 10
        pitch_period_ms = pitch_lag / fs * 1000
        plt.axvline(x=pitch_period_ms, color='r', linestyle='--',
                    label=f'Potential Pitch Period: {pitch_period_ms:.2f} ms')
        plt.legend()
    plt.title('Average Magnitude Difference Function (AMDF)')
    plt.xlabel('Lag (ms)')
    plt.ylabel('AMDF')
    plt.grid(True)

    # Plot 2: TEO
    plt.subplot(2, 1, 2)
    teo = feature_results["teo"]
    teo_time = time[1:-1]  # TEO has 2 fewer samples
    plt.plot(teo_time, teo)
    plt.title('Teager Energy Operator (TEO)')
    plt.xlabel('Time (s)')
    plt.ylabel('TEO Value')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def demonstrate_time_domain_analysis():
    t = np.linspace(0, 1, 8000)
    fs = 8000

    # Generate example signal: speech-like with changing amplitude and frequency
    signal_data = np.sin(2 * np.pi * 10 * t) * np.exp(-3 * t)  # Decaying sinusoid
    signal_data += 0.5 * np.sin(2 * np.pi * 20 * t) * (1 - np.exp(-3 * t))  # Growing sinusoid
    signal_data += 0.3 * np.random.randn(len(t))  # Add noise

    # Add a silence period and then another sound
    silence = np.random.randn(2000) * 0.05
    second_part = np.sin(2 * np.pi * 30 * t[:2000]) * 0.8
    full_signal = np.concatenate([signal_data, silence, second_part])

    analyzer = TimeDomainAnalysis(full_signal, fs)

    # Get all analysis results
    stats_results = analyzer.statistical_characteristics()
    instant_results = analyzer.instantaneous_characteristics()
    lpc_results = analyzer.linear_prediction_analysis(order=12)
    endpoint_results = analyzer.endpoint_detection()
    feature_results = analyzer.time_domain_features()

    # Plot results by category
    plot_original_signal(full_signal, fs)
    plot_statistical_characteristics(full_signal, fs, stats_results)
    plot_instantaneous_characteristics(full_signal, fs, instant_results)
    plot_linear_prediction(full_signal, fs, lpc_results)
    plot_endpoint_detection(full_signal, fs, endpoint_results)
    plot_time_domain_features(full_signal, fs, feature_results)

    return {
        "statistics": stats_results,
        "instantaneous": instant_results,
        "lpc": lpc_results,
        "endpoints": endpoint_results,
        "features": feature_results
    }


# Run the demonstration
results = demonstrate_time_domain_analysis()