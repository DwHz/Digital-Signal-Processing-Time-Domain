import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math


# Generate a test signal with noise for demonstration
def generate_test_signal(fs=1000, duration=1):
    t = np.arange(0, duration, 1 / fs)
    # Clean signal: sum of two sinusoids
    clean_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + 2 * np.sin(2 * np.pi * 80 * t)
    # Add noise: white noise and impulse noise
    white_noise = 0.6 * np.random.randn(len(t))
    impulse_noise = np.zeros_like(t)
    impulse_indices = np.random.randint(0, len(t), size=int(len(t) * 0.01))
    impulse_noise[impulse_indices] = np.random.randn(len(impulse_indices)) * 2
    noisy_signal = clean_signal + white_noise + impulse_noise
    return t, clean_signal, noisy_signal


# 1. FIR Filtering Techniques
class FIRFilter:
    def direct_form(self, x, b):
        y = np.zeros(len(x))
        N = len(b)
        for n in range(len(x)):
            for k in range(N):
                if n - k >= 0:
                    y[n] += b[k] * x[n - k]
        return y

    def window_method(self, N, fc, fs, window='hamming'):
        nyquist = fs / 2
        cutoff = fc / nyquist
        if window == 'hamming':
            window_func = np.hamming(N)
        elif window == 'blackman':
            window_func = np.blackman(N)
        elif window == 'hanning':
            window_func = np.hanning(N)
        else:
            window_func = np.ones(N)

        h_ideal = np.zeros(N)
        n = np.arange(N)
        middle = (N - 1) / 2
        for i in range(N):
            if i == middle:
                h_ideal[i] = cutoff
            else:
                h_ideal[i] = np.sin(np.pi * cutoff * (i - middle)) / (np.pi * (i - middle))

        h = h_ideal * window_func
        return h

    def frequency_sampling(self, N, cutoff_freq, fs):
        # Define the desired frequency response
        freq_resp = np.zeros(N)
        for i in range(N):
            if i <= N * cutoff_freq / fs or i >= N - N * cutoff_freq / fs:
                freq_resp[i] = 1

        # Compute the impulse response using IFFT
        h = np.real(np.fft.ifft(freq_resp))
        # Circular shift to get causal filter
        h = np.roll(h, N // 2)
        # Apply window to reduce Gibbs phenomenon
        h = h * np.hamming(N)
        return h

    def parks_mcclellan(self, N, bands, desired, fs):
        return signal.remez(N, bands, desired, fs=fs)

    def linear_phase_structure(self, b):
        N = len(b)
        is_symmetric = True
        for i in range(N // 2):
            if abs(b[i] - b[N - 1 - i]) > 1e-10:
                is_symmetric = False
                break

        if is_symmetric:
            # Type I (N odd) or Type II (N even) symmetric
            return "Linear phase filter with symmetric coefficients"

        is_antisymmetric = True
        for i in range(N // 2):
            if abs(b[i] + b[N - 1 - i]) > 1e-10:
                is_antisymmetric = False
                break

        if is_antisymmetric:
            # Type III (N odd) or Type IV (N even) antisymmetric
            return "Linear phase filter with antisymmetric coefficients"

        return "Not a linear phase filter"


# 2. IIR Filtering Techniques
class IIRFilter:
    def direct_form_I(self, x, b, a):
        y = np.zeros(len(x))
        M = len(b)
        N = len(a)

        for n in range(len(x)):
            # Feed-forward part
            for k in range(M):
                if n - k >= 0:
                    y[n] += b[k] * x[n - k]

            # Feed-back part
            for k in range(1, N):
                if n - k >= 0:
                    y[n] -= a[k] * y[n - k]

            y[n] /= a[0]  # Normalize by a[0]

        return y

    def direct_form_II(self, x, b, a):
        w = np.zeros(max(len(a), len(b)))
        y = np.zeros_like(x)

        for n in range(len(x)):
            # Compute intermediate value w[n]
            w[0] = x[n]
            for i in range(1, len(a)):
                if i < len(w):
                    w[0] -= a[i] * w[i]
            w[0] /= a[0]

            # Compute output
            y[n] = 0
            for i in range(len(b)):
                if i < len(w):
                    y[n] += b[i] * w[i]

            # Update delay line
            for i in range(len(w) - 1, 0, -1):
                w[i] = w[i - 1]

        return y

    def butterworth_design(self, N, cutoff, fs, btype='low'):
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(N, normal_cutoff, btype=btype)
        return b, a

    def chebyshev_design(self, N, cutoff, fs, rp=1, btype='low', type=1):
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        if type == 1:
            b, a = signal.cheby1(N, rp, normal_cutoff, btype=btype)
        else:
            b, a = signal.cheby2(N, rp, normal_cutoff, btype=btype)
        return b, a

    def elliptic_design(self, N, cutoff, fs, rp=1, rs=80, btype='low'):
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.ellip(N, rp, rs, normal_cutoff, btype=btype)
        return b, a

    def cascade_form(self, b, a):
        z, p, k = signal.tf2zpk(b, a)
        sos = signal.zpk2sos(z, p, k)
        return sos

    def parallel_form(self, b, a):
        r, p, k = signal.residuez(b, a)
        return r, p, k

    def lattice_form(self, b, a):
        k = signal.levinson(a, b)[0]
        return k


# 3. Adaptive Filtering Techniques
class AdaptiveFilter:
    def lms_algorithm(self, x, d, N, mu):
        x = np.array(x)
        d = np.array(d)
        L = len(x)
        w = np.zeros(N)  # Initial filter weights
        y = np.zeros(L)  # Filter output
        e = np.zeros(L)  # Error signal

        # LMS algorithm
        for n in range(N, L):
            x_window = x[n - N:n][::-1]  # Reversed window of input samples
            y[n] = np.dot(w, x_window)  # Filter output
            e[n] = d[n] - y[n]  # Error
            w = w + mu * e[n] * x_window  # Update filter weights

        return y, e, w

    def rls_algorithm(self, x, d, N, lambda_=0.99, delta=1e-4):
        x = np.array(x)
        d = np.array(d)
        L = len(x)
        w = np.zeros(N)  # Initial filter weights
        P = np.eye(N) / delta  # Initial inverse correlation matrix
        y = np.zeros(L)  # Filter output
        e = np.zeros(L)  # Error signal

        # RLS algorithm
        for n in range(N, L):
            x_window = x[n - N:n][::-1]  # Reversed window of input samples

            # Compute gain vector
            P_x = np.dot(P, x_window)
            k = P_x / (lambda_ + np.dot(x_window, P_x))

            # Update filter output and error
            y[n] = np.dot(w, x_window)
            e[n] = d[n] - y[n]

            # Update filter weights
            w = w + k * e[n]

            # Update inverse correlation matrix
            P = (P - np.outer(k, np.dot(x_window, P))) / lambda_

        return y, e, w

    def kalman_filter_algorithm(self, x, d, N, process_var=1e-4, measure_var=1e-2):
        x = np.array(x)
        d = np.array(d)
        L = len(x)
        w = np.zeros(N)  # Initial state estimate (filter weights)
        P = np.eye(N)  # Initial error covariance
        Q = process_var * np.eye(N)  # Process noise covariance
        R = measure_var  # Measurement noise variance
        y = np.zeros(L)  # Filter output
        e = np.zeros(L)  # Error signal

        # Kalman filter algorithm
        for n in range(N, L):
            x_window = x[n - N:n][::-1]  # Reversed window of input samples

            # Prediction step
            P = P + Q

            # Update step
            S = np.dot(np.dot(x_window, P), x_window) + R
            K = np.dot(P, x_window) / S  # Kalman gain

            y[n] = np.dot(w, x_window)  # Predict measurement
            e[n] = d[n] - y[n]  # Measurement error

            w = w + K * e[n]  # Update state estimate
            P = P - np.outer(K, np.dot(x_window, P))  # Update error covariance

        return y, e, w


# 4. Statistical Filtering Methods
class StatisticalFilter:
    def wiener_filter(self, x, d, filter_length):
        # Estimate auto-correlation and cross-correlation
        R_xx = np.zeros((filter_length, filter_length))
        for i in range(filter_length):
            for j in range(filter_length):
                R_xx[i, j] = np.mean(x[:-filter_length] * np.roll(x[:-filter_length], j - i))

        r_dx = np.zeros(filter_length)
        for i in range(filter_length):
            r_dx[i] = np.mean(d[filter_length:] * x[filter_length - i - 1:-i - 1])

        # Solve Wiener-Hopf equation
        w = np.linalg.solve(R_xx, r_dx)

        # Apply filter
        y = np.zeros_like(x)
        for n in range(filter_length, len(x)):
            y[n] = np.dot(w, x[n - filter_length:n][::-1])

        return y, w

    def kalman_filter(self, z, initial_x, initial_P, F, H, Q, R):
        # z: measurements
        # F: state transition matrix
        # H: observation matrix
        # Q: process noise covariance
        # R: measurement noise covariance

        n = len(z)
        x_dim = F.shape[0]

        # Initialize estimates
        x = np.zeros((n, x_dim))
        P = np.zeros((n, x_dim, x_dim))

        x[0] = initial_x
        P[0] = initial_P

        for k in range(1, n):
            # Prediction
            x_pred = np.dot(F, x[k - 1])
            P_pred = np.dot(np.dot(F, P[k - 1]), F.T) + Q

            # Kalman gain
            S = np.dot(np.dot(H, P_pred), H.T) + R
            K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))

            # Update
            y = z[k] - np.dot(H, x_pred)  # Measurement residual
            x[k] = x_pred + np.dot(K, y)
            P[k] = P_pred - np.dot(np.dot(K, H), P_pred)

        return x

    def particle_filter(self, z, N_particles, f_pred, h_obs, q_sample, p_observe, initial_sample):
        # z: measurements
        # N_particles: number of particles
        # f_pred: prediction function
        # h_obs: observation function
        # q_sample: function to sample from process noise
        # p_observe: function to compute observation likelihood
        # initial_sample: function to sample initial particles

        n = len(z)
        particles = np.zeros((n, N_particles))
        weights = np.zeros((n, N_particles))
        x_est = np.zeros(n)

        # Initialize particles
        particles[0] = initial_sample(N_particles)
        weights[0] = np.ones(N_particles) / N_particles

        for k in range(1, n):
            # Resample if needed
            if 1.0 / np.sum(weights[k - 1] ** 2) < N_particles / 2:
                indices = np.random.choice(N_particles, N_particles, p=weights[k - 1])
                particles[k - 1] = particles[k - 1, indices]
                weights[k - 1] = np.ones(N_particles) / N_particles

            # Predict
            for i in range(N_particles):
                particles[k, i] = f_pred(particles[k - 1, i]) + q_sample()

            # Update weights
            for i in range(N_particles):
                weights[k, i] = weights[k - 1, i] * p_observe(z[k], h_obs(particles[k, i]))

            # Normalize weights
            weights[k] = weights[k] / np.sum(weights[k])

            # Estimate
            x_est[k] = np.sum(particles[k] * weights[k])

        return x_est, particles, weights


# 5. Nonlinear Filtering Techniques
class NonlinearFilter:
    def median_filter(self, x, window_size):
        y = np.zeros_like(x)
        half_window = window_size // 2

        for i in range(len(x)):
            window_start = max(0, i - half_window)
            window_end = min(len(x), i + half_window + 1)
            window = x[window_start:window_end]
            y[i] = np.median(window)

        return y

    def order_statistic_filter(self, x, window_size, order):
        y = np.zeros_like(x)
        half_window = window_size // 2

        for i in range(len(x)):
            window_start = max(0, i - half_window)
            window_end = min(len(x), i + half_window + 1)
            window = x[window_start:window_end]
            sorted_window = np.sort(window)
            y[i] = sorted_window[min(order, len(sorted_window) - 1)]

        return y

    def alpha_trimmed_mean_filter(self, x, window_size, alpha):
        y = np.zeros_like(x)
        half_window = window_size // 2

        for i in range(len(x)):
            window_start = max(0, i - half_window)
            window_end = min(len(x), i + half_window + 1)
            window = x[window_start:window_end]
            sorted_window = np.sort(window)

            trim = int(alpha * len(sorted_window) / 2)
            if trim < len(sorted_window) // 2:
                y[i] = np.mean(sorted_window[trim:-trim if trim > 0 else None])
            else:
                y[i] = sorted_window[len(sorted_window) // 2]  # Fallback to median

        return y

    def max_filter(self, x, window_size):
        return self.order_statistic_filter(x, window_size, -1)

    def min_filter(self, x, window_size):
        return self.order_statistic_filter(x, window_size, 0)

    def stack_filter(self, x, window_size, thresholds):
        y = np.zeros_like(x)

        for threshold in thresholds:
            binary_x = (x > threshold).astype(float)
            filtered_binary = self.median_filter(binary_x, window_size)
            y += filtered_binary * threshold

        return y

    def homomorphic_filter(self, x, filter_func, window_size):
        if np.any(x <= 0):
            offset = abs(np.min(x)) + 1e-10
            x = x + offset

        log_x = np.log(x)
        filtered_log = filter_func(log_x, window_size)
        y = np.exp(filtered_log)

        if offset:
            y = y - offset

        return y


# Demo and visualization functions
def demo_fir_filters():
    t, clean_signal, noisy_signal = generate_test_signal()
    fs = 1000  # Sample rate in Hz

    fir = FIRFilter()

    # Window method FIR design
    N = 51  # Filter order
    fc = 80  # Cutoff frequency
    h_window = fir.window_method(N, fc, fs)

    # Apply filter
    filtered_signal = np.convolve(noisy_signal, h_window, mode='same')

    # Parks-McClellan FIR design
    bands = [0, 70, 100, 500]  # Frequency bands
    desired = [1, 0]  # Desired response in each band
    h_pm = fir.parks_mcclellan(N, bands, desired, fs)

    # Apply filter
    filtered_signal_pm = np.convolve(noisy_signal, h_pm, mode='same')

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, noisy_signal)
    plt.title('Noisy Signal')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, filtered_signal)
    plt.title('Filtered Signal (Window Method)')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, filtered_signal_pm)
    plt.title('Filtered Signal (Parks-McClellan)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def demo_iir_filters():
    t, clean_signal, noisy_signal = generate_test_signal()
    fs = 1000  # Sample rate in Hz

    iir = IIRFilter()

    # Butterworth filter design
    N = 4  # Filter order
    fc = 80  # Cutoff frequency
    b_butter, a_butter = iir.butterworth_design(N, fc, fs)

    # Apply filter
    filtered_signal_butter = signal.lfilter(b_butter, a_butter, noisy_signal)

    # Chebyshev filter design
    b_cheby, a_cheby = iir.chebyshev_design(N, fc, fs, rp=1)

    # Apply filter
    filtered_signal_cheby = signal.lfilter(b_cheby, a_cheby, noisy_signal)

    # Elliptic filter design
    b_ellip, a_ellip = iir.elliptic_design(N, fc, fs, rp=1, rs=80)

    # Apply filter
    filtered_signal_ellip = signal.lfilter(b_ellip, a_ellip, noisy_signal)

    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(t, noisy_signal)
    plt.title('Noisy Signal')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(t, filtered_signal_butter)
    plt.title('Filtered Signal (Butterworth)')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(t, filtered_signal_cheby)
    plt.title('Filtered Signal (Chebyshev)')
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(t, filtered_signal_ellip)
    plt.title('Filtered Signal (Elliptic)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def demo_adaptive_filters():
    t, clean_signal, noisy_signal = generate_test_signal()

    adaptive = AdaptiveFilter()

    # System identification problem
    # Using clean signal as input and noisy signal as desired output
    N = 32  # Filter length
    mu = 0.01  # Step size

    # LMS algorithm
    y_lms, e_lms, w_lms = adaptive.lms_algorithm(clean_signal, noisy_signal, N, mu)

    # RLS algorithm
    y_rls, e_rls, w_rls = adaptive.rls_algorithm(clean_signal, noisy_signal, N)

    plt.figure(figsize=(12, 9))

    plt.subplot(4, 1, 1)
    plt.plot(t, clean_signal)
    plt.title('Clean Signal (Input)')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(t, noisy_signal)
    plt.title('Noisy Signal (Desired)')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(t, y_lms)
    plt.title('LMS Filter Output')
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(t, y_rls)
    plt.title('RLS Filter Output')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def demo_nonlinear_filters():
    t, clean_signal, noisy_signal = generate_test_signal()

    nonlinear = NonlinearFilter()

    # Apply nonlinear filters
    window_size = 5

    median_filtered = nonlinear.median_filter(noisy_signal, window_size)
    max_filtered = nonlinear.max_filter(noisy_signal, window_size)
    min_filtered = nonlinear.min_filter(noisy_signal, window_size)
    alpha_trimmed = nonlinear.alpha_trimmed_mean_filter(noisy_signal, window_size, 0.2)

    plt.figure(figsize=(12, 10))

    plt.subplot(5, 1, 1)
    plt.plot(t, noisy_signal)
    plt.title('Noisy Signal')
    plt.grid(True)

    plt.subplot(5, 1, 2)
    plt.plot(t, median_filtered)
    plt.title('Median Filtered Signal')
    plt.grid(True)

    plt.subplot(5, 1, 3)
    plt.plot(t, max_filtered)
    plt.title('Max Filtered Signal')
    plt.grid(True)

    plt.subplot(5, 1, 4)
    plt.plot(t, min_filtered)
    plt.title('Min Filtered Signal')
    plt.grid(True)

    plt.subplot(5, 1, 5)
    plt.plot(t, alpha_trimmed)
    plt.title('Alpha-Trimmed Mean Filtered Signal')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run demonstrations
    demo_fir_filters()
    demo_iir_filters()
    demo_adaptive_filters()
    demo_nonlinear_filters()