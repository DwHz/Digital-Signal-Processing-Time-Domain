import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# Linear Convolution Implementation
def direct_convolution(x, h):
    N, M = len(x), len(h)
    y = np.zeros(N + M - 1)
    for n in range(len(y)):
        for k in range(M):
            if n - k >= 0 and n - k < N:
                y[n] += h[k] * x[n - k]
    return y


def overlap_add(x, h, L):
    N, M = len(x), len(h)
    P = L + M - 1
    K = int(np.ceil(N / L))
    y = np.zeros(N + M - 1)

    for k in range(K):
        start = k * L
        end = min(start + L, N)
        xk = x[start:end]
        if len(xk) < L:
            xk = np.pad(xk, (0, L - len(xk)))
        yk = np.convolve(xk, h)
        y[start:start + P] += yk

    return y


def overlap_save(x, h, L):
    N, M = len(x)
    R = L - M + 1
    K = int(np.ceil((N - M + 1) / R))
    y = np.zeros(N)

    x_padded = np.pad(x, (M - 1, 0))

    for k in range(K):
        start = k * R
        xk = x_padded[start:start + L]
        yk = np.convolve(xk, h, 'full')
        y[start:start + R] = yk[M - 1:M - 1 + R]

    return y[:N]


# Correlation Analysis Implementation
def autocorrelation(x, max_lag=None):
    if max_lag is None:
        max_lag = len(x) - 1

    N = len(x)
    R_xx = np.zeros(max_lag + 1)

    for m in range(max_lag + 1):
        for n in range(N - m):
            R_xx[m] += x[n] * x[n + m]
        R_xx[m] /= (N - m)

    R_xx_full = np.concatenate((R_xx[:0:-1], R_xx))
    return R_xx_full


def cross_correlation(x, y, max_lag=None):
    if max_lag is None:
        max_lag = len(x) - 1

    N = len(x)
    R_xy = np.zeros(2 * max_lag + 1)

    for m in range(-max_lag, max_lag + 1):
        sum_val = 0
        count = 0
        for n in range(N):
            if 0 <= n + m < N:
                sum_val += x[n] * y[n + m]
                count += 1
        R_xy[m + max_lag] = sum_val / count if count > 0 else 0

    return R_xy


def partial_correlation(x, y, z):
    R_xy = cross_correlation(x, y)
    R_xz = cross_correlation(x, z)
    R_yz = cross_correlation(y, z)

    mid_idx = len(R_xy) // 2
    R_xy_0 = R_xy[mid_idx]
    R_xz_0 = R_xz[mid_idx]
    R_yz_0 = R_yz[mid_idx]

    partial_corr = (R_xy_0 - R_xz_0 * R_yz_0) / np.sqrt((1 - R_xz_0 ** 2) * (1 - R_yz_0 ** 2))
    return partial_corr


# Example 1: Linear System Response
def demo_system_response():
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 5 * t)
    h = np.exp(-10 * t) * np.cos(2 * np.pi * 20 * t)
    y = direct_convolution(x, h[:100])

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, x)
    plt.title("Input Signal x(t)")

    plt.subplot(3, 1, 2)
    plt.plot(t[:100], h[:100])
    plt.title("Impulse Response h(t)")

    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(y)), y)
    plt.title("Output Signal y(t)")
    plt.tight_layout()
    plt.savefig("linear_system_response.png")


# Example 2: Periodic Signal Detection
def demo_periodic_detection():
    t = np.linspace(0, 2, 1000)
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t) + 0.2 * np.random.randn(len(t))

    R_xx = autocorrelation(x)
    lag_axis = np.arange(-len(x) + 1, len(x))

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title("Periodic Signal with Noise")

    plt.subplot(2, 1, 2)
    plt.plot(lag_axis, R_xx)
    plt.title("Autocorrelation")
    plt.tight_layout()
    plt.savefig("periodic_detection.png")


# Example 3: Time Delay Estimation
def demo_time_delay():
    t = np.linspace(0, 1, 1000)
    delay_samples = 100
    x = np.sin(2 * np.pi * 10 * t) * (t > 0.2) * (t < 0.8)
    y = np.roll(x, delay_samples) + 0.1 * np.random.randn(len(t))

    R_xy = cross_correlation(x, y)
    lag_axis = np.arange(-len(x) + 1, len(x))

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, x)
    plt.title("Original Signal")

    plt.subplot(3, 1, 2)
    plt.plot(t, y)
    plt.title("Delayed Signal")

    plt.subplot(3, 1, 3)
    plt.plot(lag_axis, R_xy)
    plt.axvline(x=delay_samples, color='r', linestyle='--')
    plt.title("Cross-correlation")
    plt.tight_layout()
    plt.savefig("time_delay_estimation.png")


# Example 4: Digital Filtering
def demo_digital_filtering():
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(len(t))

    fc = 0.1
    M = 31
    n = np.arange(M)
    h = np.sinc(2 * fc * (n - (M - 1) / 2)) * np.hamming(M)
    h = h / np.sum(h)

    y = np.convolve(x, h, 'same')

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title("Noisy Signal")

    plt.subplot(2, 1, 2)
    plt.plot(t, y)
    plt.title("Filtered Signal")
    plt.tight_layout()
    plt.savefig("digital_filtering.png")


# Example 5: Signal Similarity Analysis
def demo_signal_similarity():
    t = np.linspace(0, 1, 1000)
    x1 = np.sin(2 * np.pi * 5 * t)
    x2 = np.sin(2 * np.pi * 5 * t + np.pi / 4)
    x3 = np.sin(2 * np.pi * 10 * t)

    R_12 = cross_correlation(x1, x2)
    R_13 = cross_correlation(x1, x3)

    lag_axis = np.arange(-len(t) + 1, len(t))

    plt.figure(figsize=(10, 8))
    plt.subplot(5, 1, 1)
    plt.plot(t, x1)
    plt.title("Signal 1")

    plt.subplot(5, 1, 2)
    plt.plot(t, x2)
    plt.title("Signal 2 (Same frequency, phase shift)")

    plt.subplot(5, 1, 3)
    plt.plot(t, x3)
    plt.title("Signal 3 (Different frequency)")

    plt.subplot(5, 1, 4)
    plt.plot(lag_axis, R_12)
    plt.title("Cross-correlation: Signal 1 and 2")

    plt.subplot(5, 1, 5)
    plt.plot(lag_axis, R_13)
    plt.title("Cross-correlation: Signal 1 and 3")
    plt.tight_layout()
    plt.savefig("signal_similarity.png")


if __name__ == "__main__":
    demo_system_response()
    demo_periodic_detection()
    demo_time_delay()
    demo_digital_filtering()
    demo_signal_similarity()

    fs = 1000
    t = np.arange(0, 1, 1 / fs)
    x = np.sin(2 * np.pi * 10 * t)
    y = np.sin(2 * np.pi * 10 * t + np.pi / 4)
    z = np.sin(2 * np.pi * 20 * t)

    partial_corr_value = partial_correlation(x, y, z)
    print(f"Partial correlation between x and y controlling for z: {partial_corr_value}")
    plt.show()