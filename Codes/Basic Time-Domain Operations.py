import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io.wavfile as wavfile

# Create a time vector
fs = 1000  # Sampling frequency (Hz)
t = np.arange(0, 1, 1 / fs)  # 1 second duration

# Generate example signals
signal1 = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
signal2 = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
noise = 0.2 * np.random.randn(len(t))  # Random noise
signal_with_noise = signal1 + noise

# Create a complex test signal
test_signal = signal1 + 0.3 * np.sin(2 * np.pi * 20 * t)

# Create a test signal with more dynamic range for non-linear transformations
test_signal2 = np.sin(2 * np.pi * 3 * t) + 0.3 * np.sin(2 * np.pi * 30 * t)
test_signal2 *= 1.5  # Amplify to exceed limits in some cases

# =========================================================================
# Display Original Signals
# =========================================================================
plt.figure(figsize=(12, 6))
plt.plot(t, signal1, label='5 Hz Signal')
plt.plot(t, signal2, label='10 Hz Signal')
plt.plot(t, noise, label='Noise')
plt.title("Original Signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================================================
# I. BASIC TIME DOMAIN OPERATIONS
# =========================================================================

# -------------------------------------------------------------------------
# 1.1 Addition and Subtraction
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 10))

# Signal addition (signal mixing)
plt.subplot(2, 1, 1)
sum_signal = signal1 + signal2
plt.plot(t, signal1, label='5 Hz Signal')
plt.plot(t, signal2, label='10 Hz Signal')
plt.plot(t, sum_signal, label='Mixed Signal (5 Hz + 10 Hz)')
plt.title("Addition: Signal Mixing")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Noise cancellation example
plt.subplot(2, 1, 2)
clean_signal = signal_with_noise - noise  # In practice, noise estimate would be used
plt.plot(t, signal1, label='Original Clean Signal')
plt.plot(t, signal_with_noise, label='Noisy Signal')
plt.plot(t, clean_signal, label='After Noise Cancellation')
plt.title("Subtraction: Noise Cancellation")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 1.2 Multiplication
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 15))

# AM modulation
plt.subplot(3, 1, 1)
carrier_freq = 50  # Hz
carrier = np.cos(2 * np.pi * carrier_freq * t)
modulated_signal = signal1 * carrier
plt.plot(t[:200], signal1[:200], label='Message Signal (5 Hz)')  # Show only a segment for clarity
plt.plot(t[:200], carrier[:200], label='Carrier (50 Hz)', alpha=0.5)
plt.plot(t[:200], modulated_signal[:200], label='AM Modulated Signal')
plt.title("Multiplication: AM Modulation")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Windowing example
plt.subplot(3, 1, 2)
window = signal.windows.hamming(len(t))
windowed_signal = signal1 * window
plt.plot(t, signal1, label='Original Signal')
plt.plot(t, window, label='Hamming Window')
plt.plot(t, windowed_signal, label='Windowed Signal')
plt.title("Multiplication: Windowing")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Gating example
plt.subplot(3, 1, 3)
gate = np.zeros_like(t)
gate[200:800] = 1  # Gate open from 0.2s to 0.8s
gated_signal = signal1 * gate
plt.plot(t, signal1, 'b-', alpha=0.5, label='Original Signal')
plt.plot(t, gate, 'g-', label='Gate Signal')
plt.plot(t, gated_signal, 'r-', label='Gated Signal')
plt.title("Multiplication: Gating")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 1.3 Time Shift and Scaling
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 15))

# Time shift
plt.subplot(3, 1, 1)
shift_samples = int(0.1 * fs)  # 0.1 second shift
time_shifted_signal = np.roll(signal1, shift_samples)
plt.plot(t, signal1, label='Original Signal')
plt.plot(t, time_shifted_signal, label='Shifted Signal (0.1s)')
plt.title("Time Shift")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Time compression (faster playback)
plt.subplot(3, 1, 2)
compression_factor = 2
# Create new time axis for compressed signal
t_compressed = np.arange(0, 1, 1 / (fs / compression_factor))[:len(t) // compression_factor]
compressed_signal = signal1[::compression_factor]
# Interpolate back to original length for comparison
time_compressed_signal = np.interp(t, t_compressed, compressed_signal)
plt.plot(t, signal1, label='Original Signal')
plt.plot(t, time_compressed_signal, label='Time Compressed Signal (2x)')
plt.title("Time Compression (Faster Playback)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Time expansion (slower playback)
plt.subplot(3, 1, 3)
expansion_factor = 2
# Create expanded signal using interpolation
t_expanded = np.linspace(0, 1, len(t) * expansion_factor)[:len(t) * expansion_factor]
expanded_indices = np.linspace(0, len(t) - 1, len(t_expanded))
time_expanded_signal = np.interp(expanded_indices, np.arange(len(t)), signal1)
# For display, show a segment of the expanded signal
plt.plot(t[:500], signal1[:500], label='Original Signal (segment)')
plt.plot(t_expanded[:1000:2], time_expanded_signal[:1000:2], 'o', markersize=3, label='Expanded Signal Points')
plt.plot(t_expanded[:1000], time_expanded_signal[:1000], label='Time Expanded Signal (2x)')
plt.title("Time Expansion (Slower Playback)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 1.4 Differentiation and Integration
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 10))

# Differentiation (edge detection)
plt.subplot(2, 1, 1)
# Create a signal with sharper transitions for better edge demonstration
edge_demo_signal = np.zeros_like(t)
edge_demo_signal[250:500] = 1
edge_demo_signal[600:800] = 0.7
# Apply differentiation
edge_signal = np.diff(edge_demo_signal, prepend=edge_demo_signal[0])
plt.plot(t, edge_demo_signal, label='Original Signal')
plt.plot(t, edge_signal, label='Differentiated Signal (Edges)')
plt.title("Differentiation (Edge Detection)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Integration (cumulative measurement)
plt.subplot(2, 1, 2)
plt.plot(t, signal1, label='Original Signal')
integrated_signal = np.cumsum(signal1) / fs
plt.plot(t, integrated_signal, label='Integrated Signal')
plt.title("Integration (Cumulative Measurement)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================================
# II. POINT-BY-POINT NON-LINEAR TRANSFORMATIONS
# =========================================================================

# -------------------------------------------------------------------------
# 2.1 Quantization
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
# Create quantized signals with different levels
levels_4 = 4
levels_8 = 8
quantized_4 = np.round(test_signal2 * levels_4 / 2) / (levels_4 / 2)
quantized_8 = np.round(test_signal2 * levels_8 / 2) / (levels_8 / 2)

plt.plot(t, test_signal2, label='Original Signal')
plt.plot(t, quantized_8, label=f'{levels_8} levels')
plt.title("Quantization with 8 Levels")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, test_signal2, label='Original Signal')
plt.plot(t, quantized_4, label=f'{levels_4} levels')
plt.title("Quantization with 4 Levels")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Zoomed view of quantization
plt.figure(figsize=(12, 6))
# Show a small segment for better visualization of quantization steps
segment = slice(100, 200)
plt.step(t[segment], quantized_4[segment], label=f'{levels_4} levels', where='post')
plt.plot(t[segment], test_signal2[segment], 'k--', label='Original Signal')
plt.title("Detailed View of 4-Level Quantization")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 2.2 Threshold Processing
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 10))

# Soft thresholding
plt.subplot(2, 1, 1)
threshold = 0.5
soft_thresholded = np.where(np.abs(test_signal2) > threshold, test_signal2, 0)
plt.plot(t, test_signal2, label='Original Signal')
plt.plot(t, soft_thresholded, label='Soft Thresholded Signal')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.axhline(y=-threshold, color='r', linestyle='--')
plt.title("Soft Thresholding")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Binary thresholding (for voice activity detection)
plt.subplot(2, 1, 2)
binary_thresholded = np.where(np.abs(test_signal2) > threshold, 1, 0)
plt.plot(t, test_signal2, label='Original Signal')
plt.plot(t, binary_thresholded, label='Binary Thresholded Signal')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.axhline(y=-threshold, color='r', linestyle='--')
plt.title("Binary Thresholding (Voice Activity Detection)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 2.3 Compression and Expansion
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 15))

# µ-law compression
plt.subplot(3, 1, 1)
mu = 255
mu_compressed = np.sign(test_signal2) * np.log(1 + mu * np.abs(test_signal2)) / np.log(1 + mu)
plt.plot(t, test_signal2, label='Original Signal')
plt.plot(t, mu_compressed, label='µ-law Compressed (µ=255)')
plt.title("µ-law Compression")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# A-law compression
plt.subplot(3, 1, 2)
A = 87.6
A_compressed = np.zeros_like(test_signal2)
mask = np.abs(test_signal2) < 1 / A
A_compressed[mask] = A * np.abs(test_signal2[mask]) / (1 + np.log(A))
A_compressed[~mask] = np.sign(test_signal2[~mask]) * (1 + np.log(A * np.abs(test_signal2[~mask]))) / (1 + np.log(A))
A_compressed = np.sign(test_signal2) * A_compressed
plt.plot(t, test_signal2, label='Original Signal')
plt.plot(t, A_compressed, label='A-law Compressed (A=87.6)')
plt.title("A-law Compression")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Power-law expansion
plt.subplot(3, 1, 3)
power = 2
# Normalize test signal to [-1, 1] for better expansion demonstration
normalized_signal = test_signal2 / np.max(np.abs(test_signal2))
expanded_signal = np.sign(normalized_signal) * np.abs(normalized_signal) ** power
plt.plot(t, normalized_signal, label='Original Signal (Normalized)')
plt.plot(t, expanded_signal, label='Power-law Expanded (n=2)')
plt.title("Power-law Expansion")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Compression curves visualization
plt.figure(figsize=(10, 8))
# Create input signal range
x = np.linspace(-1, 1, 1000)
# Calculate different compression curves
mu_comp = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
A_comp = np.zeros_like(x)
mask = np.abs(x) < 1 / A
A_comp[mask] = A * np.abs(x[mask]) / (1 + np.log(A))
A_comp[~mask] = np.sign(x[~mask]) * (1 + np.log(A * np.abs(x[~mask]))) / (1 + np.log(A))
A_comp = np.sign(x) * A_comp
# Linear response (no compression)
linear = x
# Plot compression curves
plt.plot(x, linear, 'k--', label='Linear (No Compression)')
plt.plot(x, mu_comp, label='µ-law (µ=255)')
plt.plot(x, A_comp, label='A-law (A=87.6)')
plt.title("Compression Curves")
plt.xlabel("Input Amplitude")
plt.ylabel("Output Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 2.4 Clipping and Limiting
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 10))

# Hard clipping
plt.subplot(2, 1, 1)
clip_limit = 0.8
hard_clipped = np.clip(test_signal2, -clip_limit, clip_limit)
plt.plot(t, test_signal2, label='Original Signal')
plt.plot(t, hard_clipped, label='Hard Clipped Signal')
plt.axhline(y=clip_limit, color='r', linestyle='--', label='Clip Limit')
plt.axhline(y=-clip_limit, color='r', linestyle='--')
plt.title("Hard Clipping (Peak Limiting)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Soft clipping (using tanh function)
plt.subplot(2, 1, 2)
# Normalize signal for better soft clipping demonstration
norm_signal = test_signal2 / np.max(np.abs(test_signal2)) * 2  # Scale up to show saturation
soft_clipped = np.tanh(norm_signal)
plt.plot(t, norm_signal, label='Original Signal (Scaled)')
plt.plot(t, soft_clipped, label='Soft Clipped Signal')
plt.title("Soft Clipping (Saturation)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Clipping curves visualization
plt.figure(figsize=(10, 8))
x = np.linspace(-3, 3, 1000)
hard_clip = np.clip(x, -1, 1)
soft_clip = np.tanh(x)
plt.plot(x, x, 'k--', label='Linear (No Clipping)')
plt.plot(x, hard_clip, label='Hard Clipping')
plt.plot(x, soft_clip, label='Soft Clipping (tanh)')
plt.axhline(y=1, color='r', linestyle=':')
plt.axhline(y=-1, color='r', linestyle=':')
plt.title("Clipping Curves")
plt.xlabel("Input Amplitude")
plt.ylabel("Output Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 2.5 Rectification
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 10))

# Create a signal with nice negative components for rectification demo
rect_demo = np.sin(2 * np.pi * 3 * t)

# Half-wave rectification
plt.subplot(2, 1, 1)
half_wave_rectified = np.maximum(rect_demo, 0)
plt.plot(t, rect_demo, label='Original Signal')
plt.plot(t, half_wave_rectified, label='Half-wave Rectified')
plt.title("Half-wave Rectification")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# Full-wave rectification
plt.subplot(2, 1, 2)
full_wave_rectified = np.abs(rect_demo)
plt.plot(t, rect_demo, label='Original Signal')
plt.plot(t, full_wave_rectified, label='Full-wave Rectified')
plt.title("Full-wave Rectification")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------
# Comprehensive Processing Function
# -------------------------------------------------------------------------
def time_domain_process(signal, fs=1000):
    """
    Apply all time-domain processing techniques to an input signal.

    Parameters:
    signal - Input signal array
    fs - Sampling frequency in Hz

    Returns:
    Dictionary containing all processed signals
    """
    results = {}

    # Basic Signal Operations
    t = np.arange(0, len(signal) / fs, 1 / fs)[:len(signal)]

    # Addition with a 15 Hz sine wave
    sine_15hz = 0.3 * np.sin(2 * np.pi * 15 * t)
    results["addition"] = signal + sine_15hz

    # Multiplication - AM modulation with 50 Hz carrier
    carrier = np.cos(2 * np.pi * 50 * t)
    results["am_modulation"] = signal * carrier

    # Windowing
    window = signal.windows.hamming(len(signal))
    results["windowed"] = signal * window

    # Time shift by 0.1 seconds
    shift_samples = int(0.1 * fs)
    results["time_shifted"] = np.roll(signal, shift_samples)

    # Time compression (2x faster)
    compressed = signal[::2]
    results["compressed"] = np.interp(
        np.arange(0, len(signal)),
        np.arange(0, len(signal), 2)[:len(compressed)],
        compressed
    )

    # Differentiation and Integration
    results["differentiated"] = np.diff(signal, prepend=signal[0])
    integrated = np.cumsum(signal) / fs
    results["integrated"] = integrated / np.max(np.abs(integrated)) * np.max(np.abs(signal))

    # Non-Linear Transformations
    # Quantization
    results["quantized_8"] = np.round(signal * 4) / 4  # 8 levels

    # Thresholding
    threshold = 0.5 * np.max(np.abs(signal))
    results["soft_thresholded"] = np.where(np.abs(signal) > threshold, signal, 0)
    results["binary_thresholded"] = np.where(np.abs(signal) > threshold, 1, 0)

    # Compression
    mu = 255
    results["mu_law"] = np.sign(signal) * np.log(1 + mu * np.abs(signal)) / np.log(1 + mu)

    # Clipping
    clip_limit = 0.8 * np.max(np.abs(signal))
    results["hard_clipped"] = np.clip(signal, -clip_limit, clip_limit)
    results["soft_clipped"] = np.tanh(2 * signal / np.max(np.abs(signal))) * np.max(np.abs(signal))

    # Rectification
    results["half_wave"] = np.maximum(signal, 0)
    results["full_wave"] = np.abs(signal)

    return results


# Example signal for the function
sample_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t) + 0.2 * np.random.randn(len(t))

# Display example sample signal
plt.figure(figsize=(12, 6))
plt.plot(t, sample_signal)
plt.title("Sample Signal for Time Domain Processing")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()