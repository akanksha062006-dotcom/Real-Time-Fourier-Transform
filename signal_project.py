import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# REAL-TIME FOURIER TRANSFORM VISUALIZATION
# Python 3.15.1
# ==========================================

# Sampling parameters
FS = 1000
DURATION = 2
N = FS * DURATION
CHUNK_SIZE = 50

# Time axis
time = np.arange(N) / FS

# Initialize signal
signal = np.zeros(N)
sample_count = 0

# Signal phase
phase = 0.0

# Hann window
window = np.hanning(N)

# Frequency axis
frequency_axis = np.fft.rfftfreq(N, 1 / FS)

# Create graphs
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 8)
)

fig.suptitle(
    "Real-Time Fourier Transform Visualization",
    fontsize=16
)

# Time-domain graph
line1, = ax1.plot(
    time, signal, color="blue"
)

ax1.set_title("Time Domain Signal")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Amplitude")
ax1.set_ylim(-2, 2)
ax1.grid(True)

# Frequency-domain graph
line2, = ax2.plot(
    frequency_axis,
    np.zeros(len(frequency_axis)),
    color="red"
)

ax2.set_title("Frequency Domain - FFT")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
ax2.set_xlim(0, 150)
ax2.set_ylim(0, 1.5)
ax2.grid(True)


def update(frame):

    global signal, sample_count, phase

    # Generate new time samples
    index = sample_count + np.arange(CHUNK_SIZE)
    t = index / FS

    # Variable frequency
    f = 30 + 10 * np.sin(2 * np.pi * 0.1 * t)

    # Generate two signal components
    phase_steps = 2 * np.pi * f / FS
    phases = phase + np.cumsum(phase_steps)

    new_signal = (
        np.sin(phases)
        + 0.5 * np.sin(2 * phases)
        + 0.05 * np.random.randn(CHUNK_SIZE)
    )

    phase = phases[-1]
    sample_count += CHUNK_SIZE

    # Update signal buffer
    signal = np.roll(signal, -CHUNK_SIZE)
    signal[-CHUNK_SIZE:] = new_signal

    # Apply Hann window
    windowed_signal = signal * window

    # FFT calculation
    fft_values = np.fft.rfft(windowed_signal)

    # Calculate magnitude
    magnitude = (
        2 * np.abs(fft_values) / np.sum(window)
    )

    magnitude[0] /= 2

    if N % 2 == 0:
        magnitude[-1] /= 2

    # Find dominant frequency
    peak_index = np.argmax(magnitude[1:]) + 1
    dominant_frequency = frequency_axis[peak_index]

    # Update graphs
    line1.set_ydata(signal)
    line2.set_ydata(magnitude)

    ax2.set_title(
        f"FFT Spectrum | Dominant Frequency: "
        f"{dominant_frequency:.2f} Hz"
    )

    return line1, line2


# Start real-time animation
ani = FuncAnimation(
    fig,
    update,
    interval=50,
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()