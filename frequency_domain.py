import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, freqz

Fs = 250  # ECG sampling rate (250 Hz)

# Create a single figure for all three filter frequency responses
plt.figure(figsize=(12, 10))

# =========================================================
# 1. HIGH-PASS FILTER (0.5 Hz Cutoff)
# =========================================================
b_hp, a_hp = butter(4, 0.5 / (Fs / 2), btype='high')
w_hp, h_hp = freqz(b_hp, a_hp, worN=2000)

plt.subplot(3, 1, 1)
plt.plot((w_hp / np.pi) * (Fs / 2), 20 * np.log10(np.abs(h_hp) + 1e-15), color='darkorange', linewidth=1.5)
plt.title("High-Pass Filter Frequency Response (0.5 Hz Cutoff)", fontsize=11, fontweight='bold')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.ylim(-60, 5)
plt.grid(True, linestyle=':', alpha=0.6)

# =========================================================
# 2. NOTCH FILTER (50 Hz Cutoff)
# =========================================================
b_notch, a_notch = iirnotch(50, 30, Fs)
w_notch, h_notch = freqz(b_notch, a_notch, worN=2000)

plt.subplot(3, 1, 2)
plt.plot((w_notch / np.pi) * (Fs / 2), 20 * np.log10(np.abs(h_notch) + 1e-15), color='forestgreen', linewidth=1.5)
plt.title("Notch Filter Frequency Response (50 Hz Cutoff)", fontsize=11, fontweight='bold')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.ylim(-60, 5)
plt.grid(True, linestyle=':', alpha=0.6)

# =========================================================
# 3. LOW-PASS FILTER (15 Hz Cutoff)
# =========================================================
b_lp, a_lp = butter(4, 15 / (Fs / 2), btype='low')
w_lp, h_lp = freqz(b_lp, a_lp, worN=2000)

plt.subplot(3, 1, 3)
plt.plot((w_lp / np.pi) * (Fs / 2), 20 * np.log10(np.abs(h_lp) + 1e-15), color='royalblue', linewidth=1.5)
plt.title("Low-Pass Filter Frequency Response (15 Hz Cutoff)", fontsize=11, fontweight='bold')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.ylim(-60, 5)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
