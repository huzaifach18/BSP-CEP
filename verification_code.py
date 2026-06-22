import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, freqz

Fs = 250

# ==========================================
# HIGH PASS FILTER
# ==========================================
b_hp, a_hp = butter(4, 0.5/(Fs/2), btype='high')

# ==========================================
# NOTCH FILTER
# ==========================================
b_notch, a_notch = iirnotch(50, 30, Fs)

# ==========================================
# LOW PASS FILTER
# ==========================================
b_lp, a_lp = butter(4, 100/(Fs/2), btype='low')

# ==========================================
# FUNCTION FOR RESPONSE ANALYSIS
# ==========================================
def analyze_filter(b, a, title):

    w, h = freqz(b, a, worN=4000)

    freq = (w/np.pi)*(Fs/2)

    magnitude = 20 * np.log10(np.abs(h))

    phase = np.angle(h)

    # -------------------------------
    # Magnitude Response
    # -------------------------------
    plt.figure(figsize=(10,4))

    plt.plot(freq, magnitude)

    plt.title(title + " - Magnitude Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid()

    plt.show()

    # -------------------------------
    # Phase Response
    # -------------------------------
    plt.figure(figsize=(10,4))

    plt.plot(freq, phase)

    plt.title(title + " - Phase Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.grid()

    plt.show()

    return freq, magnitude

# ==========================================
# ANALYZE FILTERS
# ==========================================
freq_hp, mag_hp = analyze_filter(b_hp, a_hp, "High-Pass Filter")

freq_notch, mag_notch = analyze_filter(b_notch, a_notch, "Notch Filter")

freq_lp, mag_lp = analyze_filter(b_lp, a_lp, "Low-Pass Filter")

# ==========================================
# QUANTITATIVE VERIFICATION
# ==========================================

# High-pass attenuation near 0.1 Hz
idx_hp = np.argmin(np.abs(freq_hp - 0.1))
print("High-pass attenuation at 0.1 Hz:",
      mag_hp[idx_hp], "dB")

# Notch attenuation at 50 Hz
idx_notch = np.argmin(np.abs(freq_notch - 50))
print("Notch attenuation at 50 Hz:",
      mag_notch[idx_notch], "dB")

# Low-pass attenuation at 120 Hz
idx_lp = np.argmin(np.abs(freq_lp - 120))
print("Low-pass attenuation at 120 Hz:",
      mag_lp[idx_lp], "dB")

# ==========================================
# PASSBAND RIPPLE CHECK
# ==========================================

passband = mag_lp[(freq_lp >= 0.5) & (freq_lp <= 100)]

ripple = np.max(passband) - np.min(passband)

print("Passband Ripple:", ripple, "dB")