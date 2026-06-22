import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

# =========================================================
# LOAD ECG RECORD
# =========================================================
# Dynamically locate the data file relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
record_name = os.path.join(script_dir, 'files', '07879')

record = wfdb.rdrecord(record_name)
signal = record.p_signal[:, 0]
Fs = record.fs

# =========================================================
# ADD SAP ID BASED NOISE
# =========================================================
A_noise = 0.026  # Noise amplitude from SAP ID calculation
sap_noise = A_noise * np.ones_like(signal)
signal_noisy = signal + sap_noise

# =========================================================
# SIGNAL FILTERING (APPLIED TO FULL SIGNAL)
# =========================================================
# 1. Estimate baseline drift using a Low-Pass Filter (0.5 Hz) and subtract it
b_drift, a_drift = butter(2, 0.5 / (Fs / 2), btype='low')
drift_full = filtfilt(b_drift, a_drift, signal_noisy)
ecg_hp_full = signal_noisy - drift_full

# 2. Notch Filter (50 Hz) - Remove Powerline Interference
b_notch, a_notch = iirnotch(50, 30, Fs)
ecg_notch_full = filtfilt(b_notch, a_notch, ecg_hp_full)

# 3. Low-Pass Filter (15 Hz) - Remove High-Frequency Noise and Smooth the Wave
b_lp, a_lp = butter(4, 15 / (Fs / 2), btype='low')
ecg_filtered_full = filtfilt(b_lp, a_lp, ecg_notch_full)

# =========================================================
# EXTRACT 30-SECOND ECG SEGMENTS FOR PLOTTING
# =========================================================
start = int(60 * Fs)
end = int(90 * Fs)

segment = signal_noisy[start:end]
ecg_filtered = ecg_filtered_full[start:end]
t = np.arange(len(segment)) / Fs

# =========================================================
# TIME-DOMAIN COMPARISON PLOT
# =========================================================
plt.figure(figsize=(12, 6))

# Plot Noisy ECG with alpha transparency to keep the filtered line visible
plt.plot(t, segment, color='crimson', alpha=0.6, linewidth=1, label="Noisy ECG (With SAP Noise)")

# Plot Filtered ECG
plt.plot(t, ecg_filtered, color='blue', linewidth=1.5, label="Filtered ECG (Smooth & Clean)")

plt.title("Time-Domain Comparison: Before vs After Filtering", fontsize=12, fontweight='bold')
plt.xlabel("Time (seconds)", fontsize=10)
plt.ylabel("Amplitude (mV)", fontsize=10)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()