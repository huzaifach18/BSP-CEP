import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

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
# EXTRACT 30-SECOND ECG SEGMENT
# =========================================================
start = int(60 * Fs)
end = int(90 * Fs)

segment = signal_noisy[start:end]
ecg_filtered = ecg_filtered_full[start:end]
t = np.arange(len(ecg_filtered)) / Fs

# =========================================================
# R-PEAK DETECTION & RR INTERVALS
# =========================================================
# Using prominence-based peak finding for robust QRS detection
peaks, _ = find_peaks(ecg_filtered, distance=int(Fs * 0.4), prominence=0.15)

# Calculate RR Intervals (in seconds) and Features
RR_intervals = np.diff(peaks) / Fs
SDNN = np.std(RR_intervals)
RMSSD = np.sqrt(np.mean(np.diff(RR_intervals)**2))

# =========================================================
# PRINT ANALYSIS REPORT TO TERMINAL
# =========================================================
print("=========================================")
print("ECG FEATURE EXTRACTION & ANALYSIS")
print("=========================================")
print(f"Total Heart Beats Detected : {len(peaks)}")
print(f"Average RR Interval        : {np.mean(RR_intervals):.4f} seconds")
print(f"SDNN                       : {SDNN:.4f} seconds ({SDNN * 1000:.1f} ms)")
print(f"RMSSD                      : {RMSSD:.4f} seconds ({RMSSD * 1000:.1f} ms)")
print("-----------------------------------------")

# Simple AF Detection Rule
if SDNN > 0.1:
    print("Result: Possible Atrial Fibrillation Detected (High Heart Rate Variability)")
else:
    print("Result: Normal Rhythm (Normal Heart Rate Variability)")
print("=========================================")

# =========================================================
# PLOT R-PEAKS & RR TACHOGRAM
# =========================================================
plt.figure(figsize=(12, 8))

# Subplot 1: Filtered ECG and detected R-peaks
plt.subplot(2, 1, 1)
plt.plot(t, ecg_filtered, color='blue', label='Filtered ECG')
plt.scatter(peaks / Fs, ecg_filtered[peaks], color='red', marker='x', s=60, label='Detected R-peak (QRS)', zorder=5)
plt.title("R-Peak Detection on Filtered ECG Signal", fontsize=12, fontweight='bold')
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

# Subplot 2: RR Interval Tachogram
plt.subplot(2, 1, 2)
plt.plot(peaks[1:] / Fs, RR_intervals * 1000, color='purple', marker='o', markersize=5, linestyle='-', linewidth=1.5, label='RR Interval')
plt.title("RR Interval Tachogram (Heartbeat Variability)", fontsize=12, fontweight='bold')
plt.xlabel("Time (seconds)")
plt.ylabel("RR Interval (ms)")
plt.grid(True, linestyle=':', alpha=0.6)

# Add text box overlay for SDNN and RMSSD inside the RR plot
stats_text = f"SDNN: {SDNN*1000:.1f} ms\nRMSSD: {RMSSD*1000:.1f} ms"
plt.text(0.03, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='purple'))
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()