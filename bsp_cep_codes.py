import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
# Load ECG record
record_name = 'files/07879'
record = wfdb.rdrecord(record_name)
# Extract ECG signal and sampling frequency
signal = record.p_signal[:, 0]
Fs = record.fs
# =========================
# ADD SAP ID BASED NOISE
# =========================
A_noise = 0.026  # Noise amplitude from SAP ID calculation
# Create synthetic SAP noise
sap_noise = A_noise * np.ones_like(signal)
# Add SAP noise to ECG signal
signal = signal + sap_noise
# =======================
# SIGNAL FILTERING (APPLIED TO FULL SIGNAL)
# =========================
# 1. Estimate baseline drift using a Low-Pass Filter (0.5 Hz) and subtract it (remove 0.5 Hz noise)
b_drift, a_drift = butter(2, 0.5 / (Fs / 2), btype='low')
drift_full = filtfilt(b_drift, a_drift, signal)
ecg_hp_full = signal - drift_full
# 2. Notch Filter (50 Hz) - Remove Powerline Interference (50 Hz hum)
b_notch, a_notch = iirnotch(50, 30, Fs)
ecg_notch_full = filtfilt(b_notch, a_notch, ecg_hp_full)
# 3. Low-Pass Filter (15 Hz) - Remove High-Frequency Noise and Smooth the Wave
b_lp, a_lp = butter(4, 15 / (Fs / 2), btype='low')
ecg_filtered_full = filtfilt(b_lp, a_lp, ecg_notch_full)
# =========================
# EXTRACT 30-SECOND ECG SEGMENTS FOR PLOTTING
# =========================
start = int(60 * Fs)
end = int(90 * Fs)
segment = signal[start:end]
ecg_hp = ecg_hp_full[start:end]
ecg_notch = ecg_notch_full[start:end]
ecg_filtered = ecg_filtered_full[start:end]
# =========================================================
# PAN-TOMPKINS ALGORITHM FOR QRS DETECTION
# =========================================================
# 1. Derivative Filter (Slope detection)
ecg_deriv = np.diff(ecg_filtered)
ecg_deriv = np.append(ecg_deriv, 0)  # Maintain matching length
# 2. Squaring Function (Amplify QRS spikes, make positive)
ecg_squared = ecg_deriv ** 2
# 3. Moving Window Integration (MWI - smooths window of 150 ms)
window_len = int(0.15 * Fs)
ecg_mwi = np.convolve(ecg_squared, np.ones(window_len) / window_len, mode='same')
# 4. Find peaks on the Integrated Signal
from scipy.signal import find_peaks
mwi_peaks, _ = find_peaks(ecg_mwi, distance=int(Fs * 0.4), prominence=np.max(ecg_mwi) * 0.1)
# 5. Search local window to find exact R-peak location on the filtered ECG
r_peaks = []
search_window = int(0.1 * Fs)
for p in mwi_peaks:
    start_idx = max(0, p - search_window)
    end_idx = min(len(ecg_filtered), p + search_window)
    r_peak = start_idx + np.argmax(ecg_filtered[start_idx:end_idx])
    r_peaks.append(r_peak)
r_peaks = np.array(r_peaks)
# =========================
# PLOT RESULTS
# =========================
plt.figure(figsize=(14, 10))
t = np.arange(len(segment)) / Fs
# 1. Noisy ECG Signal
plt.subplot(4, 1, 1)
plt.plot(t, segment, color='crimson', linewidth=1)
plt.title("1. Noisy ECG Signal (With SAP Noise)", fontsize=11, fontweight='bold')
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.grid(True, linestyle=':', alpha=0.6)
# 2. After High-Pass Filter (Baseline Wander Removal via Subtraction)
plt.subplot(4, 1, 2)
plt.plot(t, ecg_hp, color='darkorange', linewidth=1)
plt.title("2. High-Pass Filtered ECG (Baseline Wander Removed)", fontsize=11, fontweight='bold')
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.grid(True, linestyle=':', alpha=0.6)
# 3. After Notch Filter (Powerline Interference Removal)
plt.subplot(4, 1, 3)
plt.plot(t, ecg_notch, color='forestgreen', linewidth=1)
plt.title("3. Notch Filtered ECG (Powerline Interference Removed)", fontsize=11, fontweight='bold')
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.grid(True, linestyle=':', alpha=0.6)
# 4. Final Filtered ECG Signal (Smooth Wave + Pan-Tompkins QRS Detection)
plt.subplot(4, 1, 4)
plt.plot(t, ecg_filtered, color='royalblue', linewidth=1, label='Filtered ECG')
plt.scatter(r_peaks / Fs, ecg_filtered[r_peaks], color='red', marker='o', s=45, label='Detected QRS (R-Peak)', zorder=5)
plt.title("4. Fully Filtered ECG Signal (QRS Peaks via Pan-Tompkins)", fontsize=11, fontweight='bold')
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()