import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# LOAD ECG RECORD
# =========================================================

# Dynamically locate the data file relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
record_name = os.path.join(script_dir, '07879')

record = wfdb.rdrecord(record_name)

signal = record.p_signal[:, 0]

Fs = record.fs

# =========================================================
# EXTRACT 30-SECOND SEGMENT & ESTIMATE BASELINE WANDER
# =========================================================

start_time = 60
end_time = 90

start_sample = int(start_time * Fs)
end_sample = int(end_time * Fs)

segment = signal[start_sample:end_sample]
t = np.arange(len(segment)) / Fs

# Estimate baseline wander on the whole signal using low-pass Butterworth filter (cutoff = 0.5 Hz)
from scipy.signal import butter, filtfilt, find_peaks
nyq = 0.5 * Fs
b, a = butter(2, 0.5 / nyq, btype='low')
full_baseline_wander = filtfilt(b, a, signal)
baseline_wander = full_baseline_wander[start_sample:end_sample]

# =========================================================
# PLOT ANNOTATED ECG
# =========================================================

plt.figure(figsize=(14, 5))

# Plot ECG signal
plt.plot(t, segment, color='blue', linewidth=1, label='ECG Signal')

# Plot estimated baseline wander
plt.plot(t, baseline_wander, color='orange', linestyle='--', linewidth=1.5, label='Estimated Baseline Wander')

# Detect regions where baseline wander deviates significantly from its segment mean
drift_mean = np.mean(baseline_wander)
threshold = 0.02  # mV threshold for significant deviation on 07879
is_wander = np.abs(baseline_wander - drift_mean) > threshold

# Find contiguous regions of high wander
diff = np.diff(is_wander.astype(int))
start_indices = np.where(diff == 1)[0] + 1
end_indices = np.where(diff == -1)[0] + 1

if len(is_wander) > 0:
    if is_wander[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if is_wander[-1]:
        end_indices = np.append(end_indices, len(is_wander) - 1)

# Highlight all high wander regions
legend_added = False
for start, end in zip(start_indices, end_indices):
    label = 'Baseline Wander Region' if not legend_added else ""
    plt.axvspan(t[start], t[end], color='red', alpha=0.12, label=label)
    legend_added = True

# Detect and plot all QRS complexes (R-peaks)
peaks, _ = find_peaks(segment, distance=int(Fs * 0.4), prominence=0.15)
plt.scatter(peaks / Fs, segment[peaks], color='red', marker='o', s=45, label='Detected QRS (R-Peak)', zorder=5)

# Annotate one QRS complex with an arrow
if len(peaks) > 0:
    qrs_idx = peaks[0]  # Point to the first detected QRS peak
    plt.annotate('QRS Complex (R-Peak)',
                 xy=(qrs_idx / Fs, segment[qrs_idx]),
                 xytext=((qrs_idx / Fs) + 2, segment[qrs_idx] + 0.3),
                 arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")

plt.title("Annotated ECG Signal with Baseline Wander Areas (07879)")

plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()

plt.show()