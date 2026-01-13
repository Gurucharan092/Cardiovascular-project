import time
import math
import random
import numpy as np
from collections import deque
from firebase_admin import credentials, initialize_app, db

# -------------------------------
# Firebase Setup
# -------------------------------
cred = credentials.Certificate("serviceAccountKey.json")
initialize_app(cred, {
    "databaseURL": "https://cardio-monitoring-demo-default-rtdb.asia-southeast1.firebasedatabase.app"
})
ref = db.reference("/devices/device_001/live")

# -------------------------------
# Parameters
# -------------------------------
UPDATE_INTERVAL = 0.2        # seconds per update
SAMPLES_PER_UPDATE = 100
TARGET_HR_BPM = 72           # mean heart rate
HR_VARIABILITY = 5           # +/- BPM variation
RR_BUFFER_SEC = 30           # rolling buffer for HRV
R_PEAK_AMPLITUDE = 1.2       # synthetic R-peak height

# -------------------------------
# Rolling R-peak timestamps
# -------------------------------
r_peak_times = deque()  # stores absolute timestamps of R-peaks

# -------------------------------
# Helper Functions
# -------------------------------
def generate_segment(t_start, sample_rate=250):
    """
    Generate realistic synthetic ECG segment with multiple R-peaks.
    sample_rate: Hz
    """
    ecg = []
    segment_r_peaks = []

    # Simulate RR interval with small variability
    bpm = random.randint(TARGET_HR_BPM - HR_VARIABILITY, TARGET_HR_BPM + HR_VARIABILITY)
    rr_interval = 60 / bpm  # seconds per beat

    # For current segment, determine R-peak times
    t = t_start
    segment_end = t_start + UPDATE_INTERVAL

    while t < segment_end + rr_interval:
        if len(r_peak_times) == 0 or t - r_peak_times[-1] >= rr_interval:
            r_peak_times.append(t)
            segment_r_peaks.append(t)
            rr_interval = 60 / random.randint(TARGET_HR_BPM - HR_VARIABILITY, TARGET_HR_BPM + HR_VARIABILITY)
        t += rr_interval / 2  # small step to allow multiple peaks in segment if needed

    # Generate ECG waveform per sample
    time_step = 1 / sample_rate
    for i in range(SAMPLES_PER_UPDATE):
        sample_time = t_start + i * time_step
        # Base sinus waveform
        base = 0.2 * math.sin(2 * math.pi * 1 * sample_time)
        # Add R-peak if within 5 ms of any R-peak
        is_r_peak = any(abs(sample_time - rp) < 0.005 for rp in segment_r_peaks)
        if is_r_peak:
            value = base + R_PEAK_AMPLITUDE
        else:
            value = base + random.uniform(-0.05, 0.05)
        ecg.append(round(value, 3))

    return ecg, segment_r_peaks

def compute_hr(r_times, current_time, window_sec=RR_BUFFER_SEC):
    """Compute HR (BPM) using R-peaks in rolling window"""
    while r_times and current_time - r_times[0] > window_sec:
        r_times.popleft()
    if len(r_times) < 2:
        return 0
    rr_intervals = np.diff(list(r_times))
    avg_rr = np.mean(rr_intervals)
    bpm = 60 / avg_rr if avg_rr > 0 else 0
    return round(bpm)

def compute_rmssd(r_times, current_time, window_sec=RR_BUFFER_SEC):
    """Compute HRV (RMSSD) using R-peaks in rolling window"""
    valid_peaks = [t for t in r_times if current_time - t <= window_sec]
    if len(valid_peaks) < 3:
        return 0
    rr_intervals = np.diff(valid_peaks)
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    return round(rmssd, 2)

# -------------------------------
# Main Loop
# -------------------------------
t_global = time.time()

while True:
    # Generate ECG segment
    ecg, _ = generate_segment(t_global, sample_rate=250)  # 250 Hz synthetic

    t_global += UPDATE_INTERVAL

    # Compute HR and HRV using rolling R-peak buffer
    bpm = compute_hr(r_peak_times, t_global)
    hrv = compute_rmssd(r_peak_times, t_global)

    # Prepare payload for Firebase
    data = {
        "heart_rate": bpm,
        "hrv": hrv,
        "ecg": ecg,
        "timestamp": int(time.time())
    }

    # Push data to Firebase
    ref.set(data)

    time.sleep(UPDATE_INTERVAL)