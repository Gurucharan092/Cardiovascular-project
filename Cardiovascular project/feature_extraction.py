import numpy as np
import pandas as pd
import neurokit2 as nk

# -----------------------------
# CONFIG
# -----------------------------
FS = 360               # Sampling frequency (MIT-BIH standard)
WINDOW_SIZE = 10       # Number of beats per window
STEP_SIZE = 5          # Sliding window step

# -----------------------------
# LOAD DATA
# -----------------------------
def load_ecg(csv_path):
    data = pd.read_csv(csv_path, header=None)
    ecg_signals = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    return ecg_signals, labels

# -----------------------------
# RECONSTRUCT ECG STREAM
# -----------------------------
def reconstruct_signal(ecg_beats):
    return ecg_beats.flatten()

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(ecg_signal):
    try:
        # Clean ECG
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=FS)

        # R-peak detection
        peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=FS)
        rpeaks = info["ECG_R_Peaks"]

        if len(rpeaks) < 5:
            return None

        # RR intervals (ms)
        rr_intervals = np.diff(rpeaks) / FS * 1000

        # Heart Rate
        mean_hr = 60000 / np.mean(rr_intervals)

        # HRV Metrics
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        sdnn = np.std(rr_intervals)

        return {
            "mean_hr": mean_hr,
            "rmssd": rmssd,
            "sdnn": sdnn,
            "rr_std": np.std(rr_intervals),
            "peak_count": len(rpeaks)
        }

    except Exception:
        return None

# -----------------------------
# WINDOW-BASED PROCESSING
# -----------------------------
def process_dataset(ecg_beats):
    features = []

    for i in range(0, len(ecg_beats) - WINDOW_SIZE, STEP_SIZE):
        window = ecg_beats[i:i + WINDOW_SIZE]
        ecg_stream = reconstruct_signal(window)

        feat = extract_features(ecg_stream)
        if feat:
            features.append(feat)

    return pd.DataFrame(features)

# -----------------------------
# RUN PIPELINE
# -----------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    ecg_beats, labels = load_ecg("mitbih_train.csv")

    print("Extracting features...")
    feature_df = process_dataset(ecg_beats)

    feature_df.to_csv("ecg_features.csv", index=False)
    print("Feature extraction complete.")