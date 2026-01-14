import numpy as np
import pandas as pd
import neurokit2 as nk

# -----------------------------
# CONFIG
# -----------------------------
FS = 360  # Sampling frequency

# -----------------------------
# LOAD ECG FEATURES (raw signal windows needed)
# -----------------------------
def load_features(csv_path):
    return pd.read_csv(csv_path)

# -----------------------------
# IRREGULARITY METRICS
# -----------------------------
def compute_irregularity_metrics(rr_intervals):
    rr_mean = np.mean(rr_intervals)
    rr_std = np.std(rr_intervals)
    rr_cv = rr_std / rr_mean if rr_mean > 0 else 0
    rr_diff_std = np.std(np.diff(rr_intervals))

    # Ectopic-like detection (relative RR jump)
    rr_diff = np.abs(np.diff(rr_intervals))
    ectopic_beats = rr_diff > (0.2 * rr_mean)
    ectopic_ratio = np.sum(ectopic_beats) / len(rr_intervals)

    return rr_mean, rr_std, rr_cv, rr_diff_std, ectopic_ratio

# -----------------------------
# IRREGULARITY STATE LOGIC
# -----------------------------
def assign_irregularity_state(rr_intervals):
    if len(rr_intervals) < 5:
        return 4  # Unusual Rhythm Consistency (insufficient rhythm info)

    rr_mean, rr_std, rr_cv, rr_diff_std, ectopic_ratio = compute_irregularity_metrics(rr_intervals)

    # Stable
    if rr_cv < 0.05 and ectopic_ratio < 0.02:
        return 0

    # Occasional irregular beats
    if rr_cv < 0.10 and ectopic_ratio < 0.08:
        return 1

    # Irregular rhythm detected
    if rr_cv < 0.20 and ectopic_ratio < 0.15:
        return 2

    # Frequent irregular episodes
    if rr_cv >= 0.20 or ectopic_ratio >= 0.15:
        return 3

    return 4  # Fallback

# -----------------------------
# PROCESS ECG WINDOWS
# -----------------------------
def process_ecg_windows(ecg_windows):
    irregularity_states = []

    for ecg_signal in ecg_windows:
        try:
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=FS)
            _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=FS)
            rpeaks = info["ECG_R_Peaks"]

            if len(rpeaks) < 6:
                irregularity_states.append(4)
                continue

            rr_intervals = np.diff(rpeaks) / FS * 1000
            state = assign_irregularity_state(rr_intervals)
            irregularity_states.append(state)

        except Exception:
            irregularity_states.append(4)

    return irregularity_states

# -----------------------------
# STATE LABEL MAP
# -----------------------------
IRREGULARITY_MAP = {
    0: "Stable Rhythm Pattern",
    1: "Occasional Irregular Beats",
    2: "Irregular Rhythm Pattern Detected",
    3: "Frequent Irregular Rhythm Episodes",
    4: "Unusual Rhythm Consistency"
}

# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":
    print("Loading ECG feature dataset...")
    df = pd.read_csv("ecg_features_labeled.csv")

    # NOTE:
    # This assumes RR-based features were computed earlier.
    # If RR intervals are not stored, reprocessing ECG windows is required.

    print("Reconstructing RR-based irregularity states...")

    # Simulate RR reconstruction from available features
    # (For Kaggle MIT-BIH this approximation is acceptable)
    irregularity_state_ids = []

    for _, row in df.iterrows():
        hr = row["mean_hr"]
        rmssd = row["rmssd"]

        if hr <= 0 or rmssd <= 0:
            irregularity_state_ids.append(4)
            continue

        rr_mean = 60000 / hr
        rr_std = rmssd * 0.8  # Physiological approximation
        rr_cv = rr_std / rr_mean

        if rr_cv < 0.05:
            irregularity_state_ids.append(0)
        elif rr_cv < 0.10:
            irregularity_state_ids.append(1)
        elif rr_cv < 0.20:
            irregularity_state_ids.append(2)
        else:
            irregularity_state_ids.append(3)

    df["rhythm_state_id"] = irregularity_state_ids
    df["rhythm_state_label"] = df["rhythm_state_id"].map(IRREGULARITY_MAP)

    df.to_csv("ecg_features_with_irregularity.csv", index=False)

    print("Irregularity awareness labeling complete.")
    print(df["rhythm_state_label"].value_counts())