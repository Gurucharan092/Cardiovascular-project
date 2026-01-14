import pandas as pd
import numpy as np

# -----------------------------
# LOAD FEATURES
# -----------------------------
df = pd.read_csv("ecg_features.csv")

# -----------------------------
# STATE DEFINITIONS
# -----------------------------
STATE_MAP = {
    0: "Resting Stable",
    1: "Active / Elevated Demand",
    2: "Relaxed",
    3: "Sympathetic Dominant",
    4: "Mild Strain",
    5: "Elevated Strain",
    6: "Fatigued",
    7: "Recovering",
    8: "Signal Unstable",
    9: "Insufficient Data"
}

# -----------------------------
# STATE LABELING FUNCTION
# -----------------------------
def assign_state(row):
    hr = row["mean_hr"]
    hrv = row["rmssd"]
    rr_std = row["rr_std"]
    peaks = row["peak_count"]

    # Safety checks
    if hr <= 0 or hrv <= 0 or peaks < 6:
        return 9  # Insufficient Data

    if rr_std > 120 or peaks < 10:
        return 8  # Signal Unstable

    # Relaxed
    if 55 <= hr <= 75 and hrv > 40:
        return 2

    # Resting Stable
    if 60 <= hr <= 80 and 25 <= hrv <= 40:
        return 0

    # Mild Strain
    if 80 < hr <= 95 and 20 <= hrv < 30:
        return 4

    # Elevated Strain
    if hr > 95 and hrv < 20:
        return 5

    # Sympathetic Dominant
    if hr > 85 and hrv < 30:
        return 3

    # Active
    if hr > 90:
        return 1

    return 0  # Default: Resting Stable

# -----------------------------
# APPLY LABELING
# -----------------------------
df["state_id"] = df.apply(assign_state, axis=1)
df["state_label"] = df["state_id"].map(STATE_MAP)

# -----------------------------
# SAVE LABELED DATASET
# -----------------------------
df.to_csv("ecg_features_labeled.csv", index=False)

print("State labeling complete.")
print(df["state_label"].value_counts())