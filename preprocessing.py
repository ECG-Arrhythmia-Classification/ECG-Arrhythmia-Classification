
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

# khai báo cấu hình chung

ROOT_DIR = Path(__file__).resolve().parent

INPUT_DIR = ROOT_DIR / "processed_data" / "split"
OUTPUT_DIR = ROOT_DIR / "processed_data" / "preprocessed"

SPLITS = ["train", "val", "test"]

SAMPLE_RATE = 360
HEARTBEAT_LENGTH = 180

LOW_CUT = 0.5
HIGH_CUT = 40.0
FILTER_ORDER = 4

MIN_STD = 1e-8


# lóad data

def load_data(split_name):
    """ tải tín hiệu ECG """

    X_path = INPUT_DIR / f"X_{split_name}.npy"
    y_path = INPUT_DIR / f"y_{split_name}.npy"

    if not X_path.exists():
        raise FileNotFoundError(
            f"Cannot find: {X_path}"
        )

    if not y_path.exists():
        raise FileNotFoundError(
            f"Cannot find: {y_path}"
        )

    X = np.load(X_path)
    y = np.load(y_path)

    return X, y

# check data
def check_data(X, y):
    
    if X.ndim != 2:
        raise ValueError(
            f"Expected X to be 2D, got {X.shape}"
        )

    if X.shape[1] != HEARTBEAT_LENGTH:
        raise ValueError(
            f"Expected heartbeat length {HEARTBEAT_LENGTH}, "
            f"got {X.shape[1]}"
        )

    if len(X) != len(y):
        raise ValueError(
            "Number of ECG signals and labels does not match."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "ECG data contains NaN or Inf."
        )


# filter signal
# Remove low-frequency and high-frequency noise
def filter_signal(X):
    """Remove low-frequency and high-frequency noise."""

    filter_sos = butter(
        FILTER_ORDER,
        [LOW_CUT, HIGH_CUT],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    X_filtered = sosfiltfilt(
        filter_sos,
        X,
        axis=1,
    )

    return X_filtered.astype(np.float32)

# normalize heartbeat

def normalize_signal(X):

    signal_mean = X.mean(
        axis=1,
        keepdims=True,
    )

    signal_std = X.std(
        axis=1,
        keepdims=True,
    )

    signal_std[
        signal_std < MIN_STD
    ] = 1.0

    X_normalized = (
        X - signal_mean
    ) / signal_std

    return X_normalized.astype(np.float32)

# preprocess data
# Apply filtering and normalization
def preprocess_data(X):

    X_float = X.astype(np.float32)

    X_filtered = filter_signal(
        X_float
    )

    X_processed = normalize_signal(
        X_filtered
    )

    return X_processed

# process data

def process_split(split_name):

    print(f"\nProcessing {split_name}...")

    X, y = load_data(
        split_name
    )

    check_data(
        X,
        y,
    )

    print(f"Input : {X.shape}")

    X_processed = preprocess_data(
        X
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_DIR / f"X_{split_name}.npy",
        X_processed,
    )

    np.save(
        OUTPUT_DIR / f"y_{split_name}.npy",
        y,
    )

    print(
        f"Output: {X_processed.shape}"
    )

# main
# control the train, val, test process
def main():

    print("=" * 50)
    print("ECG PREPROCESSING")
    print("=" * 50)

    for split_name in SPLITS:
        process_split(split_name)

    print("\nPreprocessing completed.")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
