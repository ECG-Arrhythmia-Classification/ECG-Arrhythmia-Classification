import os

import numpy as np


DATA_DIR = "processed_data/split"
SPLITS = [
    ("Train", "X_train.npy", "y_train.npy"),
    ("Validation", "X_val.npy", "y_val.npy"),
    ("Test", "X_test.npy", "y_test.npy"),
]


def verify_split(name, feature_file, label_file):
    # Build paths for the feature and label files in one split.
    feature_path = os.path.join(DATA_DIR, feature_file)
    label_path = os.path.join(DATA_DIR, label_file)

    if not os.path.exists(feature_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing files for {name} split.")

    # Load the arrays only after confirming that both files exist.
    X = np.load(feature_path)
    y = np.load(label_path)

    # Check that the data can be used safely for training and evaluation.
    if len(X) != len(y):
        raise ValueError(f"{name}: number of samples and labels does not match.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError(f"{name}: data contains NaN or infinite values.")
    if not np.all(np.isin(y, [0, 1, 2, 3, 4])):
        raise ValueError(f"{name}: invalid class label found.")

    print(f"\n{name} split")
    print("  X shape:", X.shape)
    print("  y shape:", y.shape)
    print("  Labels:", np.unique(y))
    print("  Data check: passed")


def main():
    print("Verifying processed dataset")
    # Verify train, validation, and test data using the same checks.
    for split in SPLITS:
        verify_split(*split)
    print("\nAll dataset checks passed.")


if __name__ == "__main__":
    main()
