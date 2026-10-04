import os

import numpy as np


DATA_DIR = "processed_data/split"
INPUT_LENGTH = 180
CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}


def load_split(split_name):
    """Load and validate one dataset split."""
    x_path = os.path.join(DATA_DIR, f"X_{split_name}.npy")
    y_path = os.path.join(DATA_DIR, f"y_{split_name}.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing files for the {split_name} split.")

    X = np.load(x_path)
    y = np.load(y_path)

    # Each heartbeat must have 180 samples and one matching label.
    if len(X) != len(y):
        raise ValueError(f"{split_name}: feature and label counts do not match.")
    if X.ndim != 2 or X.shape[1] != INPUT_LENGTH:
        raise ValueError(f"{split_name}: expected X shape (N, {INPUT_LENGTH}), got {X.shape}.")

    valid_labels = set(CLASS_NAMES)
    if not set(np.unique(y)).issubset(valid_labels):
        raise ValueError(f"{split_name}: invalid class label found.")

    return X, y


def load_dataset():
    """Load the train, validation, and test arrays."""
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    print("Dataset loaded successfully")
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)


if __name__ == "__main__":
    main()
