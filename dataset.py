import numpy as np


INPUT_LENGTH = 180
NUM_CLASSES = 5
CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}


class ECGDataset:
    """A small wrapper that stores ECG heartbeat segments and their labels."""

    def __init__(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self._validate()

    def _validate(self):
        # Validate the format before the data is used for model training.
        if len(self.X) != len(self.y):
            raise ValueError("Feature and label counts do not match.")
        if self.X.ndim != 2 or self.X.shape[1] != INPUT_LENGTH:
            raise ValueError(f"Expected X shape (N, {INPUT_LENGTH}), got {self.X.shape}.")
        if not np.isfinite(self.X).all():
            raise ValueError("X contains NaN or infinite values.")

        valid_labels = set(CLASS_NAMES)
        if not set(np.unique(self.y)).issubset(valid_labels):
            raise ValueError("Invalid class label found.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def create_datasets(X_train, y_train, X_val, y_val, X_test, y_test):
    """Create one ECGDataset object for each data split."""
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)
    return train_dataset, val_dataset, test_dataset


def main():
    from data_loader import load_dataset

    datasets = create_datasets(*load_dataset())
    print("ECG datasets created successfully")
    print("Train samples:", len(datasets[0]))
    print("Validation samples:", len(datasets[1]))
    print("Test samples:", len(datasets[2]))


if __name__ == "__main__":
    main()
