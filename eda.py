from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# CONFIGURATION

ROOT_DIR = Path(__file__).resolve().parent

SPLIT_DIR = ROOT_DIR / "processed_data" / "split"
PREPROCESSED_DIR = ROOT_DIR / "processed_data" / "preprocessed"
AUGMENTED_DIR = ROOT_DIR / "processed_data" / "augmented"

OUTPUT_DIR = ROOT_DIR / "results" / "eda"

CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}

# DATA LOADING

def load_data():
    """Load data needed for exploratory data analysis."""

    raw_signals = np.load(
        SPLIT_DIR / "X_train.npy"
    )

    labels = np.load(
        SPLIT_DIR / "y_train.npy"
    )

    processed_signals = np.load(
        PREPROCESSED_DIR / "X_train.npy"
    )

    augmented_labels = np.load(
        AUGMENTED_DIR / "y_train_aug.npy"
    )

    return (
        raw_signals,
        processed_signals,
        labels,
        augmented_labels,
    )

# CLASS DISTRIBUTION

def plot_class_distribution(
    original_labels,
    augmented_labels,
):
    """Compare class distribution before and after augmentation."""

    class_ids = list(CLASS_NAMES.keys())
    class_names = list(CLASS_NAMES.values())

    original_counts = [
        np.sum(original_labels == class_id)
        for class_id in class_ids
    ]

    augmented_counts = [
        np.sum(augmented_labels == class_id)
        for class_id in class_ids
    ]

    positions = np.arange(
        len(class_names)
    )

    bar_width = 0.35

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        positions - bar_width / 2,
        original_counts,
        width=bar_width,
        label="Before augmentation",
    )

    plt.bar(
        positions + bar_width / 2,
        augmented_counts,
        width=bar_width,
        label="After augmentation",
    )

    plt.xticks(
        positions,
        class_names,
    )

    plt.xlabel("ECG Class")
    plt.ylabel("Number of Samples")

    plt.title(
        "Class Distribution Before and After Augmentation"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "class_distribution.png",
        dpi=150,
    )

    plt.close()

# ECG EXAMPLES

def plot_class_examples(
    processed_signals,
    labels,
):
    """Plot one preprocessed heartbeat from each ECG class."""

    plt.figure(
        figsize=(10, 12)
    )

    for plot_index, (
        class_id,
        class_name,
    ) in enumerate(
        CLASS_NAMES.items(),
        start=1,
    ):

        class_indices = np.where(
            labels == class_id
        )[0]

        sample_index = class_indices[0]

        heartbeat = processed_signals[
            sample_index
        ]

        plt.subplot(
            len(CLASS_NAMES),
            1,
            plot_index,
        )

        plt.plot(
            heartbeat
        )

        plt.title(
            f"Class {class_name}"
        )

        plt.xlabel("Sample")
        plt.ylabel("Normalized Amplitude")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "class_examples.png",
        dpi=150,
    )

    plt.close()

# BEFORE / AFTER PREPROCESSING

def plot_preprocessing_comparison(
    raw_signals,
    processed_signals,
    labels,
):
    """Compare one heartbeat before and after preprocessing."""

    normal_indices = np.where(
        labels == 0
    )[0]

    sample_index = normal_indices[0]

    raw_signal = raw_signals[
        sample_index
    ]

    processed_signal = processed_signals[
        sample_index
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.subplot(
        2,
        1,
        1,
    )

    plt.plot(
        raw_signal
    )

    plt.title(
        "Before Preprocessing"
    )

    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(
        2,
        1,
        2,
    )

    plt.plot(
        processed_signal
    )

    plt.title(
        "After Preprocessing"
    )

    plt.xlabel("Sample")
    plt.ylabel("Normalized Amplitude")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "preprocessing_comparison.png",
        dpi=150,
    )

    plt.close()


# MAIN

def main():

    print("=" * 50)
    print("ECG EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        raw_signals,
        processed_signals,
        labels,
        augmented_labels,
    ) = load_data()

    print("\nCreating class distribution...")
    plot_class_distribution(
        labels,
        augmented_labels,
    )

    print("Creating ECG class examples...")
    plot_class_examples(
        processed_signals,
        labels,
    )

    print("Creating preprocessing comparison...")
    plot_preprocessing_comparison(
        raw_signals,
        processed_signals,
        labels,
    )

    print("\nEDA completed successfully.")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()