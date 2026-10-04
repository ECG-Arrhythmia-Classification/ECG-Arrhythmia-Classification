from pathlib import Path

import numpy as np

# CONFIGURATION

ROOT_DIR = Path(__file__).resolve().parent

PREPROCESSED_DIR = ROOT_DIR / "processed_data" / "preprocessed"
AUGMENTED_DIR = ROOT_DIR / "processed_data" / "augmented"

HEARTBEAT_LENGTH = 180
RANDOM_SEED = 42

CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}

VALID_LABELS = set(CLASS_NAMES.keys())

# Minority classes are increased up to this number of samples.
TARGET_SAMPLES_PER_CLASS = 8000

# Prevent very small classes from being augmented too aggressively.
MAX_AUGMENT_MULTIPLIER = 10

# Mild augmentation settings.
NOISE_LEVEL_RANGE = (0.01, 0.03)
AMPLITUDE_SCALE_RANGE = (0.90, 1.10)
MAX_SHIFT_SAMPLES = 4

# DATA LOADING

def load_training_data():
    """Load the preprocessed training dataset."""

    feature_path = PREPROCESSED_DIR / "X_train.npy"
    label_path = PREPROCESSED_DIR / "y_train.npy"

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Cannot find training features: {feature_path}"
        )

    if not label_path.exists():
        raise FileNotFoundError(
            f"Cannot find training labels: {label_path}"
        )

    ecg_signals = np.load(feature_path).astype(
        np.float32,
        copy=False,
    )

    labels = np.load(label_path).astype(
        np.int64,
        copy=False,
    )

    return ecg_signals, labels


# DATA VALIDATION

def validate_training_data(ecg_signals, labels):
    """Check ECG shape, labels, and numeric values."""

    if ecg_signals.ndim != 2:
        raise ValueError(
            f"Expected ECG data to be 2D, "
            f"got shape {ecg_signals.shape}."
        )

    if ecg_signals.shape[1] != HEARTBEAT_LENGTH:
        raise ValueError(
            f"Expected heartbeat length {HEARTBEAT_LENGTH}, "
            f"got {ecg_signals.shape[1]}."
        )

    if len(ecg_signals) != len(labels):
        raise ValueError(
            "Number of ECG signals and labels does not match."
        )

    if not np.isfinite(ecg_signals).all():
        raise ValueError(
            "ECG data contains NaN or Inf values."
        )

    current_labels = set(np.unique(labels))

    if not current_labels.issubset(VALID_LABELS):
        raise ValueError(
            f"Invalid labels found: {sorted(current_labels)}."
        )

# AUGMENTATION METHODS

def add_gaussian_noise(ecg_signal, random_generator):
    """Add a small amount of Gaussian noise."""

    noise_level = random_generator.uniform(
        *NOISE_LEVEL_RANGE
    )

    noise = random_generator.normal(
        loc=0.0,
        scale=noise_level,
        size=ecg_signal.shape,
    )

    noisy_signal = ecg_signal + noise

    return noisy_signal


def scale_amplitude(ecg_signal, random_generator):
    """Slightly increase or decrease ECG amplitude."""

    scale_factor = random_generator.uniform(
        *AMPLITUDE_SCALE_RANGE
    )

    scaled_signal = ecg_signal * scale_factor

    return scaled_signal


def shift_signal(ecg_signal, random_generator):
    """Shift the ECG signal slightly left or right."""

    shift_amount = int(
        random_generator.integers(
            -MAX_SHIFT_SAMPLES,
            MAX_SHIFT_SAMPLES + 1,
        )
    )

    if shift_amount == 0:
        return ecg_signal.copy()

    shifted_signal = np.zeros_like(
        ecg_signal
    )

    if shift_amount > 0:
        shifted_signal[shift_amount:] = (
            ecg_signal[:-shift_amount]
        )

    else:
        shifted_signal[:shift_amount] = (
            ecg_signal[-shift_amount:]
        )

    return shifted_signal


def augment_signal(ecg_signal, random_generator):
    """
    Apply mild transformations while preserving
    the main ECG heartbeat morphology.
    """

    augmented_signal = ecg_signal.copy()

    augmented_signal = scale_amplitude(
        augmented_signal,
        random_generator,
    )

    augmented_signal = shift_signal(
        augmented_signal,
        random_generator,
    )

    augmented_signal = add_gaussian_noise(
        augmented_signal,
        random_generator,
    )

    return augmented_signal.astype(
        np.float32,
        copy=False,
    )


# CLASS DISTRIBUTION

def print_class_distribution(labels, title):
    """Print sample count and percentage for each class."""

    print(f"\n{title}")
    print("-" * 40)

    total_samples = len(labels)

    for class_id, class_name in CLASS_NAMES.items():
        class_count = int(
            np.sum(labels == class_id)
        )

        percentage = (
            class_count / total_samples * 100
        )

        print(
            f"{class_name}: "
            f"{class_count:6d} "
            f"({percentage:6.2f}%)"
        )


def calculate_target_count(current_count):
    """
    Calculate a safe augmentation target
    for one minority class.
    """

    maximum_allowed = (
        current_count * MAX_AUGMENT_MULTIPLIER
    )

    target_count = min(
        TARGET_SAMPLES_PER_CLASS,
        maximum_allowed,
    )

    return target_count


# TRAINING DATA AUGMENTATION

def augment_training_data(ecg_signals, labels):
    """
    Generate additional ECG samples for minority classes.

    Class N is not augmented because it already
    contains significantly more samples.
    """

    random_generator = np.random.default_rng(
        RANDOM_SEED
    )

    signal_parts = [ecg_signals]
    label_parts = [labels]

    # Start from class 1 because class 0 (N) is dominant.
    for class_id in range(1, len(CLASS_NAMES)):

        class_indices = np.where(
            labels == class_id
        )[0]

        current_count = len(
            class_indices
        )

        target_count = calculate_target_count(
            current_count
        )

        samples_to_generate = (
            target_count - current_count
        )

        print(
            f"Class {CLASS_NAMES[class_id]}: "
            f"{current_count} -> {target_count}"
        )

        if samples_to_generate <= 0:
            continue

        selected_indices = random_generator.choice(
            class_indices,
            size=samples_to_generate,
            replace=True,
        )

        new_signals = np.empty(
            (
                samples_to_generate,
                HEARTBEAT_LENGTH,
            ),
            dtype=np.float32,
        )

        for new_index, source_index in enumerate(
            selected_indices
        ):
            new_signals[new_index] = augment_signal(
                ecg_signals[source_index],
                random_generator,
            )

        new_labels = np.full(
            samples_to_generate,
            class_id,
            dtype=np.int64,
        )

        signal_parts.append(
            new_signals
        )

        label_parts.append(
            new_labels
        )

    augmented_signals = np.concatenate(
        signal_parts,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    augmented_labels = np.concatenate(
        label_parts,
        axis=0,
    ).astype(
        np.int64,
        copy=False,
    )

    # Shuffle the final training dataset.
    shuffle_indices = random_generator.permutation(
        len(augmented_labels)
    )

    augmented_signals = augmented_signals[
        shuffle_indices
    ]

    augmented_labels = augmented_labels[
        shuffle_indices
    ]

    return augmented_signals, augmented_labels

# CLASS WEIGHTS

def calculate_class_weights(labels):
    """
    Calculate class weights from the original training data.

    These weights are optional and can be used by model members
    instead of augmentation when training a model.
    """

    total_samples = len(labels)
    number_of_classes = len(CLASS_NAMES)

    class_weights = np.zeros(
        number_of_classes,
        dtype=np.float32,
    )

    for class_id in CLASS_NAMES:

        class_count = np.sum(
            labels == class_id
        )

        class_weights[class_id] = (
            total_samples
            / (
                number_of_classes
                * class_count
            )
        )

    return class_weights


# SAVE DATA

def save_augmented_data(
    augmented_signals,
    augmented_labels,
    class_weights,
):
    """Save augmented training data and class weights."""

    AUGMENTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        AUGMENTED_DIR / "X_train_aug.npy",
        augmented_signals,
    )

    np.save(
        AUGMENTED_DIR / "y_train_aug.npy",
        augmented_labels,
    )

    np.save(
        AUGMENTED_DIR / "class_weights.npy",
        class_weights,
    )


# MAIN

def main():

    print("=" * 50)
    print("ECG DATA AUGMENTATION")
    print("=" * 50)

    ecg_signals, labels = load_training_data()

    validate_training_data(
        ecg_signals,
        labels,
    )

    print("\nTraining data loaded successfully.")
    print("ECG signals:", ecg_signals.shape)
    print("Labels     :", labels.shape)

    print_class_distribution(
        labels,
        "BEFORE AUGMENTATION",
    )

    augmented_signals, augmented_labels = (
        augment_training_data(
            ecg_signals,
            labels,
        )
    )

    validate_training_data(
        augmented_signals,
        augmented_labels,
    )

    print_class_distribution(
        augmented_labels,
        "AFTER AUGMENTATION",
    )

    class_weights = calculate_class_weights(
        labels
    )

    save_augmented_data(
        augmented_signals,
        augmented_labels,
        class_weights,
    )

    print("\nClass weights:")

    for class_id, class_name in CLASS_NAMES.items():
        print(
            f"{class_name}: "
            f"{class_weights[class_id]:.4f}"
        )

    print("\nFinal output:")
    print(
        "ECG signals:",
        augmented_signals.shape,
        augmented_signals.dtype,
    )

    print(
        "Labels     :",
        augmented_labels.shape,
        augmented_labels.dtype,
    )

    print("\nAugmentation completed successfully.")
    print(f"Saved to: {AUGMENTED_DIR}")


if __name__ == "__main__":
    main()