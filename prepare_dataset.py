import os

import numpy as np
import wfdb


DATA_DIR = "data"
OUTPUT_DIR = "processed_data"
# Extract a 180-sample window centred on each annotated heartbeat.
BEFORE = 90
AFTER = 90

# AAMI five-class grouping used for heartbeat classification.
CLASS_MAP = {
    #N: Normal
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    # S: Supraventricular ectopic    
    "A": 1, "a": 1, "J": 1, "S": 1,
    # V: Ventricular ectopic
    "V": 2, "E": 2,
    # F: Fusion
    "F": 3,
    # Q: Unknown / paced / other
    "Q": 4, "/": 4, "f": 4,
}
CLASS_NAMES = ["N", "S", "V", "F", "Q"]

RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def print_class_distribution(labels):
    # Show the number of beats in each AAMI class.
    print("\nClass distribution:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {np.sum(labels == class_id)}")


def main():
    # Lists are used first because the number of valid beats is not known in advance.
    heartbeats = []
    labels = []
    record_ids = []

    for record_name in RECORDS:
        print("Processing record", record_name)
        record_path = os.path.join(DATA_DIR, record_name)

        # Load the signal and its reference annotations.
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")
        signal = record.p_signal[:, 0]

        for sample, symbol in zip(annotation.sample, annotation.symbol):
            # Ignore annotations that are outside the selected AAMI classes.
            if symbol not in CLASS_MAP:
                continue

            start = sample - BEFORE
            end = sample + AFTER
            if start < 0 or end > len(signal):
                continue

            # Save the heartbeat segment, its class, and its source record.
            heartbeats.append(signal[start:end])
            labels.append(CLASS_MAP[symbol])
            record_ids.append(record_name)

    # Convert the collected lists to arrays before saving them.
    X = np.array(heartbeats)
    y = np.array(labels)
    record_ids = np.array(record_ids)

    print("\nDataset created")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print_class_distribution(y)

    # Save the features, labels, and record identifiers separately.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, "record_ids.npy"), record_ids)
    print("\nDataset saved in processed_data/")


if __name__ == "__main__":
    main()
