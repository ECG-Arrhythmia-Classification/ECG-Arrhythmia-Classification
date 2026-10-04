import os

import numpy as np


DATA_DIR = "processed_data"
OUTPUT_DIR = os.path.join(DATA_DIR, "split")
CLASS_NAMES = ["N", "S", "V", "F", "Q"]

# Records are split by record, not by individual heartbeat, to avoid leakage.
TRAIN_RECORDS = [
    "100", "101", "102", "103", "105", "106", "107", "108", "109", "111",
    "112", "113", "115", "116", "117", "119", "121", "122", "123", "124",
    "200", "201", "202", "207", "208", "212", "214", "219", "220", "221",
    "228", "230", "231", "232", "234",
]
VAL_RECORDS = ["118", "203", "213", "215", "217", "222"]
TEST_RECORDS = ["104", "114", "205", "209", "210", "223", "233"]


def check_record_split(record_ids):
    # A record must be assigned to exactly one split.
    assigned_records = TRAIN_RECORDS + VAL_RECORDS + TEST_RECORDS

    if len(assigned_records) != len(set(assigned_records)):
        raise ValueError("A record appears in more than one dataset split.")

    available_records = set(record_ids)
    missing_records = available_records - set(assigned_records)
    unknown_records = set(assigned_records) - available_records

    if missing_records:
        raise ValueError(f"Records not assigned to a split: {sorted(missing_records)}")
    if unknown_records:
        raise ValueError(f"Records not found in the dataset: {sorted(unknown_records)}")


def print_split_summary(name, X, y):
    # Print the size and class balance of one split.
    print(f"\n{name}: {X.shape}")
    for class_id, class_name in enumerate(CLASS_NAMES):
        count = np.sum(y == class_id)
        percentage = 100 * count / len(y)
        print(f"  {class_name}: {count} ({percentage:.2f}%)")


def main():
    # Load the dataset created by prepare_dataset.py.
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    record_ids = np.load(os.path.join(DATA_DIR, "record_ids.npy"))
    check_record_split(record_ids)

    # Define the record list for each output split.
    split_records = {
        "train": TRAIN_RECORDS,
        "val": VAL_RECORDS,
        "test": TEST_RECORDS,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, records in split_records.items():
        # Select all heartbeat samples that belong to these records.
        mask = np.isin(record_ids, records)
        X_split = X[mask]
        y_split = y[mask]

        print_split_summary(name.capitalize(), X_split, y_split)
        # Save feature and label arrays for model training.
        np.save(os.path.join(OUTPUT_DIR, f"X_{name}.npy"), X_split)
        np.save(os.path.join(OUTPUT_DIR, f"y_{name}.npy"), y_split)

    print("\nDataset split completed.")


if __name__ == "__main__":
    main()
