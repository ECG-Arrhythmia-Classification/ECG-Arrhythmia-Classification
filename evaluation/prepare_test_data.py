from pathlib import Path
import sys
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from preprocessing import preprocess_data

INPUT_DIR = BASE_DIR / "processed_data" / "split"
OUTPUT_DIR = BASE_DIR / "processed_data" / "preprocessed"

X_TEST_PATH = INPUT_DIR / "X_test.npy"
Y_TEST_PATH = INPUT_DIR / "y_test.npy"


def main():
    print("Loading test data...")

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print("Before preprocessing:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("\nPreprocessing test data...")

    X_test_processed = preprocess_data(X_test)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test_processed
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test
    )

    print("\nFinished.")
    print("Processed X_test:", X_test_processed.shape)
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()