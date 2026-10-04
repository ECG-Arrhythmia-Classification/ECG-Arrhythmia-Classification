import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = "processed_data/split"
CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}


def main():
    X = np.load(f"{DATA_DIR}/X_train.npy")
    y = np.load(f"{DATA_DIR}/y_train.npy")

    # Plot one example heartbeat from each AAMI class.
    plt.figure(figsize=(12, 10))
    for class_id, class_name in CLASS_NAMES.items():
        indices = np.where(y == class_id)[0]
        if len(indices) == 0:
            continue

        heartbeat = X[indices[0]]
        plt.subplot(5, 1, class_id + 1)
        plt.plot(heartbeat)
        plt.title(f"Class {class_name}")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude (mV)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
