from pathlib import Path
import sys
import json

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# QUAN TRỌNG:
# robustness phải bắt đầu từ ECG trước preprocessing
RAW_DATA_DIR = BASE_DIR / "processed_data" / "split"

X_TEST_PATH = RAW_DATA_DIR / "X_test.npy"
Y_TEST_PATH = RAW_DATA_DIR / "y_test.npy"

CNN_CHECKPOINT = (
    BASE_DIR / "results" / "cnn" / "best_cnn_model.pt"
)

RNN_CHECKPOINT = (
    BASE_DIR / "saved_models" / "best_rnn_model.pth"
)
TRANSFORMER_CHECKPOINT = (
    BASE_DIR / "results" / "transformer" / "best_model.pt"
)
OUTPUT_DIR = BASE_DIR / "results" / "robustness"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==================================================
# IMPORT PROJECT CODE
# ==================================================

from cnn_model import build_cnn_model
from rnn_model import ECG_RNN
from transformer_model import ECGTransformer
# sử dụng đúng preprocessing của Người 2
from preprocessing import preprocess_data


# ==================================================
# ADD NOISE BY SNR
# ==================================================

def add_gaussian_noise_snr(
    X,
    snr_db,
    seed=42
):
    """
    Thêm Gaussian noise theo SNR cho từng heartbeat.

    SNR càng cao -> noise càng thấp.

    Ví dụ:
        30 dB = noise thấp
        20 dB = noise vừa
        10 dB = noise cao
    """

    rng = np.random.default_rng(seed)

    X = X.astype(np.float32)

    # Công suất tín hiệu của từng heartbeat
    signal_power = np.mean(
        X ** 2,
        axis=1,
        keepdims=True
    )

    # Tránh trường hợp signal power = 0
    signal_power = np.maximum(
        signal_power,
        1e-12
    )

    # SNR(dB) = 10 log10(Psignal / Pnoise)
    noise_power = (
        signal_power /
        (10 ** (snr_db / 10.0))
    )

    noise_std = np.sqrt(
        noise_power
    )

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=X.shape
    ).astype(np.float32)

    noise = noise * noise_std

    X_noisy = X + noise

    return X_noisy.astype(np.float32)


# ==================================================
# LOAD DATA
# ==================================================

def load_raw_test_data():

    X_test = np.load(
        X_TEST_PATH
    ).astype(np.float32)

    y_test = np.load(
        Y_TEST_PATH
    )

    print("Raw X_test:", X_test.shape)
    print("y_test    :", y_test.shape)

    return X_test, y_test


# ==================================================
# LOAD CNN
# ==================================================

def load_cnn(device):

    model = build_cnn_model(
        in_channels=1,
        num_classes=5,
        input_length=180,
        dropout=0.3
    )

    checkpoint = torch.load(
        CNN_CHECKPOINT,
        map_location=device
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.to(device)
    model.eval()

    return model

# ==================================================
# LOAD TRANSFORMER
# ==================================================

def load_transformer(device):

    model = ECGTransformer(
        input_length=180,
        classes=5,
        dimension=32,
        heads=4,
        layers=2,
        dropout=0.2,
        patch_size=4
    )

    checkpoint = torch.load(
        TRANSFORMER_CHECKPOINT,
        map_location=device
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:
            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:
            model.load_state_dict(
                checkpoint
            )

    else:
        model.load_state_dict(
            checkpoint
        )

    model.to(device)
    model.eval()

    return model


# ==================================================
# LOAD RNN
# ==================================================

def load_rnn(device):

    model = ECG_RNN(
        input_size=1,
        hidden_size=32,
        num_layers=2,
        num_classes=5,
        dropout=0.5,
        model_type="lstm"
    )

    state_dict = torch.load(
        RNN_CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    return model


# ==================================================
# PREDICT CNN
# ==================================================

def predict_cnn(
    model,
    X,
    device,
    batch_size=256
):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            batch_size
        ):

            batch = X_tensor[
                start:start + batch_size
            ].to(device)

            outputs = model(batch)

            pred = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                pred.cpu().numpy()
            )

    return np.array(predictions)


# ==================================================
# PREDICT RNN
# ==================================================

def predict_rnn(
    model,
    X,
    device,
    batch_size=256
):

    X_rnn = X.reshape(
        -1,
        180,
        1
    )

    X_tensor = torch.tensor(
        X_rnn,
        dtype=torch.float32
    )

    predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            batch_size
        ):

            batch = X_tensor[
                start:start + batch_size
            ].to(device)

            outputs = model(batch)

            pred = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                pred.cpu().numpy()
            )

    return np.array(predictions)
# ==================================================
# PREDICT TRANSFORMER
# ==================================================

def predict_transformer(
    model,
    X,
    device,
    batch_size=256
):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    predictions = []

    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            batch_size
        ):

            batch = X_tensor[
                start:start + batch_size
            ].to(device)

            outputs = model(batch)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            pred = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                pred.cpu().numpy()
            )

    return np.array(
        predictions
    )


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }


# ==================================================
# CREATE TEST CONDITIONS
# ==================================================

def create_test_conditions(
    X_raw
):

    conditions = {}

    # Clean:
    # vẫn phải đi qua preprocessing giống lúc model được test
    print("\nPreparing CLEAN data...")

    conditions["clean"] = {
        "snr_db": None,
        "X": preprocess_data(
            X_raw
        )
    }

    # Noise thấp
    print(
        "Preparing LOW noise (30 dB)..."
    )

    X_low_raw = add_gaussian_noise_snr(
        X_raw,
        snr_db=30,
        seed=42
    )

    conditions["low_30db"] = {
        "snr_db": 30,
        "X": preprocess_data(
            X_low_raw
        )
    }

    # Noise vừa
    print(
        "Preparing MEDIUM noise (20 dB)..."
    )

    X_medium_raw = add_gaussian_noise_snr(
        X_raw,
        snr_db=20,
        seed=42
    )

    conditions["medium_20db"] = {
        "snr_db": 20,
        "X": preprocess_data(
            X_medium_raw
        )
    }

    # Noise cao
    print(
        "Preparing HIGH noise (10 dB)..."
    )

    X_high_raw = add_gaussian_noise_snr(
        X_raw,
        snr_db=10,
        seed=42
    )

    conditions["high_10db"] = {
        "snr_db": 10,
        "X": preprocess_data(
            X_high_raw
        )
    }

    return conditions


# ==================================================
# TEST ONE MODEL
# ==================================================

def test_model_robustness(
    model,
    model_name,
    conditions,
    y_test,
    device,
    predict_function
):

    print("\n" + "=" * 70)
    print(
        f"ROBUSTNESS TEST: {model_name}"
    )
    print("=" * 70)

    model_results = {}

    clean_macro_f1 = None
    clean_accuracy = None

    for condition_name, data in (
        conditions.items()
    ):

        X_condition = data["X"]

        y_pred = predict_function(
            model,
            X_condition,
            device
        )

        metrics = calculate_metrics(
            y_test,
            y_pred
        )

        if condition_name == "clean":

            clean_macro_f1 = (
                metrics["macro_f1"]
            )

            clean_accuracy = (
                metrics["accuracy"]
            )

        macro_f1_drop = (
            clean_macro_f1
            - metrics["macro_f1"]
        )

        accuracy_drop = (
            clean_accuracy
            - metrics["accuracy"]
        )

        # % giảm Macro F1 so với clean
        if clean_macro_f1 > 0:

            macro_f1_drop_percent = (
                macro_f1_drop
                / clean_macro_f1
                * 100
            )

        else:
            macro_f1_drop_percent = 0.0

        metrics["snr_db"] = (
            data["snr_db"]
        )

        metrics[
            "accuracy_drop_from_clean"
        ] = float(
            accuracy_drop
        )

        metrics[
            "macro_f1_drop_from_clean"
        ] = float(
            macro_f1_drop
        )

        metrics[
            "macro_f1_drop_percent"
        ] = float(
            macro_f1_drop_percent
        )

        model_results[
            condition_name
        ] = metrics

        print(
            f"\nCondition : {condition_name}"
        )

        if data["snr_db"] is None:
            print(
                "SNR       : clean"
            )
        else:
            print(
                f"SNR       : "
                f"{data['snr_db']} dB"
            )

        print(
            f"Accuracy  : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Macro F1  : "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Weighted F1: "
            f"{metrics['weighted_f1']:.4f}"
        )

        print(
            f"Acc drop  : "
            f"{accuracy_drop:.4f}"
        )

        print(
            f"F1 drop   : "
            f"{macro_f1_drop:.4f}"
        )

        print(
            f"F1 drop % : "
            f"{macro_f1_drop_percent:.2f}%"
        )

    return model_results


# ==================================================
# MAIN
# ==================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("ECG ROBUSTNESS TEST")
    print("=" * 70)

    print(
        "Device:",
        device
    )

    # ----------------------------------------------
    # RAW DATA
    # ----------------------------------------------

    X_raw, y_test = (
        load_raw_test_data()
    )

    # ----------------------------------------------
    # CREATE CLEAN / NOISY DATA
    # ----------------------------------------------

    conditions = (
        create_test_conditions(
            X_raw
        )
    )
    
        # ----------------------------------------------
    # LOAD MODELS
    # ----------------------------------------------

    print("\nLoading models...")

    cnn = load_cnn(
        device
    )

    rnn = load_rnn(
        device
    )

    transformer = load_transformer(
        device
    )

    results = {}
   
    # ----------------------------------------------
    # CNN
    # ----------------------------------------------

    results["CNN"] = (
        test_model_robustness(
            cnn,
            "CNN",
            conditions,
            y_test,
            device,
            predict_cnn
        )
    )

    # ----------------------------------------------
    # RNN
    # ----------------------------------------------

    results["RNN_LSTM"] = (
        test_model_robustness(
            rnn,
            "RNN_LSTM",
            conditions,
            y_test,
            device,
            predict_rnn
        )
    )
    results["Transformer"] = (
    test_model_robustness(
        transformer,
        "Transformer",
        conditions,
        y_test,
        device,
        predict_transformer
    )
)

    # ----------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "robustness_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print("\n" + "=" * 70)

    print(
        "Robustness results saved to:"
    )

    print(
        output_file
    )


if __name__ == "__main__":
    main()