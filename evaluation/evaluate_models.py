from pathlib import Path
import sys
import json

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "processed_data" / "preprocessed"

X_TEST_PATH = DATA_DIR / "X_test.npy"
Y_TEST_PATH = DATA_DIR / "y_test.npy"

CNN_CHECKPOINT = (
    BASE_DIR / "results" / "cnn" / "best_cnn_model.pt"
)

RNN_CHECKPOINT = (
    BASE_DIR / "saved_models" / "best_rnn_model.pth"
)
TRANSFORMER_CHECKPOINT = (
    BASE_DIR / "results" / "transformer" / "best_model.pt"
)

OUTPUT_DIR = BASE_DIR / "results" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# cho phép import cnn_model.py và rnn_model.py
sys.path.append(str(BASE_DIR))

from cnn_model import build_cnn_model
from rnn_model import ECG_RNN
from transformer_model import ECGTransformer

CLASS_NAMES = ["N", "S", "V", "F", "Q"]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_test_data():

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    return X_test, y_test


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(y_true, y_pred, model_name):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro Precision : {precision_macro:.4f}")
    print(f"Macro Recall    : {recall_macro:.4f}")
    print(f"Macro F1        : {f1_macro:.4f}")
    print(f"Weighted F1     : {f1_weighted:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(cm)

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist()
    }

    return metrics


# --------------------------------------------------
# CNN
# --------------------------------------------------

def evaluate_cnn(X_test, y_test, device):

    print("\nLoading CNN...")

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

    # hỗ trợ cả trường hợp checkpoint là state_dict trực tiếp
    # hoặc dict chứa model_state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    batch_size = 256

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

    y_pred = np.array(predictions)

    metrics = calculate_metrics(
        y_test,
        y_pred,
        "CNN"
    )

    return metrics


# --------------------------------------------------
# RNN
# --------------------------------------------------

def evaluate_rnn(X_test, y_test, device):

    print("\nLoading RNN/LSTM...")

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

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # RNN yêu cầu:
    # (batch, sequence_length, input_size)
    X_rnn = X_test.reshape(
        -1,
        180,
        1
    )

    X_tensor = torch.tensor(
        X_rnn,
        dtype=torch.float32
    )

    batch_size = 256

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

    y_pred = np.array(predictions)

    metrics = calculate_metrics(
        y_test,
        y_pred,
        "RNN_LSTM"
    )

    return metrics
# --------------------------------------------------
# TRANSFORMER
# --------------------------------------------------

def evaluate_transformer(X_test, y_test, device):

    print("\nLoading Transformer...")

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

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    batch_size = 256
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

    y_pred = np.array(
        predictions
    )

    metrics = calculate_metrics(
        y_test,
        y_pred,
        "Transformer"
    )

    return metrics

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(results):

    output_file = (
        OUTPUT_DIR /
        "evaluation_results.json"
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

    print(
        "\nResults saved to:"
    )

    print(output_file)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    X_test, y_test = load_test_data()

    results = {}

    # CNN
    results["CNN"] = evaluate_cnn(
        X_test,
        y_test,
        device
    )

    # RNN / LSTM
    results["RNN_LSTM"] = evaluate_rnn(
        X_test,
        y_test,
        device
    )

    # Transformer
    results["Transformer"] = evaluate_transformer(
        X_test,
        y_test,
        device
    )

    save_results(results)

if __name__ == "__main__":
    main()