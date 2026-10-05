"""Train the ECG Transformer using the supplied NumPy files."""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from transformer_model import ECGTransformer


# Data and training settings: edit these values to change the run.
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "processed_data"
OUTPUT_DIR = ROOT_DIR / "results" / "transformer"
EPOCHS = 20
BATCH_SIZE = 256
DIMENSION = 32
HEADS = 4
LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.0003
PATIENCE = 5
CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def get_metrics(actual, predicted):
    """Calculate accuracy and macro-F1 without extra metric dependencies."""
    accuracy = float(np.mean(actual == predicted))
    class_f1 = []
    for class_id in range(len(CLASS_NAMES)):
        true_positive = np.sum((actual == class_id) & (predicted == class_id))
        false_positive = np.sum((actual != class_id) & (predicted == class_id))
        false_negative = np.sum((actual == class_id) & (predicted != class_id))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        class_f1.append(2 * precision * recall / max(1e-12, precision + recall))
    return accuracy, float(np.mean(class_f1))


def load_data():
    """Load normalized splits and the class weights for the original train set."""
    arrays = {
        "train_x": np.load(DATA_DIR / "preprocessed" / "X_train.npy"),
        "train_y": np.load(DATA_DIR / "preprocessed" / "y_train.npy"),
        "val_x": np.load(DATA_DIR / "preprocessed" / "X_val.npy"),
        "val_y": np.load(DATA_DIR / "preprocessed" / "y_val.npy"),
        "test_x": np.load(DATA_DIR / "preprocessed" / "X_test.npy"),
        "test_y": np.load(DATA_DIR / "preprocessed" / "y_test.npy"),
        "class_weights": np.load(DATA_DIR / "augmented" / "class_weights.npy"),
    }
    for x_name, y_name in [("train_x", "train_y"), ("val_x", "val_y"), ("test_x", "test_y")]:
        x, y = arrays[x_name], arrays[y_name]
        if x.ndim != 2 or x.shape[1] != 180 or y.shape != (len(x),):
            raise ValueError(f"Bad shapes: {x_name}={x.shape}, {y_name}={y.shape}")
        if not np.isfinite(x).all() or not np.isin(y, range(5)).all():
            raise ValueError(f"Invalid values in {x_name} or {y_name}")
        arrays[x_name] = np.asarray(x, dtype=np.float32)
        arrays[y_name] = np.asarray(y, dtype=np.int64)
    return arrays


def make_loader(x, y, shuffle=False):
    data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(data, batch_size=BATCH_SIZE, shuffle=shuffle)


def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss, actual, predicted = 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_function(logits, y).item() * len(y)
            actual.append(y.cpu().numpy())
            predicted.append(logits.argmax(1).cpu().numpy())
    actual = np.concatenate(actual)
    predicted = np.concatenate(predicted)
    accuracy, macro_f1 = get_metrics(actual, predicted)
    return {
        "loss": total_loss / len(actual),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def save_progress(history):
    """Save the curve and CSV after each epoch so progress is visible."""
    with (OUTPUT_DIR / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Accuracy")
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="Macro F1")
    axes[1].set(title="Validation", xlabel="Epoch", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curve.png", dpi=150)
    plt.close(fig)


def save_attention(model, x, y, device):
    model.eval()
    with torch.no_grad():
        _, attention = model(torch.from_numpy(x[:1]).to(device), show_attention=True)
    weights = attention[0, :, 0, 1:].mean(0).cpu().numpy()
    sample_positions = np.arange(len(weights)) * model.patch_size + model.patch_size / 2

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(x[0])
    axes[0].set_ylabel("ECG amplitude")
    axes[0].set_title(f"Example heartbeat (class {CLASS_NAMES[int(y[0])]})")
    axes[1].plot(sample_positions, weights, color="darkorange")
    axes[1].set(xlabel="Sample", ylabel="Attention", title="Last-layer CLS attention")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "attention_example.png", dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    data = load_data()
    train_loader = make_loader(data["train_x"], data["train_y"], shuffle=True)
    val_loader = make_loader(data["val_x"], data["val_y"])
    test_loader = make_loader(data["test_x"], data["test_y"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGTransformer(
        dimension=DIMENSION, heads=HEADS, layers=LAYERS, dropout=DROPOUT
    ).to(device)
    class_weights = torch.tensor(data["class_weights"], dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history, best_f1, best_val_accuracy, best_epoch, wait = [], -1, 0, 0, 0
    print(f"Device: {device}; train={len(data['train_y'])}, "
          f"validation={len(data['val_y'])}, test={len(data['test_y'])}", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch_number, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            if batch_number % 50 == 0:
                print(f"Epoch {epoch}: batch {batch_number}/{len(train_loader)}", flush=True)

        val = evaluate(model, val_loader, loss_function, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(data["train_y"]),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        history.append(row)
        save_progress(history)
        print(f"Epoch {epoch}/{EPOCHS}: val accuracy={val['accuracy']:.4f}, "
              f"macro-F1={val['macro_f1']:.4f}", flush=True)

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, wait = val["macro_f1"], epoch, 0
            best_val_accuracy = val["accuracy"]
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
        else:
            wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping; best epoch was {best_epoch}.", flush=True)
            break

    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt", map_location=device))
    test = evaluate(model, test_loader, loss_function, device)
    metrics = {
        "best_epoch": best_epoch,
        "validation_accuracy": best_val_accuracy,
        "validation_macro_f1": best_f1,
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_attention(model, data["test_x"], data["test_y"], device)
    print(f"Test accuracy={test['accuracy']:.4f}, macro-F1={test['macro_f1']:.4f}")
    print(f"Saved results to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
