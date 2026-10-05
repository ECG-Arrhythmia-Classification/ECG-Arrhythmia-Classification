"""
CNN Training Script for ECG Arrhythmia Classification (Person 3).

Trains the 1D CNN model (ECGCNN) on the processed MIT-BIH dataset:
  - Supports preprocessed or augmented training data.
  - Monitors validation Macro-F1 to select the best checkpoint.
  - Evaluates the best model on the independent test set.
  - Saves best checkpoint, history, config, test metrics, and training curves.
"""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_model import build_cnn_model, count_parameters

# CONFIGURATION & CONSTANTS
ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "processed_data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "cnn"

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
NUM_CLASSES = len(CLASS_NAMES)
INPUT_LENGTH = 180


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(
    data_source: str = "augmented",
    use_class_weights: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train, validation, and test datasets.
    Validation and Test sets always come from preprocessed splits.
    Train set can come from 'augmented', 'preprocessed', or 'split'.
    """
    preprocessed_dir = PROCESSED_DIR / "preprocessed"
    augmented_dir = PROCESSED_DIR / "augmented"
    split_dir = PROCESSED_DIR / "split"

    # Always use preprocessed data for validation and test if available
    if (preprocessed_dir / "X_val.npy").exists():
        val_dir = preprocessed_dir
        test_dir = preprocessed_dir
    else:
        val_dir = split_dir
        test_dir = split_dir

    X_val = np.load(val_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(val_dir / "y_val.npy").astype(np.int64)
    X_test = np.load(test_dir / "X_test.npy").astype(np.float32)
    y_test = np.load(test_dir / "y_test.npy").astype(np.int64)

    # Determine training source
    if data_source == "augmented" and (augmented_dir / "X_train_aug.npy").exists():
        print(f"Loading augmented training data from {augmented_dir}...")
        X_train = np.load(augmented_dir / "X_train_aug.npy").astype(np.float32)
        y_train = np.load(augmented_dir / "y_train_aug.npy").astype(np.int64)
    elif data_source == "preprocessed" and (preprocessed_dir / "X_train.npy").exists():
        print(f"Loading preprocessed training data from {preprocessed_dir}...")
        X_train = np.load(preprocessed_dir / "X_train.npy").astype(np.float32)
        y_train = np.load(preprocessed_dir / "y_train.npy").astype(np.int64)
    else:
        print(f"Loading baseline split training data from {split_dir}...")
        X_train = np.load(split_dir / "X_train.npy").astype(np.float32)
        y_train = np.load(split_dir / "y_train.npy").astype(np.int64)

    # Compute or load class weights
    if use_class_weights:
        if (augmented_dir / "class_weights.npy").exists():
            class_weights = np.load(augmented_dir / "class_weights.npy").astype(np.float32)
        else:
            total_samples = len(y_train)
            class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)
            for c in range(NUM_CLASSES):
                count = np.sum(y_train == c)
                class_weights[c] = total_samples / (NUM_CLASSES * max(count, 1))
    else:
        class_weights = None

    return X_train, y_train, X_val, y_val, X_test, y_test, class_weights


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 128,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap numpy arrays into PyTorch DataLoaders."""
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a DataLoader.
    Returns: (average_loss, accuracy, macro_f1, all_targets, all_predictions)
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            total_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=1)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    total_samples = len(loader.dataset)
    avg_loss = total_loss / total_samples
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, macro_f1, all_targets, all_preds


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train for one single epoch."""
    model.train()
    total_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        preds = torch.argmax(logits, dim=1)

        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    total_samples = len(loader.dataset)
    avg_loss = total_loss / total_samples
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, macro_f1


def plot_training_curves(history: Dict, save_path: Path) -> None:
    """Plot and save training & validation Loss, Accuracy, and Macro-F1 curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r--s", label="Val Loss", markersize=4)
    axes[0].set_title("Cross-Entropy Loss vs Epoch", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Accuracy plot
    axes[1].plot(epochs, [x * 100 for x in history["train_acc"]], "b-o", label="Train Acc (%)", markersize=4)
    axes[1].plot(epochs, [x * 100 for x in history["val_acc"]], "g--s", label="Val Acc (%)", markersize=4)
    axes[1].set_title("Accuracy vs Epoch", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    # Macro-F1 plot
    axes[2].plot(epochs, history["train_macro_f1"], "b-o", label="Train Macro-F1", markersize=4)
    axes[2].plot(epochs, history["val_macro_f1"], "m--s", label="Val Macro-F1 (Selection)", markersize=4)
    best_ep = int(np.argmax(history["val_macro_f1"])) + 1
    best_f1 = max(history["val_macro_f1"])
    axes[2].scatter(best_ep, best_f1, color="red", s=100, zorder=5, label=f"Best Val F1: {best_f1:.4f} (Ep {best_ep})")
    axes[2].set_title("Macro-F1 Score vs Epoch", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("Macro-F1", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: Path,
    title: str = "Test Confusion Matrix",
) -> None:
    """Plot and save both counts and normalized confusion matrix."""
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Raw counts
    im0 = axes[0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title(f"{title} (Counts)", fontsize=12, fontweight="bold")
    fig.colorbar(im0, ax=axes[0])
    tick_marks = np.arange(len(class_names))
    axes[0].set_xticks(tick_marks)
    axes[0].set_yticks(tick_marks)
    axes[0].set_xticklabels(class_names)
    axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # Normalized
    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    axes[1].set_title(f"{title} (Normalized)", fontsize=12, fontweight="bold")
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_xticks(tick_marks)
    axes[1].set_yticks(tick_marks)
    axes[1].set_xticklabels(class_names)
    axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Label", fontsize=11)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[1].text(
                j, i, f"{cm_norm[i, j]:.2f}",
                horizontalalignment="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CNN model for ECG classification.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (epochs).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--data_source",
        type=str,
        default="augmented",
        choices=["augmented", "preprocessed", "split"],
        help="Training dataset source.",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        default=False,
        help="Use weighted cross-entropy loss.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save model checkpoints and results.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("ECG CNN TRAINING (Person 3)")
    print("=" * 60)
    print(f"Device:               {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Data source:          {args.data_source}")
    print(f"Output directory:     {output_dir}")
    print(f"Hyperparameters:      epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, weight_decay={args.weight_decay}")

    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test, class_weights = load_data(
        data_source=args.data_source,
        use_class_weights=args.use_class_weights,
    )

    print(f"\nDataset loaded:")
    print(f"  Train:      X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:       X={X_test.shape}, y={y_test.shape}")

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=args.batch_size
    )

    # Model instantiation
    model = build_cnn_model(
        in_channels=1,
        num_classes=NUM_CLASSES,
        input_length=INPUT_LENGTH,
        dropout=args.dropout,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters:     {trainable_params:,} trainable ({total_params:,} total)")

    # Loss function & Optimizer
    if class_weights is not None:
        print(f"Applying class weights to CrossEntropyLoss: {class_weights}")
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    # Checkpoint & Tracking variables
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = output_dir / "best_cnn_model.pt"

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
    }

    print("\nStarting training loop...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Scheduler step based on validation Macro-F1
        scheduler.step(val_f1)

        # Record history
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["train_macro_f1"].append(float(train_f1))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_f1))
        history["lr"].append(float(current_lr))

        epoch_duration = time.time() - epoch_start

        # Check for best validation Macro-F1
        is_best = val_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_f1,
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "config": vars(args),
                },
                best_checkpoint_path,
            )
            marker = " (*) BEST VAL F1"
        else:
            patience_counter += 1
            marker = ""

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s) "
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:6.2f}% | F1: {train_f1:.4f} || "
            f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:6.2f}% | F1: {val_f1:.4f} (lr={current_lr:.1e}){marker}"
        )

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs).")
            break

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.1f}s.")
    print(f"Best model achieved at Epoch {best_epoch} with Validation Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Best checkpoint saved at: {best_checkpoint_path}")

    # Save training curves
    curves_path = output_dir / "training_curves.png"
    plot_training_curves(history, curves_path)

    # Save history json
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Save experiment config
    config_dict = vars(args)
    config_dict["best_epoch"] = best_epoch
    config_dict["best_val_macro_f1"] = float(best_val_macro_f1)
    config_dict["total_training_time_seconds"] = round(total_training_time, 2)
    config_dict["device"] = str(device)
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved to {config_path}")

    # EVALUATION ON TEST SET USING BEST CHECKPOINT
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET (Best Checkpoint)")
    print("=" * 60)

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_macro_f1, y_test_true, y_test_pred = evaluate(
        model, test_loader, criterion, device
    )

    test_weighted_f1 = f1_score(y_test_true, y_test_pred, average="weighted", zero_division=0)
    test_macro_prec = precision_score(y_test_true, y_test_pred, average="macro", zero_division=0)
    test_macro_rec = recall_score(y_test_true, y_test_pred, average="macro", zero_division=0)

    # Per-class metrics
    per_class_f1 = f1_score(y_test_true, y_test_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_test_true, y_test_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_test_true, y_test_pred, average=None, zero_division=0)

    print(f"Test Loss:           {test_loss:.4f}")
    print(f"Test Accuracy:       {test_acc * 100:.2f}%")
    print(f"Test Macro-F1:       {test_macro_f1:.4f}")
    print(f"Test Weighted-F1:    {test_weighted_f1:.4f}")
    print(f"Test Macro-Precision:{test_macro_prec:.4f}")
    print(f"Test Macro-Recall:   {test_macro_rec:.4f}")

    print("\nDetailed Classification Report:")
    report_str = classification_report(
        y_test_true,
        y_test_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    print(report_str)

    # Compute and plot test confusion matrix
    cm = confusion_matrix(y_test_true, y_test_pred, labels=list(range(NUM_CLASSES)))
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path, title="Test Confusion Matrix")

    # Save detailed test metrics
    test_metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "test_macro_precision": float(test_macro_prec),
        "test_macro_recall": float(test_macro_rec),
        "per_class_metrics": {
            CLASS_NAMES[i]: {
                "precision": float(per_class_prec[i]),
                "recall": float(per_class_rec[i]),
                "f1_score": float(per_class_f1[i]),
                "support": int(np.sum(y_test_true == i)),
            }
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1),
    }

    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Test metrics saved to {metrics_path}")

    print("\nPerson 3 CNN implementation and baseline training successfully completed!")


if __name__ == "__main__":
    main()
