from pathlib import Path
import sys
import json
import time

import numpy as np
import torch

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

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

OUTPUT_DIR = BASE_DIR / "results" / "benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from cnn_model import build_cnn_model
from rnn_model import ECG_RNN
from transformer_model import ECGTransformer
# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def count_trainable_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def checkpoint_size_mb(path):
    return path.stat().st_size / (1024 * 1024)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

def load_test_data():
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    return X_test, y_test


# --------------------------------------------------
# LOAD CNN
# --------------------------------------------------

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

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


# --------------------------------------------------
# LOAD RNN
# --------------------------------------------------

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

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model
# --------------------------------------------------
# LOAD TRANSFORMER
# --------------------------------------------------

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


# --------------------------------------------------
# GENERIC BENCHMARK
# --------------------------------------------------

def benchmark_model(
    model,
    X,
    device,
    model_name,
    batch_size=256,
    warmup_batches=5
):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    total_samples = len(X_tensor)

    # Warm-up
    with torch.no_grad():
        for i in range(min(warmup_batches, 10)):
            start = i * batch_size
            end = min(
                start + batch_size,
                total_samples
            )

            if start >= total_samples:
                break

            batch = X_tensor[
                start:end
            ].to(device)

            _ = model(batch)

    synchronize(device)

    start_time = time.perf_counter()

    with torch.no_grad():

        for start in range(
            0,
            total_samples,
            batch_size
        ):

            batch = X_tensor[
                start:start + batch_size
            ].to(device)

            _ = model(batch)

    synchronize(device)

    end_time = time.perf_counter()

    total_time = end_time - start_time

    avg_time_per_sample = (
        total_time / total_samples
    )

    throughput = (
        total_samples / total_time
    )

    params = count_trainable_parameters(
        model
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(
        f"Trainable parameters : {params:,}"
    )

    print(
        f"Total inference time : "
        f"{total_time:.6f} seconds"
    )

    print(
        f"Time/sample          : "
        f"{avg_time_per_sample * 1000:.6f} ms"
    )

    print(
        f"Throughput           : "
        f"{throughput:.2f} samples/sec"
    )

    return {
        "model": model_name,
        "trainable_parameters": params,
        "total_inference_time_sec": total_time,
        "avg_inference_time_per_sample_ms":
            avg_time_per_sample * 1000,
        "throughput_samples_per_sec":
            throughput
    }


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

    # --------------------------------------------------
    # CNN
    # --------------------------------------------------

    print("\nLoading CNN...")

    cnn = load_cnn(device)

    cnn_result = benchmark_model(
        cnn,
        X_test,
        device,
        "CNN"
    )

    cnn_result["checkpoint_size_mb"] = (
        checkpoint_size_mb(
            CNN_CHECKPOINT
        )
    )

    results["CNN"] = cnn_result

    print(
        f"Checkpoint size      : "
        f"{cnn_result['checkpoint_size_mb']:.3f} MB"
    )

    # --------------------------------------------------
    # RNN
    # --------------------------------------------------

    print("\nLoading RNN/LSTM...")

    rnn = load_rnn(device)

    X_rnn = X_test.reshape(
        -1,
        180,
        1
    )

    rnn_result = benchmark_model(
        rnn,
        X_rnn,
        device,
        "RNN_LSTM"
    )

    rnn_result["checkpoint_size_mb"] = (
        checkpoint_size_mb(
            RNN_CHECKPOINT
        )
    )

    results["RNN_LSTM"] = rnn_result

    print(
        f"Checkpoint size      : "
        f"{rnn_result['checkpoint_size_mb']:.3f} MB"
    )
    # --------------------------------------------------
    # TRANSFORMER
    # --------------------------------------------------

    print("\nLoading Transformer...")

    transformer = load_transformer(
        device
    )

    transformer_result = benchmark_model(
        transformer,
        X_test,
        device,
        "Transformer"
    )

    transformer_result["checkpoint_size_mb"] = (
        checkpoint_size_mb(
            TRANSFORMER_CHECKPOINT
        )
    )

    results["Transformer"] = (
        transformer_result
    )

    print(
        f"Checkpoint size      : "
        f"{transformer_result['checkpoint_size_mb']:.3f} MB"
    )
    # --------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        "benchmark_results.json"
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

    print("\nBenchmark results saved to:")
    print(output_file)


if __name__ == "__main__":
    main()