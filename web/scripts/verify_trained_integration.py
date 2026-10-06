"""Check original checkpoints against the web API, without rewriting team results.

Run from any directory with the trained API dependencies installed:
    python web/scripts/verify_trained_integration.py --full-test --report .local/integration-verification.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "backend"))

from cnn_model import ECGCNN
from rnn_model import ECG_RNN
from transformer_model import ECGTransformer
from preprocessing import preprocess_data
from app import app

CHECKPOINTS = {
    "cnn": ROOT / "results/cnn/best_cnn_model.pt",
    "rnn": ROOT / "saved_models/best_rnn_model.pth",
    "transformer": ROOT / "results/transformer/best_model.pt",
}
RESULT_NAMES = {"cnn": "CNN", "rnn": "RNN_LSTM", "transformer": "Transformer"}


def original_model(model_id: str):
    if model_id == "cnn":
        model = ECGCNN(in_channels=1, num_classes=5, input_length=180, dropout=0.3)
    elif model_id == "rnn":
        model = ECG_RNN(input_size=1, hidden_size=32, num_layers=2, num_classes=5, dropout=0.5, model_type="lstm")
    else:
        model = ECGTransformer(input_length=180, classes=5, dimension=32, heads=4, layers=2, dropout=0.2, patch_size=4)
    state = torch.load(CHECKPOINTS[model_id], map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    return model.eval()


def model_input(x: np.ndarray, model_id: str):
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32).copy())
    return tensor.unsqueeze(-1) if model_id == "rnn" else tensor.unsqueeze(1)


def metrics(cm: np.ndarray) -> dict:
    support = cm.sum(axis=1)
    precision = np.divide(np.diag(cm), cm.sum(axis=0), out=np.zeros(5), where=cm.sum(axis=0) != 0)
    recall = np.divide(np.diag(cm), support, out=np.zeros(5), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=(precision + recall) != 0)
    return {"accuracy": float(np.trace(cm) / cm.sum()), "macro_f1": float(f1.mean()), "per_class_f1": f1.tolist(), "confusion_matrix": cm.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-test", action="store_true", help="Also replay all 18,098 test beats and compare confusion matrices.")
    parser.add_argument("--report", type=Path, help="Write a verification report outside the team's results directory.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    x = np.load(ROOT / "processed_data/split/X_test.npy", mmap_mode="r")
    y = np.load(ROOT / "processed_data/split/y_test.npy", mmap_mode="r")
    assert x.shape == (18098, 180) and y.shape == (18098,), "Unexpected team test split."
    selected = sorted(set(int(i) for c in range(5) for i in np.flatnonzero(y == c)[:3]) | set(np.random.default_rng(42).choice(len(y), 20, replace=False).tolist()))
    processed = preprocess_data(np.asarray(x[selected]))
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "device": "cpu", "torch_threads": 2,
        "api_samples_per_model": len(selected), "test_samples": len(y), "models": {},
    }
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert all(row["status"] == "ready" and not row["is_demo"] for row in health["models"]), health
        examples = client.get("/examples")
        assert examples.status_code == 200 and len(examples.json()["samples"]) == 5
        assert client.get("/examples/18098").status_code == 404
        assert client.get("/evaluation").status_code == 200
        saved_results = json.loads((ROOT / "results/evaluation/evaluation_results.json").read_text())
        for model_id in CHECKPOINTS:
            model = original_model(model_id)
            with torch.inference_mode():
                logits = model(model_input(processed, model_id)).numpy().astype(np.float64)
            exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
            expected = exponent / exponent.sum(axis=1, keepdims=True)
            maximum_error = 0.0
            for position, index in enumerate(selected):
                result = client.post("/predict", json={"model": model_id, "signal": x[index].tolist()})
                assert result.status_code == 200, result.text
                body = result.json()
                actual = np.array([row["probability"] for row in body["probabilities"]])
                assert not body["is_demo"] and body["backend"] == "native_pytorch"
                assert [row["code"] for row in body["probabilities"]] == ["N", "S", "V", "F", "Q"]
                assert np.allclose(body["preprocessing"]["signal"], processed[position], atol=1e-6)
                assert np.allclose(actual, expected[position], atol=1e-5, rtol=1e-5), (model_id, index, actual, expected[position])
                assert body["prediction"]["code"] == ["N", "S", "V", "F", "Q"][int(expected[position].argmax())]
                maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected[position]))))
            assert client.post("/predict", json={"model": model_id, "signal": x[0, :179].tolist()}).status_code == 422
            row = {"checkpoint_sha256": hashlib.sha256(CHECKPOINTS[model_id].read_bytes()).hexdigest(), "api_parity": "passed", "max_probability_error": maximum_error, "parameters": sum(p.numel() for p in model.parameters())}
            if args.full_test:
                predicted = []
                with torch.inference_mode():
                    for start in range(0, len(y), 256):
                        values = preprocess_data(np.asarray(x[start:start + 256]))
                        predicted.extend(model(model_input(values, model_id)).argmax(1).tolist())
                cm = np.bincount(5 * np.asarray(y) + np.asarray(predicted), minlength=25).reshape(5, 5)
                historical = saved_results[RESULT_NAMES[model_id]]["confusion_matrix"]
                assert np.array_equal(cm, historical), f"{model_id}: replayed predictions differ from saved confusion matrix."
                row.update(metrics(cm))
                row["saved_confusion_matrix_match"] = True
            report["models"][model_id] = row
            print(f"{model_id}: API parity PASS ({len(selected)} beats), full test {'PASS' if args.full_test else 'not requested'}", flush=True)
    report["status"] = "passed"
    if args.report:
        destination = args.report.resolve()
        if destination.is_relative_to(ROOT / "results"):
            raise ValueError("Use .local/ or another destination; do not overwrite team results.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Verification report: {destination}")


if __name__ == "__main__":
    main()
