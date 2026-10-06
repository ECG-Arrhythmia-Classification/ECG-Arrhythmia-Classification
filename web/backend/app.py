"""ECG checkpoint inference with the group's shared preprocessing pipeline.

Run: uvicorn app:app --host 127.0.0.1 --port 8000
Bundled native PyTorch models are the default; ECG_RUNTIME=demo is opt-in.
Checkpoint/data errors return HTTP 503 and never silently run a demo scorer.
"""
from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

TARGET_LENGTH = 256
TEAM_TARGET_LENGTH = 180
TEAM_SAMPLE_RATE = 360
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEAM_PREPROCESSING_PATH = PROJECT_ROOT / "preprocessing.py"
CLASS_ORDER = ("N", "S", "V", "F", "Q")
CLASSES = (
    {"code": "N", "label_vi": "Nhóm nhịp bình thường", "label_en": "Normal beat group"},
    {"code": "S", "label_vi": "Nhịp ngoại tâm thu trên thất", "label_en": "Supraventricular ectopic beat"},
    {"code": "V", "label_vi": "Nhịp ngoại tâm thu thất", "label_en": "Ventricular ectopic beat"},
    {"code": "F", "label_vi": "Nhịp hợp nhất", "label_en": "Fusion beat"},
    {"code": "Q", "label_vi": "Nhịp chưa phân loại", "label_en": "Unclassifiable / paced beat"},
)
MODEL_NAMES = {"cnn": "CNN", "rnn": "BiLSTM", "transformer": "Transformer"}
NATIVE_CHECKPOINTS = {
    "cnn": "results/cnn/best_cnn_model.pt",
    "rnn": "saved_models/best_rnn_model.pth",
    "transformer": "results/transformer/best_model.pt",
}
NATIVE_LAYOUTS = {"cnn": "channels_first", "rnn": "sequence", "transformer": "channels_first"}
RESULT_KEYS = {"cnn": "CNN", "rnn": "RNN_LSTM", "transformer": "Transformer"}
NOTICE = "For academic use. Results must not be used for medical diagnosis or clinical decisions."


class SignalError(ValueError):
    pass


class CheckpointError(RuntimeError):
    pass


class ArtifactError(RuntimeError):
    pass


class PreprocessingOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detrend: bool = True
    smooth: bool = True
    normalize: bool = True


class SignalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    signal: list[float] = Field(min_length=32, max_length=10000)
    preprocessing: PreprocessingOptions = Field(default_factory=PreprocessingOptions)
    model: Literal["cnn", "rnn", "transformer"] = "cnn"

    @field_validator("signal", mode="before")
    @classmethod
    def validate_signal(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("signal must be a JSON array of numbers.")
        if not 32 <= len(value) <= 10000:
            raise ValueError("signal must contain between 32 and 10,000 samples.")
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("Each ECG sample must be a number; strings, null, and booleans are not accepted.")
            try:
                finite = math.isfinite(item)
            except (OverflowError, TypeError):
                finite = False
            if not finite:
                raise ValueError("ECG samples must not contain NaN or Infinity.")
            if abs(item) > 1e9:
                raise ValueError("ECG values exceed the numeric limit of ±1e9.")
        data = np.asarray(value, dtype=np.float64)
        scale = max(1.0, float(np.max(np.abs(data))))
        if float(np.ptp(data)) <= 1e-12 * scale:
            raise ValueError("A flat signal does not contain enough information for classification.")
        return value


class PredictRequest(SignalRequest):
    pass


def statistics(signal: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(signal)),
        "maximum": float(np.max(signal)),
        "mean": float(np.mean(signal)),
        "standard_deviation": float(np.std(signal)),
    }


def preprocess(signal: list[float] | np.ndarray, options: PreprocessingOptions) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(signal, dtype=np.float64)
    data = raw.copy()
    steps: list[str] = []
    if options.detrend:
        axis = np.linspace(-1.0, 1.0, len(data))
        slope, intercept = np.linalg.lstsq(np.column_stack((axis, np.ones(len(data)))), data, rcond=None)[0]
        data = data - (slope * axis + intercept)
        steps.append("linear_detrend")
    if options.smooth:
        data = np.convolve(np.pad(data, (2, 2), mode="edge"), np.ones(5) / 5.0, mode="valid")
        steps.append("moving_average_5")
    data = np.interp(np.linspace(0.0, 1.0, TARGET_LENGTH), np.linspace(0.0, 1.0, len(data)), data)
    steps.append("resample_256")
    deviation = float(np.std(data))
    if not np.all(np.isfinite(data)) or deviation <= 1e-12 * max(1.0, float(np.max(np.abs(data)))):
        raise SignalError("The signal has no variation after preprocessing. Select a heartbeat with a clear waveform.")
    if options.normalize:
        data = (data - float(np.mean(data))) / deviation
        steps.append("z_score")
    return data.astype(np.float32), {
        "pipeline": "demo",
        "original_length": len(raw),
        "processed_length": TARGET_LENGTH,
        "steps": steps,
        "options": options.model_dump(),
        "statistics": {"before": statistics(raw), "after": statistics(data)},
    }


@lru_cache(maxsize=1)
def load_team_preprocessing() -> Any:
    """Import the group's actual pipeline by path, without a second implementation."""
    try:
        specification = importlib.util.spec_from_file_location("ecg_team_preprocessing", TEAM_PREPROCESSING_PATH)
        if specification is None or specification.loader is None:
            raise ImportError("Could not load the shared preprocessing.py module.")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        if module.HEARTBEAT_LENGTH != TEAM_TARGET_LENGTH or module.SAMPLE_RATE != TEAM_SAMPLE_RATE:
            raise ValueError("The shared input contract has changed. Update the adapter's input length and sample rate.")
        return module
    except Exception as exc:
        raise CheckpointError(f"Could not load the shared preprocessing.py pipeline: {type(exc).__name__}: {exc}") from exc


def preprocess_team(signal: list[float] | np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """A raw 180-sample heartbeat, using the exact pipeline used for training."""
    raw = np.asarray(signal, dtype=np.float64)
    if raw.ndim != 1 or len(raw) != TEAM_TARGET_LENGTH:
        raise SignalError("The shared pipeline requires exactly 180 raw samples from one heartbeat at 360 Hz; it does not resample the input.")
    module = load_team_preprocessing()
    try:
        # preprocess_data itself casts to float32 before filtering, just as in training.
        processed = np.asarray(module.preprocess_data(raw.reshape(1, TEAM_TARGET_LENGTH)), dtype=np.float32)
    except Exception as exc:
        raise CheckpointError(f"The shared preprocessing pipeline failed: {type(exc).__name__}: {exc}") from exc
    if processed.shape != (1, TEAM_TARGET_LENGTH) or not np.all(np.isfinite(processed)):
        raise CheckpointError("The shared pipeline must return a finite array with shape [1,180].")
    data = processed[0]
    if float(np.std(data)) <= 1e-8:
        raise SignalError("The signal has no variation after shared preprocessing. Select a heartbeat with a clear waveform.")
    return data, {
        "pipeline": "team",
        "source": "preprocessing.py",
        "original_length": len(raw),
        "processed_length": TEAM_TARGET_LENGTH,
        "sample_rate": TEAM_SAMPLE_RATE,
        "steps": ["bandpass_0.5_40_hz_order_4", "z_score"],
        "options": {"bandpass": True, "normalize": True},
        "options_fixed": True,
        "filter": {"low_cut": module.LOW_CUT, "high_cut": module.HIGH_CUT, "order": module.FILTER_ORDER},
        "statistics": {"before": statistics(raw), "after": statistics(data)},
    }


def demo_samples() -> dict[str, np.ndarray]:
    """Five synthetic, explicitly illustrative heartbeat shapes; no patient data."""
    t = np.linspace(0.0, 1.0, 360)
    def g(center: float, width: float, height: float) -> np.ndarray:
        return height * np.exp(-0.5 * ((t - center) / width) ** 2)
    normal = g(.21,.033,.14) + g(.395,.014,-.16) + g(.43,.012,1.05) + g(.465,.018,-.28) + g(.7,.065,.28)
    supraventricular = g(.26,.023,.07) + g(.37,.014,-.12) + g(.405,.011,.87) + g(.435,.016,-.2) + g(.64,.055,.16)
    ventricular = g(.18,.03,.05) + g(.43,.045,.85) + g(.515,.052,-.52) + g(.73,.08,-.23)
    fusion = g(.21,.03,.09) + g(.37,.016,-.08) + g(.43,.025,.96) + g(.49,.034,-.37) + g(.69,.067,.08)
    unknown = .23*np.sin(2*np.pi*t*4)*np.exp(-.5*((t-.49)/.23)**2) + g(.56,.02,.5) + g(.61,.03,-.31)
    baseline = .025*np.sin(2*np.pi*t)
    return dict(zip(CLASS_ORDER, (v+baseline for v in (normal, supraventricular, ventricular, fusion, unknown))))


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size != len(CLASS_ORDER) or not np.all(np.isfinite(values)):
        raise CheckpointError("The output must contain exactly 5 finite values in N, S, V, F, Q order.")
    values = np.exp(values - float(np.max(values)))
    return values / float(np.sum(values))


class PrototypeDemo:
    backend = "prototype_demo"
    is_demo = True

    def __init__(self, model_id: str):
        self.model_id = model_id

    def predict(self, signal: np.ndarray, options: PreprocessingOptions) -> np.ndarray:
        def normalized(values: np.ndarray) -> np.ndarray:
            return (values - np.mean(values)) / np.std(values)
        signal = normalized(signal)
        prototypes = [normalized(preprocess(sample, options)[0]) for sample in demo_samples().values()]
        distances: list[float] = []
        for prototype in prototypes:
            waveform = float(np.mean((signal - prototype) ** 2))
            derivative = float(np.mean((np.diff(signal) - np.diff(prototype)) ** 2))
            cumulative = float(np.mean((np.cumsum(signal - prototype) / 8.0) ** 2))
            if self.model_id == "cnn":
                distance = waveform + .7 * derivative
            elif self.model_id == "rnn":
                distance = .5 * waveform + .5 * cumulative
            else:
                index = np.arange(TARGET_LENGTH)
                distance = float(np.mean((signal-prototype)**2 * np.where((index > 80) & (index < 155), 2.0, .6)))
            distances.append(distance)
        return stable_softmax(-2.8 * np.asarray(distances))


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    checkpoint_path: str | None = None
    input_layout: Literal["channels_first", "sequence"] = "sequence"
    output_kind: Literal["logits", "probabilities"] = "logits"
    pipeline: Literal["demo", "team"] = "demo"
    # Preserve the explicit TorchScript integration contract for existing callers.
    # environment_configs selects native models and bundled paths by default.
    checkpoint_format: Literal["native", "torchscript"] = "torchscript"

    def input_length(self) -> int:
        return TEAM_TARGET_LENGTH if self.pipeline == "team" else TARGET_LENGTH

    def input_shape(self) -> list[int]:
        length = self.input_length()
        return [1, 1, length] if self.input_layout == "channels_first" else [1, length, 1]


class TorchScriptAdapter:
    backend = "torchscript"
    is_demo = False

    def __init__(self, config: ModelConfig):
        self.config = config
        checkpoint = Path(config.checkpoint_path or "")
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        if not checkpoint.is_file():
            raise CheckpointError(f"Checkpoint not found: {checkpoint.name}.")
        try:
            self.torch = importlib.import_module("torch")
            self.model = self.torch.jit.load(str(checkpoint), map_location="cpu")
            self.model.eval()
            # Probe the declared shape and output contract before advertising readiness.
            self.predict(np.zeros(config.input_length(), dtype=np.float32), PreprocessingOptions())
        except Exception as exc:
            raise CheckpointError(f"Could not load or run the TorchScript checkpoint: {type(exc).__name__}: {exc}") from exc

    def predict(self, signal: np.ndarray, options: PreprocessingOptions) -> np.ndarray:
        try:
            tensor = self.torch.from_numpy(np.asarray(signal, dtype=np.float32).copy()).reshape(self.config.input_shape())
            with self.torch.inference_mode():
                output = self.model(tensor)
            if not isinstance(output, self.torch.Tensor):
                raise CheckpointError("The checkpoint must return a tensor of logits or probabilities, not a tuple or dictionary.")
            if tuple(output.shape) not in ((1, len(CLASS_ORDER)), (len(CLASS_ORDER),)):
                raise CheckpointError(f"Output shape {tuple(output.shape)} must be [1,5] or [5].")
            values = output.detach().cpu().numpy().astype(np.float64).reshape(-1)
            if self.config.output_kind == "logits":
                return stable_softmax(values)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0) or not np.isclose(np.sum(values), 1.0, atol=1e-4):
                raise CheckpointError("Probabilities must be finite, within [0,1], and sum to 1.")
            return values / float(np.sum(values))
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"Checkpoint inference failed: {type(exc).__name__}: {exc}") from exc


@lru_cache(maxsize=3)
def load_team_architecture(model_id: str) -> Any:
    """Use the team's original architectures rather than a web-specific copy."""
    paths = {"cnn": "cnn_model.py", "rnn": "rnn_model.py", "transformer": "transformer_model.py"}
    specification = importlib.util.spec_from_file_location(f"ecg_team_{model_id}", PROJECT_ROOT / paths[model_id])
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load the {model_id} architecture.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")


class NativeCheckpointAdapter(TorchScriptAdapter):
    backend = "native_pytorch"

    def __init__(self, config: ModelConfig):
        self.config = config
        if config.pipeline != "team" or config.input_layout != NATIVE_LAYOUTS[config.model_id] or config.output_kind != "logits":
            raise CheckpointError("Native checkpoints require the team pipeline, logits, and an input layout matching the model architecture.")
        checkpoint = Path(config.checkpoint_path or "")
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        if not checkpoint.is_file():
            raise CheckpointError(f"Checkpoint not found: {checkpoint.name}.")
        try:
            if is_lfs_pointer(checkpoint):
                raise CheckpointError(f"{checkpoint.name} is a Git LFS pointer. Run git lfs pull to download the weights.")
            self.torch = importlib.import_module("torch")
            threads = int(os.getenv("ECG_TORCH_THREADS", "2"))
            if not 1 <= threads <= 32:
                raise CheckpointError("ECG_TORCH_THREADS must be an integer from 1 to 32.")
            self.torch.set_num_threads(threads)
            architecture = load_team_architecture(config.model_id)
            if config.model_id == "cnn":
                self.model = architecture.ECGCNN(in_channels=1, num_classes=5, input_length=180, dropout=0.3)
            elif config.model_id == "rnn":
                self.model = architecture.ECG_RNN(input_size=1, hidden_size=32, num_layers=2, num_classes=5, dropout=0.5, model_type="lstm")
            else:
                self.model = architecture.ECGTransformer(input_length=180, classes=5, dimension=32, heads=4, layers=2, dropout=0.2, patch_size=4)
            # The repository supplies trusted state_dict files. Restrict deserialization
            # to tensors/primitive containers instead of arbitrary pickle objects.
            checkpoint_data = self.torch.load(str(checkpoint), map_location="cpu", weights_only=True)
            state = checkpoint_data.get("model_state_dict", checkpoint_data) if isinstance(checkpoint_data, dict) else checkpoint_data
            self.model.load_state_dict(state, strict=True)
            self.model.eval()
            self.predict(np.zeros(TEAM_TARGET_LENGTH, dtype=np.float32), PreprocessingOptions())
            digest = hashlib.sha256()
            with checkpoint.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.checkpoint_metadata = {"name": checkpoint.name, "sha256": digest.hexdigest(), "format": "native", "architecture": type(self.model).__name__, "device": "cpu"}
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"Could not load or run the PyTorch checkpoint: {type(exc).__name__}: {exc}") from exc


class ModelRegistry:
    def __init__(self, configurations: dict[str, ModelConfig]):
        # A prototype scorer is always the demo pipeline, regardless of an unused
        # checkpoint pipeline override. Its descriptors must match its execution.
        self.configurations = {
            model_id: replace(config, pipeline="demo") if not config.checkpoint_path and config.checkpoint_format == "torchscript" and config.pipeline in ("demo", "team") else config
            for model_id, config in configurations.items()
        }
        self.models: dict[str, PrototypeDemo | TorchScriptAdapter] = {}
        self.errors: dict[str, str] = {}
        for model_id, config in self.configurations.items():
            if config.input_layout not in ("channels_first", "sequence") or config.output_kind not in ("logits", "probabilities") or config.pipeline not in ("demo", "team") or config.checkpoint_format not in ("native", "torchscript"):
                self.errors[model_id] = "Invalid input layout, output kind, pipeline, or checkpoint format."
                continue
            if not config.checkpoint_path:
                if config.checkpoint_format == "native":
                    self.errors[model_id] = "No native checkpoint is configured. Specify the path to the PyTorch weights."
                    continue
                self.models[model_id] = PrototypeDemo(model_id)
                continue
            try:
                adapter = NativeCheckpointAdapter if config.checkpoint_format == "native" else TorchScriptAdapter
                trained_model = adapter(config)
                if config.pipeline == "team":
                    load_team_preprocessing()
                self.models[model_id] = trained_model
            except CheckpointError as exc:
                self.errors[model_id] = str(exc)

    def describe(self) -> list[dict[str, Any]]:
        rows = []
        for model_id, config in self.configurations.items():
            model = self.models.get(model_id)
            rows.append({
                "id": model_id,
                "name": MODEL_NAMES[model_id],
                "backend": model.backend if model else "native_pytorch" if config.checkpoint_format == "native" else "torchscript",
                "checkpoint_format": config.checkpoint_format if config.checkpoint_path else None,
                "checkpoint_name": Path(config.checkpoint_path).name if config.checkpoint_path else None,
                "is_demo": model.is_demo if model else False,
                "status": "ready" if model and model_id not in self.errors else "unavailable",
                "input_shape": config.input_shape(),
                "input_length": config.input_length(),
                "pipeline": config.pipeline,
                "sample_rate": TEAM_SAMPLE_RATE if config.pipeline == "team" else None,
                "class_order": list(CLASS_ORDER),
                "error": self.errors.get(model_id),
            })
        return rows

    def ensure_available(self, model_id: str) -> PrototypeDemo | TorchScriptAdapter:
        if model_id in self.errors or model_id not in self.models:
            raise CheckpointError(self.errors.get(model_id, "The model is unavailable."))
        return self.models[model_id]

    def preprocess(self, model_id: str, signal: list[float], options: PreprocessingOptions) -> tuple[np.ndarray, dict[str, Any]]:
        if self.configurations[model_id].pipeline not in ("demo", "team"):
            raise CheckpointError(self.errors.get(model_id, "Invalid preprocessing pipeline."))
        if self.configurations[model_id].pipeline == "team":
            return preprocess_team(signal)
        return preprocess(signal, options)

    def predict(self, model_id: str, signal: np.ndarray, options: PreprocessingOptions) -> tuple[np.ndarray, PrototypeDemo | TorchScriptAdapter]:
        model = self.ensure_available(model_id)
        try:
            return model.predict(signal, options), model
        except CheckpointError as exc:
            # Preserve the failed trained-model state. Never replace it with a demo scorer.
            self.errors[model_id] = str(exc)
            raise


def environment_configs() -> dict[str, ModelConfig]:
    runtime = os.getenv("ECG_RUNTIME", "trained").strip().lower()
    if runtime not in ("trained", "demo"):
        raise ValueError("ECG_RUNTIME must be trained or demo.")
    result = {}
    for model_id in MODEL_NAMES:
        prefix = f"ECG_{model_id.upper()}"
        checkpoint = os.getenv(f"{prefix}_CHECKPOINT") or (NATIVE_CHECKPOINTS[model_id] if runtime == "trained" else None)
        result[model_id] = ModelConfig(
            model_id=model_id,
            checkpoint_path=checkpoint,
            input_layout=os.getenv(f"{prefix}_INPUT_LAYOUT", NATIVE_LAYOUTS[model_id]),
            output_kind=os.getenv(f"{prefix}_OUTPUT_KIND", "logits"),
            pipeline=os.getenv(f"{prefix}_PIPELINE", "team" if checkpoint else "demo"),
            checkpoint_format=os.getenv(f"{prefix}_CHECKPOINT_FORMAT", "native" if runtime == "trained" else "torchscript"),
        )
    return result


@lru_cache(maxsize=1)
def load_test_dataset(repository: str) -> tuple[np.ndarray, np.ndarray]:
    """Read raw held-out heartbeats without copying the full dataset into memory."""
    directory = Path(repository) / "processed_data" / "split"
    paths = (directory / "X_test.npy", directory / "y_test.npy")
    try:
        for path in paths:
            if not path.is_file():
                raise ArtifactError(f"Missing {path.name}. Run git lfs pull to download the test data.")
            if is_lfs_pointer(path):
                raise ArtifactError(f"{path.name} is a Git LFS pointer. Run git lfs pull, then restart the API.")
        signals, labels = (np.load(path, mmap_mode="r", allow_pickle=False) for path in paths)
        if signals.ndim != 2 or signals.shape[1] != TEAM_TARGET_LENGTH or labels.shape != (len(signals),):
            raise ArtifactError("Test data requires X_test [N,180] and y_test [N] with matching heartbeat counts.")
        if len(signals) == 0 or signals.dtype.kind not in "fiu" or not np.issubdtype(labels.dtype, np.integer):
            raise ArtifactError("The test set is empty or its signal or label data type is invalid.")
        if not np.all(np.isfinite(signals)) or np.any(labels < 0) or np.any(labels >= len(CLASS_ORDER)):
            raise ArtifactError("The test data contains non-finite samples or labels outside N, S, V, F, Q.")
        return signals, labels
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"Could not read the test data: {type(exc).__name__}: {exc}") from exc


@lru_cache(maxsize=1)
def load_evaluation(repository: str) -> dict[str, Any]:
    """Expose the saved team experiments; this endpoint does not rerun a benchmark."""
    root = Path(repository) / "results"
    try:
        def read_result(relative: str) -> Any:
            return json.loads((root / relative).read_text(encoding="utf-8"))

        metrics = read_result("evaluation/evaluation_results.json")
        benchmark = read_result("benchmark/benchmark_results.json")
        robustness = read_result("robustness/robustness_results.json")
        rows = []
        sample_counts = []
        def metric(value: Any) -> float:
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ArtifactError("Evaluation metrics must be finite numbers within [0,1].")
            return float(value)

        for model_id, key in RESULT_KEYS.items():
            evaluation, timing = metrics[key], benchmark[key]
            confusion = np.asarray(evaluation["confusion_matrix"])
            if confusion.shape != (5, 5) or not np.issubdtype(confusion.dtype, np.integer) or np.any(confusion < 0):
                raise ArtifactError("The confusion matrix must contain 5×5 non-negative integer counts.")
            sample_counts.append(int(confusion.sum()))
            parameter_count = timing["trainable_parameters"]
            latency = timing["avg_inference_time_per_sample_ms"]
            if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count <= 0:
                raise ArtifactError("Invalid model parameter count.")
            if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
                raise ArtifactError("Invalid saved inference time.")
            noisy_rows = []
            for level, snr in (("low_30db", 30), ("medium_20db", 20), ("high_10db", 10)):
                noisy = robustness[key][level]
                if noisy["snr_db"] != snr:
                    raise ArtifactError("Robustness results must use SNR levels of 30, 20, and 10 dB.")
                noisy_rows.append({"snr_db": snr, "accuracy": metric(noisy["accuracy"]), "macro_f1": metric(noisy["macro_f1"])})
            rows.append({
                "id": model_id, "name": MODEL_NAMES[model_id],
                "accuracy": metric(evaluation["accuracy"]), "macro_f1": metric(evaluation["f1_macro"]),
                "precision": metric(evaluation["precision_macro"]), "recall": metric(evaluation["recall_macro"]),
                "weighted_f1": metric(evaluation["f1_weighted"]), "parameters": parameter_count,
                "inference_ms_per_sample": float(latency), "robustness": noisy_rows,
                "class_order": list(CLASS_ORDER), "confusion_matrix": confusion.tolist(),
            })
        if len(set(sample_counts)) != 1 or sample_counts[0] <= 0:
            raise ArtifactError("The three evaluation results have different test-sample counts.")
        return {
            "models": rows, "test_samples": sample_counts[0], "source": "saved_results",
            "notice": "Saved test and benchmark results; these are not new measurements on the current machine. Per-sample times are averaged over batch inference and differ from API request latency.",
        }
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError(f"Could not read the saved evaluation results: {type(exc).__name__}: {exc}") from exc


def create_app(configurations: dict[str, ModelConfig] | None = None) -> FastAPI:
    configs = environment_configs() if configurations is None else configurations
    if set(configs) != set(MODEL_NAMES):
        raise ValueError("The registry must define cnn, rnn, and transformer.")
    registry = ModelRegistry(configs)
    application = FastAPI(title="ECG Sequence Lab API", version="2.0.0", description=NOTICE)
    origins = [origin.strip() for origin in os.getenv("ECG_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
    application.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    application.state.registry = registry

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Any, error: RequestValidationError) -> JSONResponse:
        # Do not echo waveform values (especially NaN/Infinity) into JSON errors.
        details = [{"type": item["type"], "loc": list(item["loc"]), "msg": item["msg"]} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @application.get("/health")
    def health() -> dict[str, Any]:
        rows = registry.describe()
        ready = [row for row in rows if row["status"] == "ready"]
        modes = set(row["backend"] for row in ready)
        mode = "mixed" if len(modes) > 1 else "demo" if "prototype_demo" in modes else next(iter(modes)) if modes else "unavailable"
        lengths = {row["input_length"] for row in rows}
        return {"status": "ok" if len(ready) == len(rows) else "degraded", "mode": mode, "input_length": next(iter(lengths)) if len(lengths) == 1 else TARGET_LENGTH, "team_input_length": TEAM_TARGET_LENGTH, "models": rows, "unavailable_models": [row["id"] for row in rows if row["status"] != "ready"]}

    @application.get("/models")
    def models() -> dict[str, Any]:
        rows = registry.describe()
        lengths = {row["input_length"] for row in rows}
        return {"classes": list(CLASSES), "input_length": next(iter(lengths)) if len(lengths) == 1 else TARGET_LENGTH, "team_input_length": TEAM_TARGET_LENGTH, "models": rows, "notice": NOTICE}

    @application.get("/examples")
    def examples() -> dict[str, Any]:
        try:
            signals, labels = load_test_dataset(str(PROJECT_ROOT))
            samples = []
            for label_index, row in enumerate(CLASSES):
                indices = np.flatnonzero(labels == label_index)
                if len(indices):
                    samples.append({"index": int(indices[0]), "code": row["code"], "label_vi": row["label_vi"], "label_en": row["label_en"], "sample_rate": TEAM_SAMPLE_RATE, "length": TEAM_TARGET_LENGTH})
            return {"samples": samples, "total": len(signals), "source": "MIT-BIH test", "is_synthetic": False}
        except ArtifactError as exc:
            raise HTTPException(status_code=503, detail={"code": "dataset_unavailable", "message": str(exc)}) from exc

    @application.get("/examples/{index}")
    def example(index: int) -> dict[str, Any]:
        try:
            signals, labels = load_test_dataset(str(PROJECT_ROOT))
        except ArtifactError as exc:
            raise HTTPException(status_code=503, detail={"code": "dataset_unavailable", "message": str(exc)}) from exc
        if not 0 <= index < len(signals):
            raise HTTPException(status_code=404, detail={"code": "example_not_found", "message": "The index is outside the test set."})
        expected = CLASSES[int(labels[index])]
        return {"signal": signals[index].tolist(), "index": index, "expected_class": {"code": expected["code"], "label_vi": expected["label_vi"], "label_en": expected["label_en"]}, "sample_rate": TEAM_SAMPLE_RATE, "source": "MIT-BIH test", "is_synthetic": False}

    @application.get("/evaluation")
    def evaluation() -> dict[str, Any]:
        try:
            return load_evaluation(str(PROJECT_ROOT))
        except ArtifactError as exc:
            raise HTTPException(status_code=503, detail={"code": "evaluation_unavailable", "message": str(exc)}) from exc

    @application.post("/preprocess")
    def preprocess_endpoint(request: SignalRequest) -> dict[str, Any]:
        try:
            processed, metadata = registry.preprocess(request.model, request.signal, request.preprocessing)
        except SignalError as exc:
            raise HTTPException(status_code=422, detail={"code": "invalid_signal", "message": str(exc)}) from exc
        except CheckpointError as exc:
            raise HTTPException(status_code=503, detail={"code": "checkpoint_unavailable", "model_id": request.model, "message": str(exc)}) from exc
        return {"signal": processed.tolist(), **metadata}

    @application.post("/predict")
    def predict_endpoint(request: PredictRequest) -> dict[str, Any]:
        try:
            # Report an unavailable configured checkpoint before applying any
            # different-length preprocessing; never silently execute a demo.
            registry.ensure_available(request.model)
            processed, metadata = registry.preprocess(request.model, request.signal, request.preprocessing)
        except SignalError as exc:
            raise HTTPException(status_code=422, detail={"code": "invalid_signal", "message": str(exc)}) from exc
        except CheckpointError as exc:
            raise HTTPException(status_code=503, detail={"code": "checkpoint_unavailable", "model_id": request.model, "message": str(exc)}) from exc
        started = time.perf_counter()
        try:
            probabilities, model = registry.predict(request.model, processed, request.preprocessing)
        except CheckpointError as exc:
            raise HTTPException(status_code=503, detail={"code": "checkpoint_unavailable", "model_id": request.model, "message": str(exc)}) from exc
        inference_ms = (time.perf_counter() - started) * 1000.0
        winner = int(np.argmax(probabilities))
        return {
            "model_id": request.model,
            "model_name": MODEL_NAMES[request.model],
            "backend": model.backend,
            "is_demo": model.is_demo,
            "checkpoint": getattr(model, "checkpoint_metadata", None),
            "prediction": dict(CLASSES[winner]),
            "confidence": float(probabilities[winner]),
            "probabilities": [{"code": row["code"], "label_vi": row["label_vi"], "label_en": row["label_en"], "probability": float(probabilities[index])} for index, row in enumerate(CLASSES)],
            "inference_ms": round(inference_ms, 4),
            "preprocessing": {**metadata, "signal": processed.tolist()},
            "notice": NOTICE,
            "probability_note": "Prototype similarity score; not calibrated using test data." if model.is_demo else "Checkpoint softmax score; calibration must be assessed on the test set.",
        }

    return application


app = create_app()
