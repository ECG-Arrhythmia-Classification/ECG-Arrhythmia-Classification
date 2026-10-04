"""ECG demo / shared team preprocessing / TorchScript inference API.

Run: uvicorn app:app --host 127.0.0.1 --port 8000
Demo scorers are deterministic prototype distances, NOT trained deep models.
TorchScript is optional: explicit checkpoint errors always return HTTP 503.
"""
from __future__ import annotations

import importlib
import importlib.util
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
    {"code": "N", "label_vi": "Nhóm nhịp bình thường", "label_en": "Normal group"},
    {"code": "S", "label_vi": "Nhịp ngoại tâm thu trên thất", "label_en": "Supraventricular ectopic"},
    {"code": "V", "label_vi": "Nhịp ngoại tâm thu thất", "label_en": "Ventricular ectopic"},
    {"code": "F", "label_vi": "Nhịp hợp nhất", "label_en": "Fusion beat"},
    {"code": "Q", "label_vi": "Nhịp chưa phân loại", "label_en": "Unclassified beat"},
)
MODEL_NAMES = {"cnn": "CNN", "rnn": "LSTM / GRU", "transformer": "Transformer"}
NOTICE = "Demo học thuật; không dùng kết quả để chẩn đoán hoặc đưa ra quyết định y tế."


class SignalError(ValueError):
    pass


class CheckpointError(RuntimeError):
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
            raise ValueError("signal phải là một mảng số JSON.")
        if not 32 <= len(value) <= 10000:
            raise ValueError("signal phải chứa từ 32 đến 10000 mẫu.")
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("Mỗi mẫu ECG phải là số; không nhận chuỗi, null hoặc boolean.")
            try:
                finite = math.isfinite(item)
            except (OverflowError, TypeError):
                finite = False
            if not finite:
                raise ValueError("ECG không được chứa NaN hoặc Infinity.")
            if abs(item) > 1e9:
                raise ValueError("Giá trị ECG vượt giới hạn số học ±1e9.")
        data = np.asarray(value, dtype=np.float64)
        scale = max(1.0, float(np.max(np.abs(data))))
        if float(np.ptp(data)) <= 1e-12 * scale:
            raise ValueError("Tín hiệu phẳng không đủ thông tin để phân loại.")
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
        raise SignalError("Tín hiệu mất biến thiên sau tiền xử lý. Chọn một đoạn ECG có nhịp rõ ràng.")
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
            raise ImportError("Không nạp được preprocessing.py của nhóm.")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        if module.HEARTBEAT_LENGTH != TEAM_TARGET_LENGTH or module.SAMPLE_RATE != TEAM_SAMPLE_RATE:
            raise ValueError("Contract nhóm đã đổi; cần cập nhật input length/sample rate của adapter.")
        return module
    except Exception as exc:
        raise CheckpointError(f"Không tải được pipeline nhóm preprocessing.py: {type(exc).__name__}: {exc}") from exc


def preprocess_team(signal: list[float] | np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """A raw 180-sample heartbeat, using the exact pipeline used for training."""
    raw = np.asarray(signal, dtype=np.float64)
    if raw.ndim != 1 or len(raw) != TEAM_TARGET_LENGTH:
        raise SignalError("Pipeline nhóm cần đúng 180 mẫu thô của một heartbeat ở 360 Hz; không tự resample.")
    module = load_team_preprocessing()
    try:
        # preprocess_data itself casts to float32 before filtering, just as in training.
        processed = np.asarray(module.preprocess_data(raw.reshape(1, TEAM_TARGET_LENGTH)), dtype=np.float32)
    except Exception as exc:
        raise CheckpointError(f"Pipeline nhóm không chạy được: {type(exc).__name__}: {exc}") from exc
    if processed.shape != (1, TEAM_TARGET_LENGTH) or not np.all(np.isfinite(processed)):
        raise CheckpointError("Pipeline nhóm phải trả mảng hữu hạn có shape [1,180].")
    data = processed[0]
    if float(np.std(data)) <= 1e-8:
        raise SignalError("Tín hiệu mất biến thiên sau pipeline nhóm; cần chọn một heartbeat có nhịp rõ ràng.")
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
        raise CheckpointError("Output phải có đúng 5 giá trị hữu hạn theo thứ tự N, S, V, F, Q.")
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
            raise CheckpointError(f"Không tìm thấy checkpoint: {checkpoint.name}.")
        try:
            self.torch = importlib.import_module("torch")
            self.model = self.torch.jit.load(str(checkpoint), map_location="cpu")
            self.model.eval()
            # Probe the declared shape and output contract before advertising readiness.
            self.predict(np.zeros(config.input_length(), dtype=np.float32), PreprocessingOptions())
        except Exception as exc:
            raise CheckpointError(f"Không tải/chạy được TorchScript checkpoint: {type(exc).__name__}: {exc}") from exc

    def predict(self, signal: np.ndarray, options: PreprocessingOptions) -> np.ndarray:
        try:
            tensor = self.torch.from_numpy(np.asarray(signal, dtype=np.float32).copy()).reshape(self.config.input_shape())
            with self.torch.inference_mode():
                output = self.model(tensor)
            if not isinstance(output, self.torch.Tensor):
                raise CheckpointError("Checkpoint phải trả về Tensor logits hoặc probability; không nhận tuple/dict.")
            if tuple(output.shape) not in ((1, len(CLASS_ORDER)), (len(CLASS_ORDER),)):
                raise CheckpointError(f"Output shape {tuple(output.shape)} không đúng [1,5] hoặc [5].")
            values = output.detach().cpu().numpy().astype(np.float64).reshape(-1)
            if self.config.output_kind == "logits":
                return stable_softmax(values)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0) or not np.isclose(np.sum(values), 1.0, atol=1e-4):
                raise CheckpointError("Probability phải nằm trong [0,1], hữu hạn và có tổng bằng 1.")
            return values / float(np.sum(values))
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"Inference checkpoint thất bại: {type(exc).__name__}: {exc}") from exc


class ModelRegistry:
    def __init__(self, configurations: dict[str, ModelConfig]):
        # A prototype scorer is always the demo pipeline, regardless of an unused
        # checkpoint pipeline override. Its descriptors must match its execution.
        self.configurations = {
            model_id: replace(config, pipeline="demo") if not config.checkpoint_path and config.pipeline in ("demo", "team") else config
            for model_id, config in configurations.items()
        }
        self.models: dict[str, PrototypeDemo | TorchScriptAdapter] = {}
        self.errors: dict[str, str] = {}
        for model_id, config in self.configurations.items():
            if config.input_layout not in ("channels_first", "sequence") or config.output_kind not in ("logits", "probabilities") or config.pipeline not in ("demo", "team"):
                self.errors[model_id] = "Cấu hình input layout, output kind hoặc pipeline không hợp lệ."
                continue
            if not config.checkpoint_path:
                self.models[model_id] = PrototypeDemo(model_id)
                continue
            try:
                trained_model = TorchScriptAdapter(config)
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
                "backend": model.backend if model else "torchscript",
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
            raise CheckpointError(self.errors.get(model_id, "Model chưa sẵn sàng."))
        return self.models[model_id]

    def preprocess(self, model_id: str, signal: list[float], options: PreprocessingOptions) -> tuple[np.ndarray, dict[str, Any]]:
        if self.configurations[model_id].pipeline not in ("demo", "team"):
            raise CheckpointError(self.errors.get(model_id, "Pipeline không hợp lệ."))
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
    result = {}
    for model_id in MODEL_NAMES:
        prefix = f"ECG_{model_id.upper()}"
        checkpoint = os.getenv(f"{prefix}_CHECKPOINT") or None
        result[model_id] = ModelConfig(
            model_id=model_id,
            checkpoint_path=checkpoint,
            input_layout=os.getenv(f"{prefix}_INPUT_LAYOUT", "channels_first" if model_id == "cnn" else "sequence"),
            output_kind=os.getenv(f"{prefix}_OUTPUT_KIND", "logits"),
            pipeline=os.getenv(f"{prefix}_PIPELINE", "team" if checkpoint else "demo"),
        )
    return result


def create_app(configurations: dict[str, ModelConfig] | None = None) -> FastAPI:
    configs = environment_configs() if configurations is None else configurations
    if set(configs) != set(MODEL_NAMES):
        raise ValueError("Registry phải khai báo đủ cnn, rnn và transformer.")
    registry = ModelRegistry(configs)
    application = FastAPI(title="ECG Sequence Lab API", version="1.0.0", description=NOTICE)
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
        mode = "mixed" if len(modes) > 1 else "demo" if "prototype_demo" in modes else "torchscript" if modes else "unavailable"
        return {"status": "ok", "mode": mode, "input_length": TARGET_LENGTH, "team_input_length": TEAM_TARGET_LENGTH, "models": rows, "unavailable_models": [row["id"] for row in rows if row["status"] != "ready"]}

    @application.get("/models")
    def models() -> dict[str, Any]:
        return {"classes": list(CLASSES), "input_length": TARGET_LENGTH, "team_input_length": TEAM_TARGET_LENGTH, "models": registry.describe(), "notice": NOTICE}

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
            "prediction": dict(CLASSES[winner]),
            "confidence": float(probabilities[winner]),
            "probabilities": [{"code": row["code"], "label_vi": row["label_vi"], "probability": float(probabilities[index])} for index, row in enumerate(CLASSES)],
            "inference_ms": round(inference_ms, 4),
            "preprocessing": {**metadata, "signal": processed.tolist()},
            "notice": NOTICE,
            "probability_note": "Điểm tương đồng prototype, chưa hiệu chuẩn bằng dữ liệu test." if model.is_demo else "Probability của checkpoint; độ hiệu chuẩn cần được kiểm chứng trên test set.",
        }

    return application


app = create_app()
