"""API, shared pipeline and mocked adapter checks; python -m unittest test_api -v."""
from __future__ import annotations

import math
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app import CLASS_ORDER, CheckpointError, ModelConfig, PreprocessingOptions, PrototypeDemo, TorchScriptAdapter, create_app, demo_samples, environment_configs, load_team_preprocessing


def demo_configs() -> dict[str, ModelConfig]:
    return {key: ModelConfig(key, input_layout="channels_first" if key == "cnn" else "sequence") for key in ("cnn", "rnn", "transformer")}


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(demo_configs()))
        self.normal = demo_samples()["N"].tolist()

    def test_registry_is_explicit_demo(self) -> None:
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual([row["code"] for row in response.json()["classes"]], list(CLASS_ORDER))
        self.assertTrue(all(row["is_demo"] and row["status"] == "ready" for row in response.json()["models"]))
        self.assertEqual(self.client.get("/health").json()["mode"], "demo")

    def test_preprocess_restores_fixed_length_and_standardization(self) -> None:
        signal = np.interp(np.linspace(0, 1, 401), np.linspace(0, 1, len(self.normal)), self.normal) + np.linspace(0.2, 0.8, 401)
        response = self.client.post("/preprocess", json={"signal": signal.tolist()})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["original_length"], 401)
        self.assertEqual(len(body["signal"]), 256)
        self.assertTrue(all(math.isfinite(value) for value in body["signal"]))
        self.assertAlmostEqual(float(np.mean(body["signal"])), 0.0, places=6)
        self.assertAlmostEqual(float(np.std(body["signal"])), 1.0, places=6)
        self.assertEqual(body["steps"], ["linear_detrend", "moving_average_5", "resample_256", "z_score"])

    def test_rejects_invalid_and_flat_signals(self) -> None:
        invalid = ([1.0] * 31, [1.0] * 10001, [2.0] * 256, [True] + self.normal[1:], ["0.4"] + self.normal[1:], [None] + self.normal[1:], [1e10] + self.normal[1:])
        for signal in invalid:
            with self.subTest(signal_type=type(signal[0]).__name__, length=len(signal)):
                response = self.client.post("/predict", json={"signal": signal, "model": "cnn"})
                self.assertEqual(response.status_code, 422)
        for bad_value in ("NaN", "Infinity", "-Infinity"):
            payload = '{"signal":[' + bad_value + ',' + ','.join(map(str, self.normal[1:])) + '],"model":"cnn"}'
            response = self.client.post("/predict", content=payload, headers={"Content-Type": "application/json"})
            self.assertEqual(response.status_code, 422)

    def test_rejects_signal_that_detrends_to_flat(self) -> None:
        response = self.client.post("/preprocess", json={"signal": np.linspace(0, 1, 256).tolist()})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], "invalid_signal")

    def test_probabilities_and_prediction_depend_on_input(self) -> None:
        for model in ("cnn", "rnn", "transformer"):
            for code, signal in demo_samples().items():
                with self.subTest(model=model, code=code):
                    response = self.client.post("/predict", json={"signal": signal.tolist(), "model": model})
                    self.assertEqual(response.status_code, 200)
                    body = response.json()
                    self.assertEqual(body["prediction"]["code"], code)
                    probabilities = [entry["probability"] for entry in body["probabilities"]]
                    self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
                    self.assertTrue(all(0 <= probability <= 1 for probability in probabilities))
                    self.assertEqual(body["confidence"], max(probabilities))
                    repeated = self.client.post("/predict", json={"signal": signal.tolist(), "model": model}).json()
                    self.assertEqual(body["probabilities"], repeated["probabilities"])

    def test_three_modes_are_not_identical(self) -> None:
        outputs = [self.client.post("/predict", json={"signal": self.normal, "model": model}).json()["probabilities"] for model in ("cnn", "rnn", "transformer")]
        self.assertNotEqual(outputs[0], outputs[1])
        self.assertNotEqual(outputs[1], outputs[2])

    def test_browser_and_api_demo_agree(self) -> None:
        fixtures = json.loads((Path(__file__).resolve().parent.parent / "tests" / "parity-fixture.json").read_text(encoding="utf-8"))
        for fixture in fixtures:
            with self.subTest(code=fixture["code"], model=fixture["model"]):
                body = self.client.post("/predict", json={"signal": fixture["signal"], "model": fixture["model"]}).json()
                self.assertEqual(body["prediction"]["code"], fixture["code"])
                self.assertEqual(len(body["preprocessing"]["signal"]), 256)
                actual = [row["probability"] for row in body["probabilities"]]
                np.testing.assert_allclose(actual, fixture["probabilities"], atol=1e-6, rtol=1e-5)

    def test_invalid_model_and_unknown_options(self) -> None:
        self.assertEqual(self.client.post("/predict", json={"signal": self.normal, "model": "anything"}).status_code, 422)
        self.assertEqual(self.client.post("/preprocess", json={"signal": self.normal, "model": "anything"}).status_code, 422)
        self.assertEqual(self.client.post("/preprocess", json={"signal": self.normal, "preprocessing": {"invented": True}}).status_code, 422)

    def test_missing_configured_checkpoint_never_becomes_demo(self) -> None:
        configs = demo_configs()
        configs["cnn"] = ModelConfig("cnn", str(Path(tempfile.gettempdir()) / "definitely_missing_ecg_checkpoint.pt"), "channels_first")
        client = TestClient(create_app(configs))
        descriptor = client.get("/models").json()["models"][0]
        self.assertFalse(descriptor["is_demo"])
        self.assertEqual(descriptor["status"], "unavailable")
        response = client.post("/predict", json={"signal": self.normal, "model": "cnn"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "checkpoint_unavailable")
        # Other models can remain usable even when one checkpoint is unavailable.
        self.assertEqual(client.post("/predict", json={"signal": self.normal, "model": "rnn"}).status_code, 200)

    def test_corrupt_configured_checkpoint_never_becomes_demo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "broken.pt"
            checkpoint.write_text("not a torchscript checkpoint", encoding="utf-8")
            configs = demo_configs()
            configs["rnn"] = ModelConfig("rnn", str(checkpoint))
            # Explicitly simulate a missing optional torch dependency in a portable test.
            with patch("app.importlib.import_module", side_effect=ModuleNotFoundError("torch missing")):
                client = TestClient(create_app(configs))
            response = client.post("/predict", json={"signal": self.normal, "model": "rnn"})
            self.assertEqual(response.status_code, 503)
            self.assertIn("torch missing", response.json()["detail"]["message"])

    def test_runtime_checkpoint_failure_is_visible_and_persistent(self) -> None:
        app = create_app(demo_configs())
        client = TestClient(app)
        class BrokenTrainedModel:
            backend = "torchscript"
            is_demo = False
            def predict(self, signal, options):
                raise CheckpointError("trained checkpoint failed during inference")
        app.state.registry.models["cnn"] = BrokenTrainedModel()
        for _ in range(2):
            response = client.post("/predict", json={"signal": self.normal, "model": "cnn"})
            self.assertEqual(response.status_code, 503)
        descriptor = client.get("/models").json()["models"][0]
        self.assertFalse(descriptor["is_demo"])
        self.assertEqual(descriptor["status"], "unavailable")


class AdapterContractTests(unittest.TestCase):
    def test_adapter_declares_cnn_and_sequence_shapes(self) -> None:
        observed_shapes = []
        class Tensor:
            def __init__(self, array):
                self.array = np.asarray(array)
            @property
            def shape(self):
                return self.array.shape
            def reshape(self, shape):
                return Tensor(self.array.reshape(shape))
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.array
        class Module:
            def eval(self):
                return self
            def __call__(self, tensor):
                observed_shapes.append(tuple(tensor.shape))
                return Tensor([[0, 1, 2, 3, 4]])
        fake_torch = SimpleNamespace(Tensor=Tensor, from_numpy=Tensor, inference_mode=nullcontext, jit=SimpleNamespace(load=lambda path, map_location: Module()))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "contract.pt"
            checkpoint.write_bytes(b"mock adapter contract only")
            for pipeline, length in (("demo", 256), ("team", 180)):
                for layout, shape in (("channels_first", (1, 1, length)), ("sequence", (1, length, 1))):
                    with self.subTest(layout=layout, pipeline=pipeline), patch("app.importlib.import_module", return_value=fake_torch):
                        adapter = TorchScriptAdapter(ModelConfig("cnn", str(checkpoint), layout, pipeline=pipeline))
                        probabilities = adapter.predict(np.ones(length, dtype=np.float32), PreprocessingOptions())
                        self.assertEqual(observed_shapes[-1], shape)
                        self.assertEqual(int(np.argmax(probabilities)), 4)
                        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=12)

    def test_relative_checkpoint_is_resolved_from_repository_root(self) -> None:
        observed_paths = []
        fake_torch = SimpleNamespace(jit=SimpleNamespace(load=lambda path, map_location: observed_paths.append(path) or SimpleNamespace(eval=lambda: None)))
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            (repository / "models").mkdir()
            checkpoint = repository / "models" / "best.ts.pt"
            checkpoint.write_bytes(b"mock adapter path only")
            with patch("app.PROJECT_ROOT", repository), patch("app.importlib.import_module", return_value=fake_torch), patch.object(TorchScriptAdapter, "predict", return_value=np.full(5, .2)):
                TorchScriptAdapter(ModelConfig("cnn", "models/best.ts.pt", pipeline="team"))
                TorchScriptAdapter(ModelConfig("cnn", str(checkpoint), pipeline="team"))
            self.assertEqual(observed_paths, [str(checkpoint), str(checkpoint)])


class SharedPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        normal = demo_samples()["N"]
        self.raw = np.interp(np.linspace(0, 1, 180), np.linspace(0, 1, len(normal)), normal)
        self.configs = demo_configs()
        self.configs["cnn"] = ModelConfig("cnn", "models/definitely_missing_team_checkpoint.ts.pt", "channels_first", pipeline="team")

    def test_team_preview_matches_actual_training_pipeline_exactly(self) -> None:
        expected = load_team_preprocessing().preprocess_data(self.raw.reshape(1, 180))[0]
        response = TestClient(create_app(self.configs)).post("/preprocess", json={
            "signal": self.raw.tolist(), "model": "cnn",
            "preprocessing": {"detrend": False, "smooth": False, "normalize": False},
        })
        self.assertEqual(response.status_code, 200)
        body = response.json()
        np.testing.assert_array_equal(np.asarray(body["signal"], dtype=np.float32), expected)
        self.assertEqual(body["pipeline"], "team")
        self.assertEqual(body["processed_length"], 180)
        self.assertEqual(body["sample_rate"], 360)
        self.assertTrue(body["options_fixed"])
        self.assertEqual(body["options"], {"bandpass": True, "normalize": True})
        self.assertEqual(body["steps"], ["bandpass_0.5_40_hz_order_4", "z_score"])

    def test_team_pipeline_rejects_wrong_length_without_resampling(self) -> None:
        client = TestClient(create_app(self.configs))
        for length in (179, 181, 256, 360):
            signal = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, 180), self.raw)
            with self.subTest(length=length):
                response = client.post("/preprocess", json={"signal": signal.tolist(), "model": "cnn"})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["detail"]["code"], "invalid_signal")
                self.assertIn("180", response.json()["detail"]["message"])

    def test_team_prediction_receives_shared_preprocessed_signal_once(self) -> None:
        observed = []
        class MockTrainedModel:
            backend = "torchscript"
            is_demo = False
            def predict(self, signal, options):
                observed.append(signal.copy())
                return np.asarray([.1, .1, .6, .1, .1])
        with patch("app.TorchScriptAdapter", return_value=MockTrainedModel()):
            client = TestClient(create_app(self.configs))
        response = client.post("/predict", json={"signal": self.raw.tolist(), "model": "cnn"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        expected = load_team_preprocessing().preprocess_data(self.raw.reshape(1, 180))[0]
        self.assertEqual(len(observed), 1)
        np.testing.assert_array_equal(observed[0], expected)
        self.assertEqual(body["prediction"]["code"], "V")
        self.assertFalse(body["is_demo"])
        self.assertEqual(body["preprocessing"]["pipeline"], "team")
        self.assertEqual(len(body["preprocessing"]["signal"]), 180)
        self.assertEqual(client.post("/predict", json={"signal": demo_samples()["N"].tolist(), "model": "cnn"}).status_code, 422)

    def test_missing_team_checkpoint_is_503_before_wrong_length_preprocessing(self) -> None:
        client = TestClient(create_app(self.configs))
        for signal in (self.raw.tolist(), demo_samples()["N"].tolist()):
            response = client.post("/predict", json={"signal": signal, "model": "cnn"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["detail"]["code"], "checkpoint_unavailable")
        descriptor = client.get("/models").json()["models"][0]
        self.assertFalse(descriptor["is_demo"])
        self.assertEqual(descriptor["status"], "unavailable")
        self.assertEqual(descriptor["input_length"], 180)
        self.assertEqual(descriptor["input_shape"], [1, 1, 180])
        self.assertEqual(descriptor["pipeline"], "team")
        self.assertEqual(descriptor["sample_rate"], 360)
        health = client.get("/health").json()
        self.assertEqual(health["team_input_length"], 180)
        self.assertEqual(health["input_length"], 256)
        self.assertEqual(health["models"][0]["pipeline"], "team")

    def test_no_checkpoint_remains_demo_even_with_team_override(self) -> None:
        configs = demo_configs()
        configs["cnn"] = ModelConfig("cnn", pipeline="team", input_layout="channels_first")
        client = TestClient(create_app(configs))
        descriptor = client.get("/models").json()["models"][0]
        self.assertTrue(descriptor["is_demo"])
        self.assertEqual(descriptor["pipeline"], "demo")
        self.assertEqual(descriptor["input_shape"], [1, 1, 256])
        body = client.post("/predict", json={"signal": self.raw.tolist(), "model": "cnn"}).json()
        self.assertEqual(body["preprocessing"]["pipeline"], "demo")
        self.assertEqual(len(body["preprocessing"]["signal"]), 256)

    def test_environment_checkpoint_defaults_to_team_and_explicit_override(self) -> None:
        with patch.dict("os.environ", {"ECG_CNN_CHECKPOINT": "models/best.ts.pt"}, clear=True):
            configs = environment_configs()
            self.assertEqual(configs["cnn"].pipeline, "team")
            self.assertEqual(configs["cnn"].input_shape(), [1, 1, 180])
            self.assertEqual(configs["rnn"].pipeline, "demo")
        with patch.dict("os.environ", {"ECG_CNN_CHECKPOINT": "models/best.ts.pt", "ECG_CNN_PIPELINE": "demo", "ECG_RNN_CHECKPOINT": "models/rnn.ts.pt"}, clear=True):
            configs = environment_configs()
            self.assertEqual(configs["cnn"].pipeline, "demo")
            self.assertEqual(configs["cnn"].input_shape(), [1, 1, 256])
            self.assertEqual(configs["rnn"].input_shape(), [1, 180, 1])

    def test_invalid_pipeline_configuration_is_unavailable(self) -> None:
        configs = demo_configs()
        configs["cnn"] = ModelConfig("cnn", pipeline="unknown")
        client = TestClient(create_app(configs))
        self.assertEqual(client.get("/models").json()["models"][0]["status"], "unavailable")
        self.assertEqual(client.post("/predict", json={"signal": self.raw.tolist(), "model": "cnn"}).status_code, 503)
        self.assertEqual(client.post("/preprocess", json={"signal": self.raw.tolist(), "model": "cnn"}).status_code, 503)


if __name__ == "__main__":
    unittest.main()
