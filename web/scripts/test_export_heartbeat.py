"""Check that the handoff exporter preserves RAW samples and rejects invalid rows."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from export_heartbeat import export_heartbeat


class ExportHeartbeatTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data_dir = self.root / "processed_data" / "split"
        self.data_dir.mkdir(parents=True)
        self.X = np.vstack((np.linspace(-0.4, 1.2, 180), np.linspace(2.0, -1.0, 180)))
        self.y = np.array([0, 2])
        self.save()

    def save(self):
        np.save(self.data_dir / "X_test.npy", self.X)
        np.save(self.data_dir / "y_test.npy", self.y)

    def export(self, **kwargs):
        return export_heartbeat("test", kwargs.pop("index", 1), kwargs.pop("output", "exports/beat.csv"), repo_root=self.root, **kwargs)

    def test_exports_exact_raw_values_and_reference_label(self):
        target, label = self.export()
        self.assertEqual(target, self.root / "exports" / "beat.csv")
        self.assertEqual(label, 2)
        self.assertEqual(target.read_text().splitlines()[0], "signal")
        np.testing.assert_array_equal(np.loadtxt(target, skiprows=1), self.X[1])
        np.testing.assert_array_equal(np.load(self.data_dir / "X_test.npy"), self.X)

    def test_rejects_negative_and_out_of_range_indexes(self):
        for index in (-1, 2):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.export(index=index)

    def test_rejects_wrong_length_and_label_count(self):
        for X, y in ((self.X[:, :179], self.y), (self.X, self.y[:1])):
            with self.subTest(shape=X.shape, labels=len(y)):
                self.X, self.y = X, y
                self.save()
                with self.assertRaises(ValueError):
                    self.export(index=0)

    def test_rejects_nonfinite_waveform(self):
        for invalid in (np.nan, np.inf):
            with self.subTest(value=invalid):
                self.X[1, 20] = invalid
                self.save()
                with self.assertRaises(ValueError):
                    self.export()

    def test_rejects_invalid_reference_label(self):
        for invalid in (5.0, 1.5, np.nan):
            with self.subTest(value=invalid):
                self.y = np.array([0, invalid])
                self.save()
                with self.assertRaises(ValueError):
                    self.export()

    def test_refuses_overwrite_until_explicitly_requested(self):
        target, _ = self.export()
        target.write_text("existing file\n")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(target.read_text(), "existing file\n")
        self.export(overwrite=True)
        self.assertEqual(target.read_text().splitlines()[0], "signal")

    def test_rejects_non_csv_output(self):
        with self.assertRaises(ValueError):
            self.export(output="processed_data/split/X_test.npy", overwrite=True)

    def test_cli_help_supports_windows_legacy_output_encoding(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("export_heartbeat.py")), "--help"],
            capture_output=True,
            env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
        self.assertIn("Ghi đè CSV", result.stdout.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
