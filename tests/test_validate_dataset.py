"""Focused offline tests for declared CARLA dataset modalities."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.dataset.validate_dataset import inspect_dataset


class DatasetInspectionTests(unittest.TestCase):
    def test_empty_dataset_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(inspect_dataset(directory)['valid'])

    def test_nonfinite_points_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            base, stem = self.make_sample(directory)
            np.savez_compressed(base / 'radar' / f'{stem}.npz',
                                points=np.full((1, 4), np.nan, np.float32))
            self.assertFalse(inspect_dataset(directory)['valid'])

    def make_sample(self, root):
        base = Path(root) / "train" / "Town02" / "clear"
        stem = "seed0000000042_000000"
        for name in ("rgb", "road_line", "semantic", "depth", "lidar", "radar"):
            (base / name).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 3)).save(base / "rgb" / f"{stem}.jpg")
        Image.new("L", (4, 3)).save(base / "road_line" / f"{stem}.png")
        Image.new("L", (4, 3)).save(base / "semantic" / f"{stem}.png")
        Image.fromarray(np.zeros((3, 4), dtype=np.uint16)).save(
            base / "depth" / f"{stem}.png")
        for name in ("lidar", "radar"):
            np.savez_compressed(
                base / name / f"{stem}.npz",
                points=np.zeros((1, 4), dtype=np.float32))
        (base / "annotations.jsonl").write_text(json.dumps({
            "sample_id": stem,
            "frame_id": 1,
            "town": "Town02",
            "weather": "clear",
            "split": "train",
            "objects": [],
        }) + "\n", encoding="utf-8")
        return base, stem

    def test_readable_declared_modalities_are_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_sample(directory)
            self.assertTrue(inspect_dataset(directory)["valid"])

    def test_corrupt_image_and_npz_are_both_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            base, stem = self.make_sample(directory)
            (base / "semantic" / f"{stem}.png").write_bytes(b"broken")
            (base / "lidar" / f"{stem}.npz").write_bytes(b"broken")

            report = inspect_dataset(directory)

            self.assertFalse(report["valid"])
            self.assertEqual(
                {issue["modality"] for issue in report["issues"]
                 if issue["type"] == "unreadable_file"},
                {"semantic", "lidar"})


if __name__ == "__main__":
    unittest.main()
