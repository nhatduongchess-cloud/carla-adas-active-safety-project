"""Compressed frame replay format for perception benchmarks without CARLA."""

import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


class ReplayWriter:
    def __init__(self, output_dir, metadata=None, append=True):
        self.root = Path(output_dir).resolve()
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self.index = open(self.index_path, "a" if append else "w", encoding="utf-8")
        meta = {"schema_version": SCHEMA_VERSION, **dict(metadata or {})}
        with open(self.root / "manifest.json", "w", encoding="utf-8") as stream:
            json.dump(meta, stream, indent=2, ensure_ascii=False)

    def write(self, frame_id, timestamp, rgb, lidar, radar=None, labels=None,
              sensor_states=None):
        frame_id = int(frame_id)
        name = f"{frame_id:010d}.npz"
        path = self.frames_dir / name
        np.savez_compressed(
            path, rgb=np.asarray(rgb, dtype=np.uint8),
            lidar=np.asarray(lidar, dtype=np.float32),
            radar=np.asarray(radar if radar is not None else [], dtype=np.float32))
        record = {
            "frame_id": frame_id, "timestamp": float(timestamp),
            "file": f"frames/{name}", "labels": labels or {},
            "sensor_states": sensor_states or {},
        }
        self.index.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    def close(self):
        if not self.index.closed:
            self.index.flush()
            self.index.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class ReplayReader:
    def __init__(self, replay_dir):
        self.root = Path(replay_dir).resolve()
        with open(self.root / "manifest.json", "r", encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        if int(self.manifest.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported replay schema")

    def __iter__(self):
        with open(self.root / "index.jsonl", "r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                path = (self.root / record["file"]).resolve()
                if self.root not in path.parents:
                    raise ValueError(f"replay path escapes root: {path}")
                with np.load(path, allow_pickle=False) as arrays:
                    yield record, {name: arrays[name].copy() for name in arrays.files}
