"""Reproducible runtime/model provenance without network access."""

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    absolute = os.path.abspath(path)
    if not os.path.isfile(absolute):
        return {"path": absolute, "exists": False}
    return {"path": absolute, "exists": True, "size_bytes": os.path.getsize(absolute),
            "sha256": sha256_file(absolute)}


def runtime_manifest(model_paths=None, carla_version=None, config=None):
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "carla": carla_version,
        "models": [file_record(path) for path in (model_paths or [])],
        "config": dict(config or {}),
    }
    try:
        import torch
        manifest["torch"] = torch.__version__
        manifest["cuda_available"] = bool(torch.cuda.is_available())
        manifest["cuda_runtime"] = getattr(torch.version, "cuda", None)
        manifest["gpu"] = (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else None)
    except Exception as exc:
        manifest["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import cv2
        manifest["opencv"] = cv2.__version__
    except Exception:
        pass
    return manifest


def write_manifest(path, manifest):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = os.path.abspath(path) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, os.path.abspath(path))


def verify_artifact(path, expected_sha256):
    if not os.path.isfile(path):
        return False, "missing artifact"
    actual = sha256_file(path)
    if actual.lower() != str(expected_sha256).lower():
        return False, f"sha256 mismatch: expected {expected_sha256}, actual {actual}"
    return True, actual

