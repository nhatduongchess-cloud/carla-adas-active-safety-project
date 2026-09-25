"""Reproducible runtime/model provenance without network access."""

import hashlib
import json
import os
import platform
import subprocess
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


def git_state(repo_root):
    """Commit, dirty flag and a hash of the uncommitted diff, or a reason.

    Read once per run, outside any loop. Nothing is inferred: if git is not
    available the fields are None and ``unknown_reason`` says why.
    """
    def run(*args):
        return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True,
                              text=True, timeout=10, check=True).stdout
    try:
        commit = run("rev-parse", "HEAD").strip()
        status = run("status", "--porcelain", "--untracked-files=no")
        diff = run("diff", "HEAD") if status.strip() else ""
    except Exception as exc:
        return {"git_commit": None, "git_dirty": None, "git_diff_sha256": None,
                "unknown_reason": f"{type(exc).__name__}: {exc}"}
    return {"git_commit": commit, "git_dirty": bool(status.strip()),
            "git_diff_sha256": (hashlib.sha256(diff.encode("utf-8")).hexdigest()
                                if diff else None),
            "unknown_reason": None}


def carla_versions(client):
    """Client and server versions as reported by CARLA, or a reason."""
    versions = {"client": None, "server": None, "unknown_reason": None}
    try:
        versions["client"] = client.get_client_version()
        versions["server"] = client.get_server_version()
    except Exception as exc:
        versions["unknown_reason"] = f"{type(exc).__name__}: {exc}"
    return versions


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

