"""Create an immutable W1.1 baseline manifest without network or data mutation.

The manifest deliberately records the current dirty working tree rather than
pretending that ``git HEAD`` identifies the starting point.  Large/untracked
directories are recorded as collapsed entries so a baseline cannot accidentally
walk or rewrite a dataset, cache, or generated-output tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config as cfg


ROOT = Path(__file__).resolve().parent
RELEVANT_PACKAGES = (
    "carla",
    "numpy",
    "opencv-python",
    "Pillow",
    "pygame",
    "PyYAML",
    "scipy",
    "torch",
    "torchvision",
    "ultralytics",
)


def _run(command: list[str], *, timeout: float = 10.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:  # diagnostics must remain best-effort
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_snapshot() -> dict[str, Any]:
    head = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain=v1", "--untracked-files=normal"])
    diff_stat = _run(["git", "diff", "--stat"])
    staged_stat = _run(["git", "diff", "--cached", "--stat"])

    entries: list[dict[str, Any]] = []
    for line in status["stdout"].splitlines():
        if len(line) < 4:
            continue
        code = line[:2]
        relative = line[3:]
        # Rename records are intentionally not resolved: retaining the raw
        # status is safer than guessing a path in a dirty user-owned tree.
        if " -> " in relative:
            entries.append({"status": code, "path": relative, "kind": "rename", "sha256": None})
            continue
        path = (ROOT / relative).resolve()
        record: dict[str, Any] = {"status": code, "path": relative}
        try:
            if path.is_file():
                record.update({
                    "kind": "file",
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                })
            elif path.is_dir():
                record.update({
                    "kind": "directory",
                    "sha256": None,
                    "hash_scope": "collapsed_by_git_status; not recursively read",
                })
            else:
                record.update({"kind": "missing_or_special", "sha256": None})
        except OSError as exc:
            record.update({"kind": "unreadable", "sha256": None, "read_error": str(exc)})
        entries.append(record)

    return {
        "head": head["stdout"] or None,
        "status_command": status["command"],
        "status_returncode": status["returncode"],
        "status_stderr": status["stderr"] or None,
        "dirty_entries": entries,
        "dirty_entry_count": len(entries),
        "diff_stat": diff_stat["stdout"],
        "staged_diff_stat": staged_stat["stdout"],
    }


def _packages() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in RELEVANT_PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _gpu_snapshot() -> dict[str, Any]:
    result = _run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    return result


def _carla_snapshot(host: str, port: int) -> dict[str, Any]:
    try:
        import carla

        client = carla.Client(host, int(port))
        client.set_timeout(5.0)
        world = client.get_world()
        settings = world.get_settings()
        snapshot = world.get_snapshot()
        return {
            "reachable": True,
            "host": host,
            "port": int(port),
            "server_version": client.get_server_version(),
            "map": world.get_map().name,
            "frame": int(snapshot.frame),
            "timestamp": float(snapshot.timestamp.elapsed_seconds),
            "synchronous_mode": bool(settings.synchronous_mode),
            "fixed_delta_seconds": settings.fixed_delta_seconds,
        }
    except Exception as exc:
        return {
            "reachable": False,
            "host": host,
            "port": int(port),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _carla_processes() -> dict[str, Any]:
    # Read only: retain command lines only for CARLA and project clients.
    script = (
        "$items = Get-CimInstance Win32_Process | Where-Object { "
        "($_.Name -like 'CarlaUE4*') -or "
        "(($_.Name -like 'python*') -and ($_.CommandLine -match 'Carla Simulator')) "
        "}; $items | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    result = _run(["powershell", "-NoProfile", "-Command", script])
    if result["returncode"] != 0 or not result["stdout"]:
        return {"query": result, "processes": []}
    try:
        value = json.loads(result["stdout"])
        return {"query": result, "processes": value if isinstance(value, list) else [value]}
    except json.JSONDecodeError:
        return {"query": result, "processes": [], "parse_error": "invalid JSON"}


def _effective_config() -> dict[str, Any]:
    names = (
        "FPS", "FIXED_DELTA", "CAM_WIDTH", "CAM_HEIGHT", "CAM_FOV",
        "YOLO_IMGSZ", "DETECT_EVERY_N", "LANE_EVERY_N", "RENDER_EVERY_N",
        "ENABLE_RADAR", "YOLO_DEVICE", "LEARNED_LANE_DEVICE", "USE_ENSEMBLE",
        "CAMERA_SENSOR_TICK_S", "LIDAR_SENSOR_TICK_S", "RADAR_SENSOR_TICK_S",
        "RADAR_X", "RADAR_Y", "RADAR_Z", "RADAR_RANGE_M",
        "RADAR_HORIZONTAL_FOV", "RADAR_VERTICAL_FOV", "RADAR_POINTS_PER_SECOND",
        "RADAR_OPTIONAL_TIMEOUT_S", "RADAR_MAX_MISSING_FRAMES",
    )
    values: dict[str, Any] = {}
    for name in names:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                values[name] = value
    return values


def build_manifest(host: str, port: int, launch_args: str | None) -> dict[str, Any]:
    generated = datetime.now(timezone.utc)
    model_paths = []
    for name in ("YOLO_MODEL_A", "YOLO_MODEL_B", "LEARNED_LANE_MODEL_PATH"):
        value = getattr(cfg, name, None)
        if value and value not in model_paths:
            model_paths.append(value)
    models = []
    for raw_path in model_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        item: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
        if path.is_file():
            item.update({"size_bytes": path.stat().st_size, "sha256": _sha256(path)})
        models.append(item)

    disk = shutil.disk_usage(ROOT)
    return {
        "schema_version": 1,
        "manifest_type": "W1.1_baseline",
        "generated_at_utc": generated.isoformat(),
        "local_timezone": datetime.now().astimezone().tzname(),
        "root": str(ROOT),
        "platform": platform.platform(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "packages": _packages(),
        "git": _git_snapshot(),
        "carla": _carla_snapshot(host, port),
        "carla_processes": _carla_processes(),
        "launch_args": launch_args,
        "gpu": _gpu_snapshot(),
        "resources": {
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
            "disk_used_bytes": disk.used,
        },
        "models": models,
        "effective_config": _effective_config(),
        "provenance_notes": [
            "Read-only inventory; no dependency/model download or installation.",
            "Dirty working tree is intentional user-owned baseline.",
            "Collapsed directories are not recursively hashed to avoid touching datasets/caches.",
            "CARLA readiness is a point-in-time observation; this is not a crash-recovery claim.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new JSON path; existing files are rejected")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--launch-args", default=None,
                        help="exact server command line captured by the operator")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.host, args.port, args.launch_args)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"written": str(output), "dirty_entries": manifest["git"]["dirty_entry_count"],
                      "carla": manifest["carla"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
