"""Resumable 20k-frame CARLA collection orchestrator for the Week-2 plan.

The script invokes the audited local collector only; it downloads or installs
nothing. Jobs are grouped by town to minimize expensive CARLA map reloads.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List


WEATHERS = ("clear", "light_rain", "heavy_rain", "fog", "storm")
SPLIT_TOWNS = {
    "train": ("Town01", "Town02", "Town03", "Town04"),
    "validation": ("Town05",),
    "test": ("Town10HD",),
}
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
TRAFFIC_LEVELS = (8, 15, 22)
SEEDS = (42, 1337, 2026)
COLLECTOR_PROFILE = {
    "ego_control": "custom",
    "target_speed_kmh": 20.0,
    "radar_policy": "required",
    "jpeg_quality": 90,
    "targeted_classes": [],
    "target_instances_per_class": 2,
}
STATE_SCHEMA_VERSION = 2
PLAN_SCHEMA_VERSION = 2


def distribute(total: int, count: int) -> List[int]:
    base, remainder = divmod(int(total), int(count))
    return [base + (index < remainder) for index in range(count)]


def build_collection_plan(total_frames: int = 20000) -> List[Dict]:
    total_frames = int(total_frames)
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    split_totals = {
        "train": round(total_frames * SPLIT_RATIOS["train"]),
        "validation": round(total_frames * SPLIT_RATIOS["validation"]),
    }
    split_totals["test"] = total_frames - sum(split_totals.values())
    jobs = []
    global_index = 0
    for split, towns in SPLIT_TOWNS.items():
        pairs = [(town, weather) for town in towns for weather in WEATHERS]
        allocations = distribute(split_totals[split], len(pairs))
        for (town, weather), frames in zip(pairs, allocations):
            if frames <= 0:
                continue
            jobs.append({
                "job_index": global_index,
                "split": split,
                "town": town,
                "weather": weather,
                "frames": frames,
                "npcs": TRAFFIC_LEVELS[global_index % len(TRAFFIC_LEVELS)],
                "walkers": 8,
                "two_wheelers": 4,
                "heavy_vehicles": 2,
                "seed": SEEDS[global_index % len(SEEDS)],
            })
            global_index += 1
    if sum(job["frames"] for job in jobs) != total_frames:
        raise AssertionError("collection plan frame allocation is inconsistent")
    return jobs


def carla_preflight():
    import carla

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    settings = world.get_settings()
    if settings.synchronous_mode:
        raise RuntimeError(
            "CARLA is already synchronous; another client or a crashed run may own ticks")
    return {"map": world.get_map().name, "frame": world.get_snapshot().frame}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def plan_hash(plan):
    """Hash only the immutable collection contract, not resume selectors."""
    encoded = json.dumps(
        plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def job_contract(job):
    """Reuse the collector's exact sensor/control contract for resume identity."""
    import config as cfg
    from collect_carla_dataset import capture_contract
    from modules.dataset_labels import POLICY

    plan = {key: value for key, value in job.items() if key != "job_index"}
    plan.update(COLLECTOR_PROFILE)
    plan["label_policy"] = POLICY
    return capture_contract(plan, cfg)


def job_hash(job):
    return plan_hash(job_contract(job))


def run_checked(command, timeout_s, runner=subprocess.run):
    timeout_s = float(timeout_s)
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("subprocess timeout must be finite and positive")
    return runner(command, check=True, timeout=timeout_s)


def manifest_path(root: Path, job: Dict) -> Path:
    return (Path(root) / job["split"] / job["town"] / job["weather"]
            / "manifest.json")


def _verify_manifest_identity(manifest, job):
    expected = {key: value for key, value in job.items() if key != "job_index"}
    expected.update(COLLECTOR_PROFILE)
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"capture manifest does not match job: {mismatches}")
    expected_hash = job_hash(job)
    if manifest.get("job_hash") != expected_hash:
        raise RuntimeError(
            "capture manifest has a stale or mismatched capture contract hash")


def verify_completed_job(root: Path, job: Dict):
    """Fail closed unless the exact job has a verified completed manifest."""
    path = manifest_path(root, job)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"completed manifest is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"completed manifest is unreadable: {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"capture manifest must be an object: {path}")
    _verify_manifest_identity(manifest, job)
    if manifest.get("status") != "completed":
        raise RuntimeError(
            f"capture manifest is not completed: {manifest.get('status')!r}")
    if manifest.get("total_job_samples") != job["frames"]:
        raise RuntimeError(
            "capture manifest sample count does not match requested frames")
    if manifest.get("cleanup", {}).get("verified") is not True:
        raise RuntimeError("capture manifest cleanup is not verified")
    from collect_carla_dataset import sample_stem, validate_resume_state
    try:
        rows = validate_resume_state(path.parent / 'annotations.jsonl',
                                     path.parent, job_hash(job))
    except (ValueError, OSError) as exc:
        raise RuntimeError(f'completed sample verification failed: {exc}') from exc
    expected_ids = {sample_stem(job['seed'], i) for i in range(job['frames'])}
    if {row['sample_id'] for row in rows} != expected_ids:
        raise RuntimeError('completed manifest does not have the exact saved sample set')
    return manifest


def load_state(path: Path, expected_plan_hash: str, job_indices):
    if not path.is_file():
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "plan_hash": expected_plan_hash,
            "completed_jobs": [],
            "failed_job": None,
            "validation": {"status": "pending"},
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"collection state is unreadable: {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise RuntimeError("collection state must be an object")
    if (state.get("schema_version") != STATE_SCHEMA_VERSION
            or state.get("plan_hash") != expected_plan_hash):
        raise RuntimeError(
            "collection state belongs to a different plan; use a new output root")
    completed = state.get("completed_jobs")
    valid = set(job_indices)
    if (not isinstance(completed, list)
            or any(isinstance(index, bool) or not isinstance(index, int)
                   for index in completed)
            or len(completed) != len(set(completed))
            or not set(completed).issubset(valid)):
        raise RuntimeError("collection state has invalid completed job indices")
    return state


def ensure_plan(path: Path, plan, expected_hash: str):
    record = {**plan, "plan_hash": expected_hash}
    if not path.is_file():
        write_json(path, record)
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"collection plan is unreadable: {path}: {exc}") from exc
    observed_hash = existing.pop("plan_hash", None) if isinstance(existing, dict) else None
    if observed_hash != expected_hash or plan_hash(existing) != expected_hash:
        raise RuntimeError(
            "output root contains a different collection plan; refusing overwrite")


def _validate_args(args):
    if args.start_job < 0:
        raise SystemExit("--start-job must be non-negative")
    if args.max_jobs is not None and args.max_jobs < 0:
        raise SystemExit("--max-jobs must be non-negative")
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        raise SystemExit("--min-free-gb must be finite and non-negative")
    for name in ("job_timeout_s", "validator_timeout_s"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and positive")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=20000)
    parser.add_argument("--start-job", type=int, default=0)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--min-free-gb", type=float, default=25.0)
    parser.add_argument("--job-timeout-s", type=float, default=1800.0)
    parser.add_argument("--validator-timeout-s", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _validate_args(args)

    root = Path(args.output).resolve()
    jobs = build_collection_plan(args.frames)
    selected = [job for job in jobs if job["job_index"] >= args.start_job]
    if args.max_jobs is not None:
        selected = selected[:args.max_jobs]
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "output": str(root),
        "target_frames": args.frames,
        "split_targets": {
            split: sum(job["frames"] for job in jobs if job["split"] == split)
            for split in SPLIT_TOWNS
        },
        "collector_profile": COLLECTOR_PROFILE,
        "job_hashes": {str(job["job_index"]): job_hash(job) for job in jobs},
        "jobs": jobs,
    }
    digest = plan_hash(plan)
    print(json.dumps({
        **plan,
        "plan_hash": digest,
        "selected_job_indices": [job["job_index"] for job in selected],
        "job_timeout_s": args.job_timeout_s,
        "validator_timeout_s": args.validator_timeout_s,
    }, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    if not selected:
        raise SystemExit("no collection jobs selected")

    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "collection_state.json"
    state = load_state(
        state_path, digest, (job["job_index"] for job in jobs))
    ensure_plan(root / "collection_plan.json", plan, digest)

    collector = Path(__file__).with_name("collect_carla_dataset.py")
    for job in selected:
        index = job["job_index"]
        if index in state.get("completed_jobs", []):
            verify_completed_job(root, job)
            print(f"[week2] skip completed job {index}")
            continue
        path = manifest_path(root, job)
        if path.is_file():
            existing_manifest = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing_manifest, dict):
                raise RuntimeError(f"capture manifest must be an object: {path}")
            _verify_manifest_identity(existing_manifest, job)
            if existing_manifest.get("status") == "completed":
                verify_completed_job(root, job)
                state.setdefault("completed_jobs", []).append(index)
                state["completed_jobs"].sort()
                state["failed_job"] = None
                write_json(state_path, state)
                print(f"[week2] reconciled completed job {index}")
                continue
        system = carla_preflight()
        print(f"[week2] preflight job={index}: {system}")
        command = [
            sys.executable, str(collector), "--output", str(root),
            "--frames", str(job["frames"]), "--town", job["town"],
            "--weather", job["weather"], "--npcs", str(job["npcs"]),
            "--walkers", str(job["walkers"]),
            "--two-wheelers", str(job["two_wheelers"]),
            "--heavy-vehicles", str(job["heavy_vehicles"]),
            "--seed", str(job["seed"]), "--min-free-gb", str(args.min_free_gb),
            "--ego-control", COLLECTOR_PROFILE["ego_control"],
            "--target-speed-kmh", str(COLLECTOR_PROFILE["target_speed_kmh"]),
            "--radar-policy", COLLECTOR_PROFILE["radar_policy"],
            "--jpeg-quality", str(COLLECTOR_PROFILE["jpeg_quality"]),
            "--targeted-classes", "",
            "--target-instances-per-class",
            str(COLLECTOR_PROFILE["target_instances_per_class"]),
        ]
        try:
            run_checked(command, args.job_timeout_s)
            verify_completed_job(root, job)
        except Exception as exc:
            state["failed_job"] = {
                "job_index": index,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if isinstance(exc, subprocess.TimeoutExpired):
                state["failed_job"]["timeout_s"] = args.job_timeout_s
            write_json(state_path, state)
            raise
        state.setdefault("completed_jobs", []).append(index)
        state["completed_jobs"].sort()
        state["failed_job"] = None
        write_json(state_path, state)

    validator = Path(__file__).with_name("validate_dataset.py")
    state["validation"] = {"status": "running"}
    write_json(state_path, state)
    try:
        run_checked([
            sys.executable, str(validator), str(root), "--output",
            str(root / "dataset_integrity.json")], args.validator_timeout_s)
    except Exception as exc:
        state["validation"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if isinstance(exc, subprocess.TimeoutExpired):
            state["validation"]["timeout_s"] = args.validator_timeout_s
        write_json(state_path, state)
        raise
    state["validation"] = {"status": "completed"}
    write_json(state_path, state)


if __name__ == "__main__":
    main()
