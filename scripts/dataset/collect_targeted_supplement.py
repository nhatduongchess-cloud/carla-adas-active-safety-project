"""Collect a resumable Car/Motorcycle/TrafficLight balancing supplement."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.dataset.collect_week2_dataset import carla_preflight, write_json


WEATHERS = ("clear", "light_rain", "heavy_rain", "fog", "storm")
TOWNS = (("Town05", 5042), ("Town10HD_Opt", 10042))


def build_targeted_plan(frames_per_job=50):
    frames_per_job = int(frames_per_job)
    if frames_per_job <= 0:
        raise ValueError("frames_per_job must be positive")
    jobs = []
    for town, seed_base in TOWNS:
        for weather_index, weather in enumerate(WEATHERS):
            jobs.append({
                "job_index": len(jobs),
                "town": town,
                "weather": weather,
                "frames": frames_per_job,
                "seed": seed_base + weather_index,
            })
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-per-job", type=int, default=50)
    parser.add_argument("--start-job", type=int, default=0)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--min-free-gb", type=float, default=25.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.output).resolve()
    jobs = build_targeted_plan(args.frames_per_job)
    selected = [job for job in jobs if job["job_index"] >= args.start_job]
    if args.max_jobs is not None:
        selected = selected[:max(0, args.max_jobs)]
    plan = {
        "schema_version": 1,
        "purpose": "balance Car, Motorcycle and TrafficLight without modifying baseline",
        "output": str(root),
        "targeted_classes": ["Car", "Motorcycle", "TrafficLight"],
        "target_instances_per_class": 2,
        "target_frames": sum(job["frames"] for job in jobs),
        "jobs": jobs,
        "selected_job_indices": [job["job_index"] for job in selected],
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    if not selected:
        raise SystemExit("no targeted jobs selected")

    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "targeted_collection_plan.json", plan)
    state_path = root / "targeted_collection_state.json"
    state = {"schema_version": 1, "completed_jobs": [], "failed_job": None}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    collector = Path(__file__).with_name("collect_carla_dataset.py")
    for job in selected:
        index = job["job_index"]
        if index in state.get("completed_jobs", []):
            print(f"[targeted] skip completed job {index}")
            continue
        system = carla_preflight()
        print(f"[targeted] preflight job={index}: {system}")
        command = [
            sys.executable, str(collector), "--output", str(root),
            "--frames", str(job["frames"]), "--town", job["town"],
            "--weather", job["weather"], "--seed", str(job["seed"]),
            "--npcs", "0", "--walkers", "0", "--two-wheelers", "0",
            "--heavy-vehicles", "0",
            "--targeted-classes", "Car,Motorcycle,TrafficLight",
            "--target-instances-per-class", "2",
            "--min-free-gb", str(args.min_free_gb),
        ]
        try:
            subprocess.run(command, check=True)
        except Exception as exc:
            state["failed_job"] = {"job_index": index, "error": str(exc)}
            write_json(state_path, state)
            raise
        state.setdefault("completed_jobs", []).append(index)
        state["completed_jobs"] = sorted(set(state["completed_jobs"]))
        state["failed_job"] = None
        write_json(state_path, state)

    validator = Path(__file__).with_name("validate_dataset.py")
    subprocess.run([
        sys.executable, str(validator), str(root), "--output",
        str(root / "dataset_integrity.json")], check=True)


if __name__ == "__main__":
    main()
