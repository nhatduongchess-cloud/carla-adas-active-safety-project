"""Aggregate JSON artifacts into reproducible Markdown and standalone HTML."""

import argparse
import html
import json
import math
from pathlib import Path


def load_optional(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def bar(value, target, higher=True, width=24):
    ratio = (value / target if higher else target / max(value, 1e-9)) if target else 0.0
    filled = max(0, min(width, int(round(width * min(1.0, ratio)))))
    return "█" * filled + "░" * (width - filled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="logs/scenario_test_report.json")
    parser.add_argument("--replay", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--weather-scenarios", default=None)
    parser.add_argument("--runtime-manifest", default="logs/runtime_manifest.json")
    parser.add_argument("--output", default="logs/safety_demo_report.md")
    args = parser.parse_args()
    scenarios = load_optional(args.scenarios)
    replay = load_optional(args.replay)
    dataset = load_optional(args.dataset)
    weather_report = load_optional(args.weather_scenarios)
    manifest = load_optional(args.runtime_manifest)

    summary = scenarios.get("summary", {}) if scenarios else {}
    gate = scenarios.get("acceptance_gate", {}) if scenarios else {}
    rows = [
        ("Scenario pass rate", float(summary.get("pass_rate_pct", 0.0)), 95.0, True, "%"),
    ]
    scenario_rows = scenarios.get("scenarios", []) if scenarios else []
    if scenario_rows:
        safety_p99 = max(float(row.get("stage_metrics", {}).get("stages", {})
                                .get("safety_loop", {}).get("p99_ms") or 0.0)
                         for row in scenario_rows)
        perception_p95 = max(float(row.get("stage_metrics", {}).get("stages", {})
                                    .get("perception", {}).get("p95_ms") or 0.0)
                             for row in scenario_rows)
        radar_min = min(float(row.get("radar_availability", 0.0))
                        for row in scenario_rows)
        rows.extend([
            ("Worst scenario safety p99", safety_p99, 25.0, False, "ms"),
            ("Worst scenario perception p95", perception_p95, 50.0, False, "ms"),
            ("Minimum radar availability", radar_min * 100.0, 99.5, True, "%"),
        ])
    if replay:
        perception = replay.get("stages", {}).get("object", {})
        safety = replay.get("stages", {}).get("safety_geometry", {})
        rows.extend([
            ("Perception p95", float(perception.get("p95_ms") or 0.0), 50.0, False, "ms"),
            ("Safety geometry p99", float(safety.get("p99_ms") or 0.0), 25.0, False, "ms"),
            ("VRAM peak", float(replay.get("gpu_peak_mb") or 0.0), 7680.0, False, "MB"),
        ])
    lines = ["# Self-Driving Perception — Safety Demo validation", "",
             "Generated only from machine-readable artifacts; missing runs are not treated as PASS.", "",
             "## Acceptance overview", "",
             "| KPI | Observed | Target | Result |", "|---|---:|---:|---|"]
    for name, value, target, higher, unit in rows:
        passed = value >= target if higher else 0.0 < value <= target
        lines.append(f"| {name} | {value:.2f} {unit} | "
                     f"{'≥' if higher else '≤'} {target:.2f} {unit} | "
                     f"{'PASS' if passed else 'FAIL/NOT RUN'} |")
    lines.extend(["", "```text"])
    for name, value, target, higher, unit in rows:
        lines.append(f"{name:24} {bar(value, target, higher)} {value:.2f}{unit}")
    lines.extend(["```", "", "## Scenario gates", "",
                  f"- Gate status: `{gate.get('status', 'NOT_EVALUATED')}`",
                  f"- Core: `{gate.get('core', {}).get('observed', '0/0')}` (requires 18/18)",
                  f"- Catalog: `{gate.get('catalog', {}).get('observed', '0/0')}` (requires ≥43/45)",
                  f"- Collision-free valid runs: `{gate.get('no_collision_in_valid_runs', False)}`"])
    if scenario_rows:
        collisions = sum(int(row.get("collisions", 0)) for row in scenario_rows)
        frame_errors = sum(int(row.get("sensor_frame_errors", 0))
                           for row in scenario_rows)
        reactions = [float(row["reaction_delay_s"]) for row in scenario_rows
                     if row.get("reaction_delay_s") is not None]
        clearances = [float(row["scenario_min_clearance_m"])
                      for row in scenario_rows
                      if row.get("scenario_min_clearance_m") is not None
                      and math.isfinite(float(row["scenario_min_clearance_m"]))]
        safety_misses = sum(int(row.get("stage_metrics", {}).get("stages", {})
                                .get("safety_loop", {}).get("deadline_misses") or 0)
                            for row in scenario_rows)
        perception_misses = sum(int(row.get("stage_metrics", {}).get("stages", {})
                                    .get("perception", {}).get("deadline_misses") or 0)
                                for row in scenario_rows)
        lines.extend(["", "## Runtime and safety evidence", "",
                      f"- Collisions: `{collisions}`",
                      f"- Sensor frame errors: `{frame_errors}`",
                      f"- Maximum reaction delay: `{max(reactions) if reactions else 'N/A'} s`",
                      f"- Minimum ground-truth clearance: `{min(clearances) if clearances else 'N/A'} m`",
                      f"- Safety deadline misses: `{safety_misses}`",
                      f"- Perception deadline misses: `{perception_misses}` (p95 gate remains authoritative)"])
    if weather_report:
        grouped = {}
        for row in weather_report.get("scenarios", []):
            grouped.setdefault(row.get("weather", "unknown"), []).append(row)
        lines.extend(["", "## Weather / ODD matrix", "",
                      "| Weather | Runs | Pass | Collisions | ODD violation seen | MRM seen |",
                      "|---|---:|---:|---:|---:|---:|"])
        for weather, items in sorted(grouped.items()):
            lines.append(
                f"| {weather} | {len(items)} | {sum(bool(x.get('pass')) for x in items)} | "
                f"{sum(int(x.get('collisions', 0)) for x in items)} | "
                f"{sum(bool(x.get('odd_violation_seen')) for x in items)} | "
                f"{sum(bool(x.get('mrm_seen')) for x in items)} |")
    if dataset:
        lines.extend(["", "## Dataset integrity", "",
                      f"- Frames: `{dataset.get('frames', 0)}`",
                      f"- Valid: `{dataset.get('valid', False)}`",
                      f"- Split leakage groups: `{dataset.get('split_leakage_groups', 0)}`",
                      f"- Class counts: `{json.dumps(dataset.get('class_counts', {}), ensure_ascii=False)}`"])
    if manifest:
        lines.extend(["", "## Provenance", "",
                      f"- Python: `{str(manifest.get('python', '')).split()[0]}`",
                      f"- Torch/CUDA: `{manifest.get('torch')}` / `{manifest.get('cuda_runtime')}`",
                      f"- GPU: `{manifest.get('gpu')}`"])
        for model in manifest.get("models", []):
            lines.append(f"- Model: `{model.get('path')}` — SHA-256 `{model.get('sha256', 'missing')}`")
    lines.extend(["", "## Not evaluated / pending", "",
                  "- Learned-lane accuracy and runtime are not evaluated: UFLDv2 is still blocked by the external-code approval gate.",
                  "- Object mAP/VRU recall and traffic-light macro-F1 require the planned held-out CARLA dataset and training run.",
                  "- The 30-minute soak, ablation matrix, dataset integrity gate and measured VRAM peak have not yet been supplied as artifacts."])
    markdown = "\n".join(lines) + "\n"
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    html_path = output.with_suffix(".html")
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Safety Demo V&V</title>"
        "<style>body{max-width:980px;margin:40px auto;font:16px system-ui;line-height:1.5}"
        "pre{background:#111;color:#b9f6ca;padding:18px;border-radius:8px;overflow:auto}</style>"
        f"<pre>{html.escape(markdown)}</pre>", encoding="utf-8")
    print(json.dumps({"markdown": str(output), "html": str(html_path)}, indent=2))


if __name__ == "__main__":
    main()
