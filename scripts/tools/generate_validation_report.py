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
    if value is None:
        return "·" * width                     # not measured: no bar, not an empty bar
    ratio = (value / target if higher else target / max(value, 1e-9)) if target else 0.0
    filled = max(0, min(width, int(round(width * min(1.0, ratio)))))
    return "█" * filled + "░" * (width - filled)


def _number(value):
    """A finite float, or None. Missing is never turned into 0."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def worst(scenario_rows, extract, higher_is_worse=True):
    """(worst value, cases with a value, total cases). Cases without the
    metric are counted, not dropped: a worst-case over 3 of 45 cases says so."""
    values = [v for v in (_number(extract(row)) for row in scenario_rows) if v is not None]
    if not values:
        return None, 0, len(scenario_rows)
    return (max(values) if higher_is_worse else min(values)), len(values), len(scenario_rows)


def stage(row, name, key):
    return row.get("stage_metrics", {}).get("stages", {}).get(name, {}).get(key)


def verdict(value, target, higher):
    if value is None:
        return "NOT EVALUATED"
    return "PASS" if (value >= target if higher else 0.0 <= value <= target) else "FAIL"


def fmt(value, unit, coverage=None):
    if value is None:
        text = "N/A"
    else:
        text = f"{value:.2f} {unit}"
    return f"{text} ({coverage[0]}/{coverage[1]} cases)" if coverage else text


def total(scenario_rows, key):
    """Sum of a count over cases, or None if any case lacks a valid count."""
    counts = [_number(row.get(key)) for row in scenario_rows]
    if any(c is None or c < 0 for c in counts):
        return None
    return int(sum(counts))


def show_total(items):
    value = total(items, "collisions")
    return "N/A" if value is None else value


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
    pass_rate = _number(summary.get("pass_rate_pct")) if summary.get("scenarios") else None
    rows = [("Scenario pass rate", pass_rate, 95.0, True, "%", None)]
    scenario_rows = scenarios.get("scenarios", []) if scenarios else []
    if scenario_rows:
        safety_p99, *safety_cov = worst(scenario_rows, lambda r: stage(r, "safety_loop", "p99_ms"))
        perception_p95, *perception_cov = worst(
            scenario_rows, lambda r: stage(r, "perception", "p95_ms"))
        radar_min, *radar_cov = worst(scenario_rows, lambda r: r.get("radar_availability"),
                                      higher_is_worse=False)
        rows.extend([
            ("Worst scenario safety p99", safety_p99, 25.0, False, "ms", safety_cov),
            ("Worst scenario perception p95", perception_p95, 50.0, False, "ms", perception_cov),
            ("Minimum radar availability", None if radar_min is None else radar_min * 100.0,
             99.5, True, "%", radar_cov),
        ])
    if replay:
        perception = replay.get("stages", {}).get("object", {})
        safety = replay.get("stages", {}).get("safety_geometry", {})
        rows.extend([
            ("Perception p95", _number(perception.get("p95_ms")), 50.0, False, "ms", None),
            ("Safety geometry p99", _number(safety.get("p99_ms")), 25.0, False, "ms", None),
            ("VRAM peak", _number(replay.get("gpu_peak_mb")), 7680.0, False, "MB", None),
        ])
    lines = ["# Self-Driving Perception — simulation validation report", "",
             "Generated only from machine-readable artifacts. A metric that was not "
             "measured is shown as N/A / NOT EVALUATED, never as 0; worst-case rows "
             "state how many cases carried the metric.", "",
             "## Acceptance overview", "",
             "| KPI | Observed | Target | Result |", "|---|---:|---:|---|"]
    for name, value, target, higher, unit, coverage in rows:
        lines.append(f"| {name} | {fmt(value, unit, coverage)} | "
                     f"{'≥' if higher else '≤'} {target:.2f} {unit} | "
                     f"{verdict(value, target, higher)} |")
    lines.extend(["", "```text"])
    for name, value, target, higher, unit, _ in rows:
        shown = "N/A" if value is None else f"{value:.2f}{unit}"
        lines.append(f"{name:24} {bar(value, target, higher)} {shown}")
    suite = scenarios.get("suite", {}) if scenarios else {}
    lines.extend(["```", "", "## Scenario gates", "",
                  f"- Gate status: `{gate.get('status', 'NOT_EVALUATED')}`"
                  f" (policy `{gate.get('policy', 'core_plus_catalog')}`)",
                  f"- Core: `{gate.get('core', {}).get('observed', '0/0')}` (requires 18/18)",
                  f"- Catalog: `{gate.get('catalog', {}).get('observed', '0/0')}` (requires ≥43/45)",
                  f"- Collision-free valid runs: `{gate.get('no_collision_in_valid_runs')}`"])
    if suite:
        lines.append("- Cases: " + ", ".join(
            f"{k} `{suite.get(k)}`" for k in ("planned", "attempted", "passed", "failed",
                                               "invalid", "error", "not_run")))
    if scenario_rows:
        collisions = total(scenario_rows, "collisions")
        frame_errors = total(scenario_rows, "sensor_frame_errors")
        reaction, *reaction_cov = worst(scenario_rows, lambda r: r.get("reaction_delay_s"))
        clearance, *clearance_cov = worst(scenario_rows,
                                          lambda r: r.get("scenario_min_clearance_m"),
                                          higher_is_worse=False)
        safety_misses = total([{"v": stage(r, "safety_loop", "deadline_misses")}
                               for r in scenario_rows], "v")
        perception_misses = total([{"v": stage(r, "perception", "deadline_misses")}
                                   for r in scenario_rows], "v")

        def show(value):
            return "N/A (not recorded for every case)" if value is None else value

        lines.extend(["", "## Runtime and safety evidence", "",
                      f"- Collisions: `{show(collisions)}`",
                      f"- Sensor frame errors: `{show(frame_errors)}`",
                      f"- Maximum reaction delay: `{fmt(reaction, 's', reaction_cov)}`",
                      f"- Minimum clearance (as recorded; published reports before "
                      f"2026-09-25 hold centre-to-centre distance here): "
                      f"`{fmt(clearance, 'm', clearance_cov)}`",
                      f"- Safety deadline misses: `{show(safety_misses)}`",
                      f"- Perception deadline misses: `{show(perception_misses)}` (p95 gate remains authoritative)"])
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
                f"{show_total(items)} | "
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
        "<!doctype html><meta charset='utf-8'><title>Simulation validation report</title>"
        "<style>body{max-width:980px;margin:40px auto;font:16px system-ui;line-height:1.5}"
        "pre{background:#111;color:#b9f6ca;padding:18px;border-radius:8px;overflow:auto}</style>"
        f"<pre>{html.escape(markdown)}</pre>", encoding="utf-8")
    print(json.dumps({"markdown": str(output), "html": str(html_path)}, indent=2))


if __name__ == "__main__":
    main()
