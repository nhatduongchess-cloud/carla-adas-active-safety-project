"""Chấm điểm tuân thủ L3 + xuất báo cáo JSON (logic thuần, test được).

Một kịch bản ĐẠT khi:
  1. Phân loại ODD khớp kỳ vọng (expect_odd trong weather_config.yaml).
  2. Không va chạm (collisions == 0).
  3. Hành vi MRM đúng: ODD VIOLATION -> có thực thi MRM; ngược lại -> KHÔNG MRM.
  4. Không "ngắt nhầm" (false disengagement): không kích Fallback khi ODD còn NORMAL.
  5. No late braking: no brake onset with TTC below 0.8 s. This was computed and
     reported but never gated - a profile with five late brakes still passed.
     A number named after a safety event that cannot fail a run reads as a
     check that was done; it is now a check that is done.

A collision count that is missing is a failure, not zero.

Default report title is project-neutral. It previously read "Mercedes-Benz DRIVE
PILOT L3 Validation Report", which presents a student simulation as a validation
report for a real manufacturer's product. It is not one.
"""
# fmt: off
# isort: skip_file
import os
import json
import datetime


def assess_profile(name, expect_odd, actual_odd, kpi_summary,
                   mrm_triggered, tor_count, false_disengagements, late_braking_events):
    reasons = []
    ok = True

    if actual_odd != expect_odd:
        ok = False
        reasons.append(f"ODD sai: kỳ vọng {expect_odd}, thực tế {actual_odd}")

    raw_collisions = kpi_summary.get("collisions")
    collisions = _count(raw_collisions)
    if collisions is None:
        ok = False
        reasons.append("collision count was not observed" if raw_collisions is None
                       else f"collision count is not a valid count: {raw_collisions!r}")
    elif collisions > 0:
        ok = False
        reasons.append(f"{collisions} va chạm")

    if expect_odd == "VIOLATION" and not mrm_triggered:
        ok = False
        reasons.append("ODD VIOLATION nhưng KHÔNG thực thi MRM")
    if expect_odd != "VIOLATION" and mrm_triggered:
        ok = False
        reasons.append("Thực thi MRM ngoài vùng ODD violation")

    if false_disengagements > 0:
        ok = False
        reasons.append(f"{false_disengagements} lần ngắt nhầm (false disengagement)")

    if late_braking_events > 0:
        ok = False
        reasons.append(f"{late_braking_events} late braking event(s): brake onset with TTC < 0.8 s")

    return {
        "profile": name,
        "expect_odd": expect_odd,
        "actual_odd": actual_odd,
        "mrm_triggered": bool(mrm_triggered),
        "tor_count": int(tor_count),
        "false_disengagements": int(false_disengagements),
        "late_braking_events": int(late_braking_events),
        "kpi": kpi_summary,
        "pass": ok,
        "reasons": reasons,
    }


def build_report(results, title="CARLA L3 ODD / MRM validation report (simulation)", meta=None):
    """Gói kết quả thành báo cáo V&V. `meta` (tùy chọn) ghi lại NGỮ CẢNH KIỂM THỬ
    (seed, ngưỡng cấu hình, tiêu chí nghiệm thu...) để báo cáo TÁI LẬP được — đúng
    tinh thần một artifact kỹ thuật, không phải log tùy hứng."""
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    acceptance = {
        "collisions": "== 0",
        "reacted": "true (FSM rời NORMAL khi có vật cản trong làn)",
        "note": "Chỉ số êm ái (max_decel/jerk) đã loại gai va chạm; xem impact_decel_spikes.",
    }
    if meta and meta.get("acceptance_criteria"):
        acceptance = dict(meta["acceptance_criteria"])

    report = {
        "title": title,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "acceptance_criteria": acceptance,
        "summary": {
            "scenarios": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate_pct": round(100.0 * passed / max(1, total), 1),
        },
        "scenarios": results,
    }
    # The gate counts CASES, not rows. A case is one (scenario, seed, weather,
    # fault) run. Counting rows let 45 copies of a single passing case satisfy
    # both "45 scenarios" and "18 core scenarios" and report PASS; the
    # 2026-09-25 evidence review reproduced exactly that.
    distinct, duplicate_rows = {}, 0
    for result in results:
        key = _case_key(result)
        if key in distinct:
            duplicate_rows += 1
        else:
            distinct[key] = result
    cases = list(distinct.values())
    case_passed = sum(1 for r in cases if r.get("pass") is True)

    core_names = set((meta or {}).get("core_scenarios", []))
    core_results = [r for r in cases if r.get("name") in core_names]
    # Every declared core scenario must actually be present; 18 runs of one of
    # them is not the core suite.
    core_missing = sorted(core_names - {r.get("name") for r in core_results})
    core_ready = len(core_results) >= 18 and not core_missing

    valid_results = [r for r in cases if r.get("triggered", True)]
    collision_counts = [_count(r.get("collisions")) for r in valid_results]
    if not valid_results:
        collision_free = None           # nothing ran; "no collisions" would be vacuous
    elif any(c is None for c in collision_counts):
        collision_free = False          # an unobserved count is not a zero
    else:
        collision_free = all(c == 0 for c in collision_counts)

    catalog_ready = len(cases) >= 45
    core_all_pass = all(r.get("pass") is True for r in core_results)
    if duplicate_rows:
        status = "INVALID"
    elif core_ready and catalog_ready:
        status = ("PASS" if collision_free is True and core_all_pass and case_passed >= 43
                  else "FAIL")
    else:
        status = "NOT_EVALUATED"

    report["acceptance_gate"] = {
        "core": {
            "required": "18/18",
            "observed": f"{sum(r.get('pass') is True for r in core_results)}/{len(core_results)}",
            "missing_core_scenarios": core_missing,
            "pass": (core_all_pass if core_ready else None) if not duplicate_rows else None,
        },
        "catalog": {
            "required": ">=43/45",
            "observed": f"{case_passed}/{len(cases)}",
            "pass": ((case_passed >= 43) if catalog_ready else None) if not duplicate_rows else None,
        },
        "duplicate_case_rows": duplicate_rows,
        "no_collision_in_valid_runs": collision_free,
        "status": status,
    }
    if meta:
        report["run"] = {k: v for k, v in meta.items() if k != "acceptance_criteria"}
    return report


def write_report(path, results, title="CARLA L3 ODD / MRM validation report (simulation)", meta=None):
    report = build_report(results, title=title, meta=meta)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def _case_key(result):
    return (result.get("name"), result.get("seed"), result.get("weather"), result.get("fault"))


def _count(value):
    """A non-negative whole number, or None if the value is not one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")) or f < 0 or f != int(f):
        return None
    return int(f)
