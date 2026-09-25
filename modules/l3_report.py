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

A collision count that is missing is a failure, not zero. A profile with no
declared expected ODD is INVALID - it used to default to the observed state,
which graded the monitor against itself.

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
    invalid = []
    ok = True

    if expect_odd not in ("NORMAL", "DEGRADED", "VIOLATION"):
        ok = False
        invalid.append(f"expected ODD not declared or unknown: {expect_odd!r}")
    elif actual_odd != expect_odd:
        ok = False
        reasons.append(f"ODD sai: kỳ vọng {expect_odd}, thực tế {actual_odd}")

    raw_collisions = kpi_summary.get("collisions")
    collisions = _count(raw_collisions)
    if collisions is None:
        ok = False
        invalid.append("collision count was not observed" if raw_collisions is None
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
        "status": "FAIL" if reasons else "INVALID" if invalid else "PASS",
        "pass": ok,
        "reasons": reasons + invalid,
        "invalid_reasons": invalid,
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
    # Scenario rows carry "collisions"; L3 profile rows carry it in "kpi".
    collision_counts = [_count(r["collisions"] if "collisions" in r
                               else (r.get("kpi") or {}).get("collisions"))
                        for r in valid_results]
    if not valid_results:
        collision_free = None           # nothing ran; "no collisions" would be vacuous
    elif any(c is None for c in collision_counts):
        collision_free = False          # an unobserved count is not a zero
    else:
        collision_free = all(c == 0 for c in collision_counts)

    catalog_ready = len(cases) >= 45
    core_all_pass = all(r.get("pass") is True for r in core_results)

    # Case statuses. Rows written before statuses existed (the published
    # reports) have only `pass`; they are read as PASS/FAIL, so their verdicts
    # do not change.
    statuses = [case_status(r) for r in cases]
    counts = {k: statuses.count(k) for k in ("PASS", "FAIL", "INVALID", "ERROR")}
    planned = list((meta or {}).get("planned_case_ids") or [])
    attempted_ids = {case_id(r) for r in cases}
    not_run = sorted(set(planned) - attempted_ids) if planned else []
    unexpected = sorted(attempted_ids - set(planned)) if planned else []
    suite = {
        "planned": len(planned) if planned else None,
        "attempted": len(cases),
        "completed": len(cases) - counts["ERROR"],
        "valid": counts["PASS"] + sum(1 for r, st in zip(cases, statuses)
                                      if st == "FAIL" and not r.get("invalid_reasons")),
        "passed": counts["PASS"],
        "failed": counts["FAIL"],
        "invalid": counts["INVALID"],
        "error": counts["ERROR"],
        "not_run": len(not_run) if planned else None,
        "not_run_case_ids": not_run,
        "unexpected_case_ids": unexpected,
        "duplicate_case_rows": duplicate_rows,
        "pass_rate_denominators": {
            "passed_over_planned": (f"{counts['PASS']}/{len(planned)}" if planned else None),
            "passed_over_attempted": f"{counts['PASS']}/{len(cases)}",
        },
    }
    report["suite"] = suite

    policy = (meta or {}).get("suite_policy", "core_plus_catalog")
    if duplicate_rows or unexpected:
        status = "INVALID"
    elif policy == "all_planned_cases":
        # A suite that is not the 45-case catalog (e.g. 6 core x 5 weathers)
        # is judged against its own planned matrix, not a hard-coded 18/45.
        if not planned:
            status = "NOT_EVALUATED"
        elif counts["FAIL"]:
            status = "FAIL"
        elif counts["ERROR"] or counts["INVALID"]:
            status = "INVALID"
        elif not_run:
            status = "NOT_EVALUATED"
        else:
            status = "PASS" if collision_free is True else "NOT_EVALUATED"
    elif core_ready and catalog_ready:
        status = ("PASS" if collision_free is True and core_all_pass and case_passed >= 43
                  else "FAIL")
    else:
        status = "NOT_EVALUATED"
    # Whatever the policy, an ERROR/INVALID case or an unrun planned case
    # cannot sit under an overall PASS.
    if status == "PASS" and (counts["ERROR"] or counts["INVALID"] or not_run):
        status = "INVALID" if (counts["ERROR"] or counts["INVALID"]) else "NOT_EVALUATED"

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
        "policy": policy,
        "status": status,
    }
    if meta:
        report["run"] = {k: v for k, v in meta.items() if k != "acceptance_criteria"}
    return report


def write_report(path, results, title="CARLA L3 ODD / MRM validation report (simulation)", meta=None):
    """Write atomically (temp file + replace), with NaN/Infinity rejected.

    A crash mid-write can no longer leave a truncated report, and a checkpoint
    written after every case keeps the finished cases if the server dies.
    """
    report = build_report(results, title=title, meta=meta)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    text = json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return report


def case_id(result):
    """Stable identifier of one planned (scenario, seed, weather, fault) case."""
    name, seed, weather, fault = _case_key(result)
    return f"{name}|seed={seed}|weather={weather}|fault={fault}"


def case_status(result):
    """PASS / FAIL / INVALID / ERROR for one row, legacy rows included."""
    status = result.get("status")
    if status in ("PASS", "FAIL", "INVALID", "ERROR"):
        return status
    return "PASS" if result.get("pass") is True else "FAIL"


def exit_code_for(report):
    """0 only for an overall PASS. FAIL 1, INVALID 2, NOT_EVALUATED 3."""
    status = (report.get("acceptance_gate") or {}).get("status")
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}.get(status, 3)


def _json_safe(obj):
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _case_key(result):
    # L3 profile rows carry "profile", scenario rows carry "name".
    name = result.get("name", result.get("profile"))
    return (name, result.get("seed"), result.get("weather"), result.get("fault"))


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
