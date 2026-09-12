"""Chấm điểm tuân thủ L3 + xuất báo cáo JSON (logic thuần, test được).

Một kịch bản ĐẠT khi:
  1. Phân loại ODD khớp kỳ vọng (expect_odd trong weather_config.yaml).
  2. Không va chạm (collisions == 0).
  3. Hành vi MRM đúng: ODD VIOLATION -> có thực thi MRM; ngược lại -> KHÔNG MRM.
  4. Không "ngắt nhầm" (false disengagement): không kích Fallback khi ODD còn NORMAL.
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

    collisions = kpi_summary.get("collisions", 0)
    if collisions > 0:
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


def build_report(results, title="Mercedes-Benz DRIVE PILOT L3 Validation Report", meta=None):
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
    core_names = set((meta or {}).get("core_scenarios", []))
    core_results = [result for result in results if result.get("name") in core_names]
    valid_results = [result for result in results if result.get("triggered", True)]
    collision_free = all(result.get("collisions", 0) == 0 for result in valid_results)
    core_ready = len(core_results) >= 18
    catalog_ready = len(results) >= 45
    report["acceptance_gate"] = {
        "core": {
            "required": "18/18",
            "observed": f"{sum(bool(r.get('pass')) for r in core_results)}/{len(core_results)}",
            "pass": (all(r.get("pass") for r in core_results) and core_ready)
                    if core_ready else None,
        },
        "catalog": {
            "required": ">=43/45",
            "observed": f"{passed}/{total}",
            "pass": (passed >= 43 and catalog_ready) if catalog_ready else None,
        },
        "no_collision_in_valid_runs": collision_free,
        "status": ("PASS" if core_ready and catalog_ready and collision_free
                   and all(r.get("pass") for r in core_results) and passed >= 43
                   else "FAIL" if core_ready and catalog_ready else "NOT_EVALUATED"),
    }
    if meta:
        report["run"] = {k: v for k, v in meta.items() if k != "acceptance_criteria"}
    return report


def write_report(path, results, title="Mercedes-Benz DRIVE PILOT L3 Validation Report", meta=None):
    report = build_report(results, title=title, meta=meta)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
