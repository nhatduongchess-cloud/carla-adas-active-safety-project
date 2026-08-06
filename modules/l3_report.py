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


def build_report(results, title="Mercedes-Benz DRIVE PILOT L3 Validation Report"):
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    return {
        "title": title,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "scenarios": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate_pct": round(100.0 * passed / max(1, total), 1),
        },
        "scenarios": results,
    }


def write_report(path, results, title="Mercedes-Benz DRIVE PILOT L3 Validation Report"):
    report = build_report(results, title=title)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
