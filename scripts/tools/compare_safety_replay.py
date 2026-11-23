"""Đối chiếu quyết định Python đã ghi với FSM chạy lại từ đầu episode."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules.safety_replay import EXIT_CODES, compare_replay, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="thư mục có manifest.json và steps.jsonl")
    parser.add_argument("--report", required=True, type=Path, help="đường dẫn báo cáo JSON")
    args = parser.parse_args(argv)
    report = compare_replay(args.input)
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        if args.report.resolve() in {
            (args.input / "manifest.json").resolve(), (args.input / "steps.jsonl").resolve(),
            (args.input / "first_mismatch.json").resolve(),
        }:
            raise ValueError("--report must not overwrite the capture")
        write_json(args.report, report)
        if report["first_mismatch"] is not None:
            write_json(args.input / "first_mismatch.json", report["first_mismatch"])
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CODES["ERROR"]
    print(f"{report['status']}: {report['compared_steps']}/{report['expected_steps']} "
          f"steps compared. Report: {args.report}")
    if report["reason"]:
        print(report["reason"])
    return EXIT_CODES[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
