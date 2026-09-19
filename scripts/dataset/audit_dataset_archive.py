"""Audit a local dataset ZIP before any extraction or training use."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".txt", ".yaml", ".yml", ".json", ".md",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_member_name(name):
    """Return a safe POSIX archive name or raise ``ValueError``."""
    raw = str(name).replace("\\", "/")
    if not raw or "\x00" in raw:
        raise ValueError("empty_or_nul_name")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("absolute_or_traversal_path")
    if any(":" in part for part in path.parts):
        raise ValueError("drive_or_stream_path")
    return path.as_posix()


def audit_zip(path, expected_sha256=None, max_entries=100000,
              max_total_uncompressed_gb=5.0, max_single_file_mb=100.0,
              max_compression_ratio=500.0):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = sha256_file(path)
    expected = str(expected_sha256 or "").strip().lower()
    issues = []
    if expected and actual_sha256.lower() != expected:
        issues.append({"type": "sha256_mismatch", "expected": expected,
                       "actual": actual_sha256})

    extension_counts = Counter()
    seen = set()
    total_compressed = total_uncompressed = 0
    with zipfile.ZipFile(path, "r") as archive:
        entries = archive.infolist()
        if len(entries) > int(max_entries):
            issues.append({"type": "too_many_entries", "count": len(entries),
                           "limit": int(max_entries)})
        for info in entries:
            try:
                normalized = normalized_member_name(info.filename)
            except ValueError as exc:
                issues.append({"type": str(exc), "entry": info.filename})
                continue
            key = normalized.casefold()
            if key in seen:
                issues.append({"type": "duplicate_normalized_path",
                               "entry": info.filename})
            seen.add(key)
            if info.flag_bits & 0x1:
                issues.append({"type": "encrypted_entry", "entry": info.filename})
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
                issues.append({"type": "symlink_entry", "entry": info.filename})
            if info.is_dir():
                continue
            suffix = PurePosixPath(normalized).suffix.lower()
            extension_counts[suffix or "<none>"] += 1
            if suffix not in ALLOWED_EXTENSIONS:
                issues.append({"type": "disallowed_extension",
                               "entry": info.filename, "extension": suffix})
            total_compressed += int(info.compress_size)
            total_uncompressed += int(info.file_size)
            if info.file_size > float(max_single_file_mb) * 1024 ** 2:
                issues.append({"type": "single_file_too_large",
                               "entry": info.filename, "bytes": info.file_size})
            ratio = info.file_size / max(1, info.compress_size)
            if ratio > float(max_compression_ratio):
                issues.append({"type": "suspicious_compression_ratio",
                               "entry": info.filename, "ratio": round(ratio, 2)})
        max_total = float(max_total_uncompressed_gb) * 1024 ** 3
        if total_uncompressed > max_total:
            issues.append({"type": "archive_too_large_uncompressed",
                           "bytes": total_uncompressed,
                           "limit_bytes": int(max_total)})

    return {
        "schema_version": 1,
        "archive": str(path),
        "sha256": actual_sha256,
        "expected_sha256": expected or None,
        "entries": len(entries),
        "compressed_bytes": total_compressed,
        "uncompressed_bytes": total_uncompressed,
        "extension_counts": dict(sorted(extension_counts.items())),
        "issues": issues,
        "safe_to_extract": not issues,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        report = audit_zip(args.archive, args.expected_sha256)
    except (OSError, zipfile.BadZipFile) as exc:
        report = {
            "schema_version": 1,
            "archive": str(Path(args.archive).resolve()),
            "issues": [{"type": "invalid_or_unreadable_zip", "error": str(exc)}],
            "safe_to_extract": False,
        }
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(output)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["safe_to_extract"] else 1)


if __name__ == "__main__":
    main()
