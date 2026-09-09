"""Reject credentials, private absolute paths, large files, and binary artifacts."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


FORBIDDEN_SUFFIXES = {".npy", ".npz", ".pth", ".pt", ".h5", ".safetensors"}
PRIVATE_PATTERNS = (
    re.compile(rb"(?:password|passwd)\s*[:=]\s*[^\s]+", re.IGNORECASE),
    re.compile(b"[A-Za-z0-9._-]+" + b"@" + b"hpc", re.IGNORECASE),
    re.compile(b"/data/" + b"user/", re.IGNORECASE),
    re.compile(rb"[A-Za-z]:[\\/]work_" + b"list", re.IGNORECASE),
)


def tracked_files(root: Path) -> list[Path]:
    process = subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return [root / item.decode("utf-8") for item in process.stdout.split(b"\0") if item]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    problems = []
    for path in tracked_files(root):
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            problems.append(f"missing tracked file: {relative}")
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"tracked binary artifact: {relative}")
        if path.stat().st_size > 50 * 1024 * 1024:
            problems.append(f"tracked file exceeds 50 MiB: {relative}")
        raw = path.read_bytes()
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(raw):
                problems.append(f"private path or credential marker in: {relative}")
                break
    if problems:
        print("\n".join(problems))
        return 1
    print(f"PASS: {len(tracked_files(root))} tracked files contain no forbidden payloads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
