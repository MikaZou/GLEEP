"""Validate local EXP2 arrays against the manifests downloaded with server caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "artifacts" / "cache" / "exp2" / "published"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    rows = []
    label_hashes = set()
    for model in ("ResNet18", "ResNet34"):
        for source in ("CIFAR10", "ImageNet"):
            directory = CACHE / model / source
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            for name, expected in manifest["arrays"].items():
                path = directory / expected["path"]
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                digest = sha256(path)
                if name == "labels":
                    label_hashes.add(digest)
                rows.append({
                    "model": model,
                    "source": source,
                    "array": name,
                    "bytes_ok": path.stat().st_size == expected["bytes"],
                    "sha256_ok": digest == expected["sha256"],
                    "shape_ok": list(array.shape) == expected["shape"],
                    "dtype_ok": str(array.dtype) == expected["dtype"],
                    "manifest_finite": expected["finite"],
                })
    complete = all(
        row["bytes_ok"] and row["sha256_ok"] and row["shape_ok"]
        and row["dtype_ok"] and row["manifest_finite"]
        for row in rows
    ) and len(label_hashes) == 1
    report = {
        "manifest_origin": "downloaded together with the server-generated EXP2 cache",
        "groups": 4,
        "arrays": len(rows),
        "all_label_hashes_identical": len(label_hashes) == 1,
        "rows": rows,
        "complete": complete,
    }
    destination = ROOT / "results" / "reproduced" / "exp2_cache_manifest_validation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# EXP2 server-cache manifest validation",
        "",
        "The four manifests were downloaded together with the server-generated cache. "
        "Every local array was re-hashed and checked against its manifest.",
        "",
        "| Model | Source | Array | Bytes | SHA256 | Shape | Dtype | Finite |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['source']} | {row['array']} | "
            f"{'PASS' if row['bytes_ok'] else 'FAIL'} | "
            f"{'PASS' if row['sha256_ok'] else 'FAIL'} | "
            f"{'PASS' if row['shape_ok'] else 'FAIL'} | "
            f"{'PASS' if row['dtype_ok'] else 'FAIL'} | "
            f"{'PASS' if row['manifest_finite'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        f"All four label hashes identical: **{'PASS' if len(label_hashes) == 1 else 'FAIL'}**",
        "",
        f"Verdict: **{'PASS' if complete else 'FAIL'}**",
        "",
    ])
    destination.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"arrays": len(rows), "complete": complete}, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
