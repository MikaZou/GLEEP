"""Validate retained NumPy arrays without loading complete large arrays into RAM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def finite_by_chunks(array: np.ndarray, chunk_rows: int = 4096) -> bool:
    if array.ndim == 0:
        return bool(np.isfinite(array))
    return all(
        bool(np.isfinite(array[start : start + chunk_rows]).all())
        for start in range(0, len(array), chunk_rows)
    )


def validate(root: Path) -> dict[str, Any]:
    index = json.loads((root / "artifacts" / "index.json").read_text(encoding="utf-8"))
    rows = []
    for entry in index["entries"]:
        if not entry["path"].endswith(".npy"):
            continue
        path = root / entry["path"]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        row = {
            "path": entry["path"],
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "shape_ok": list(array.shape) == entry["shape"],
            "dtype_ok": str(array.dtype) == entry["dtype"],
            "finite": finite_by_chunks(array),
        }
        rows.append(row)
        print(f"[{len(rows):03d}] {entry['path']}: {row['shape']} {row['dtype']}", flush=True)

    exp1_root = root / "artifacts" / "cache" / "exp1" / "group1"
    pairs = []
    for output_path in sorted(exp1_root.glob("*_output.npy")):
        label_path = output_path.with_name(output_path.name.replace("_output.npy", "_label.npy"))
        output = np.load(output_path, mmap_mode="r", allow_pickle=False)
        label = np.load(label_path, mmap_mode="r", allow_pickle=False)
        pairs.append({
            "stem": output_path.name.removesuffix("_output.npy"),
            "output_rank": output.ndim,
            "label_rank": label.ndim,
            "sample_count_match": len(output) == len(label),
        })

    exp2_root = root / "artifacts" / "cache" / "exp2" / "published"
    exp2_labels = sorted(exp2_root.glob("*/*/labels.npy"))
    label_arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in exp2_labels]
    exp2_label_order_identical = bool(label_arrays) and all(
        np.array_equal(label_arrays[0], other) for other in label_arrays[1:]
    )

    invalid_rows = [
        row for row in rows if not (row["shape_ok"] and row["dtype_ok"] and row["finite"])
    ]
    invalid_pairs = [
        pair for pair in pairs
        if pair["output_rank"] != 2 or pair["label_rank"] != 1 or not pair["sample_count_match"]
    ]
    return {
        "array_count": len(rows),
        "exp1_pair_count": len(pairs),
        "exp2_label_files": len(exp2_labels),
        "exp2_label_order_identical": exp2_label_order_identical,
        "invalid_arrays": invalid_rows,
        "invalid_exp1_pairs": invalid_pairs,
        "complete": not invalid_rows and not invalid_pairs and exp2_label_order_identical,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = validate(root)
    destination = args.output or root / "results" / "reproduced" / "artifact_array_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
