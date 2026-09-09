"""Build the tracked manifest for Git-ignored experiment artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


OFFICIAL_COMMIT = "20d8d7f91c8c5f409ef31081257676d655a27f29"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_metadata(path: Path) -> tuple[list[int] | None, str | None]:
    if path.suffix != ".npy":
        return None, None
    import numpy as np

    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return list(array.shape), str(array.dtype)


def iter_artifacts(root: Path) -> Iterable[tuple[Path, list[str], str, str, str]]:
    exp1 = root / "artifacts" / "cache" / "exp1" / "group1"
    for path in sorted(exp1.glob("*_output.npy")):
        yield path, ["exp1"], "source-model logits", "local-only", (
            "python -m gleep_repro run exp1 --source stream --metrics gleep leep"
        )
    for path in sorted(exp1.glob("*_label.npy")):
        yield path, ["exp1"], "target labels", "local-only", (
            "python -m gleep_repro run exp1 --source stream --metrics gleep leep"
        )

    exp2 = root / "artifacts" / "cache" / "exp2" / "published"
    for path in sorted(item for item in exp2.rglob("*") if item.is_file()):
        role = "cache manifest" if path.suffix == ".json" else "source-model feature/logit cache"
        yield path, ["exp2"], role, "local-and-server", (
            "python -m gleep_repro run exp2 --mode cache --semantics published"
        )

    checkpoints = root / "artifacts" / "checkpoints" / "Pretrain"
    for path in sorted(checkpoints.rglob("best_model.pth")):
        yield path, ["source-checkpoints"], "CIFAR10 source checkpoint", "local-and-server", (
            "python -m gleep_repro run exp2 --mode train --retain-checkpoints best"
        )


def build(root: Path) -> dict[str, Any]:
    entries = []
    for number, (path, profiles, role, status, rebuild) in enumerate(iter_artifacts(root), 1):
        relative = path.relative_to(root).as_posix()
        print(f"[{number:03d}] hashing {relative}", flush=True)
        shape, dtype = array_metadata(path)
        entries.append({
            "path": relative,
            "profiles": profiles,
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "shape": shape,
            "dtype": dtype,
            "storage_status": status,
            "remote_uri": None,
            "rebuild": rebuild,
        })
    profiles = {}
    for profile in ("exp1", "exp2", "source-checkpoints"):
        selected = [entry for entry in entries if profile in entry["profiles"]]
        profiles[profile] = {
            "files": len(selected),
            "bytes": sum(entry["bytes"] for entry in selected),
        }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_commit": OFFICIAL_COMMIT,
        "remote_policy": (
            "Large artifacts are excluded from Git. remote_uri remains null until a stable "
            "Hugging Face or cloud-storage release is published."
        ),
        "profiles": profiles,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    result = build(root)
    destination = root / "artifacts" / "index.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {destination.relative_to(root).as_posix()} with {len(result['entries'])} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
