from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterable

from .config import (
    PACKAGE_ROOT,
    cifar_source_checkpoint,
    default_report_dir,
    exp1_metrics_root,
    exp2_results_root,
    load_config,
)
from .io_utils import human_bytes, sha256, write_json


MAX_PACKAGE_BYTES = 150 * 1024 * 1024


def _package_files() -> Iterable[tuple[Path, str]]:
    excluded_parts = {
        "dist", "reports", "results", "historical", "artifacts", "__pycache__",
        ".venv", ".venv-smoke", ".git", ".pytest_cache", ".cache",
        "sample_workspace",
    }
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if (
            not path.is_file()
            or path == PACKAGE_ROOT / "artifact_manifest.json"
            or path.suffix.lower() == ".zip"
            or any(part in excluded_parts for part in path.relative_to(PACKAGE_ROOT).parts)
        ):
            continue
        yield path, path.relative_to(PACKAGE_ROOT).as_posix()


def _legacy_exp1_sources(workspace: Path) -> Iterable[tuple[Path, str]]:
    root = workspace / "EXP1"
    selected_roots = [root, root / "datasets", root / "models" / "group1"]
    seen: set[Path] = set()
    for selected in selected_roots:
        if not selected.exists():
            continue
        iterator = selected.glob("*.py") if selected == root else selected.rglob("*.py")
        for path in sorted(iterator):
            if path in seen or "__pycache__" in path.parts:
                continue
            seen.add(path)
            relative = path.relative_to(root).as_posix()
            yield path, f"legacy_source/EXP1/{relative}"


def build_server_package(
    workspace: Path,
    *,
    destination: Path | None = None,
    profile: str = "server-minimal",
) -> dict[str, object]:
    if profile != "server-minimal":
        raise ValueError("only server-minimal is supported")
    destination = destination or PACKAGE_ROOT / "dist" / "gleep-server-minimal.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[Path, str, str]] = []

    for path, archive_name in _package_files():
        entries.append((path, archive_name, "implementation"))
    for path, archive_name in _legacy_exp1_sources(workspace):
        entries.append((path, archive_name, "legacy_source"))
    for path in sorted(exp1_metrics_root(workspace).rglob("*.json")):
        relative = path.relative_to(exp1_metrics_root(workspace)).as_posix()
        entries.append((path, f"results/published/EXP1/results_metrics/{relative}", "historical_result"))
    for path in sorted(exp2_results_root(workspace).rglob("*.json")):
        relative = path.relative_to(exp2_results_root(workspace)).as_posix()
        entries.append((path, f"results/published/EXP2/result/{relative}", "historical_result"))
    for model in ("ResNet18", "ResNet34"):
        path = cifar_source_checkpoint(workspace, model)
        if not path.exists():
            raise FileNotFoundError(f"required source checkpoint is missing: {path}")
        archive_name = f"artifacts/checkpoints/Pretrain/{model}/40_64_0.001/best_model.pth"
        entries.append((path, archive_name, "source_checkpoint"))

    manifest_entries = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, archive_name, role in entries:
            digest = sha256(path)
            archive.write(path, archive_name)
            manifest_entries.append({
                "path": archive_name,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "source": str(path),
            })
        config = load_config()
        embedded_manifest = {
            "profile": profile,
            "official_repository": config["official_repository"],
            "official_commit": config["official_commit"],
            "entries": manifest_entries,
            "excluded": [
                "EXP1/results_f (19.18 GiB feature/logit/label cache)",
                "Transferlearning/checkpoint downstream task checkpoints (38.35 GiB tree)",
                "raw datasets",
                "ImageNet torchvision weights",
            ],
        }
        archive.writestr(
            "artifact_manifest.json",
            json.dumps(embedded_manifest, ensure_ascii=False, indent=2) + "\n",
        )

    size = destination.stat().st_size
    report = {
        "profile": profile,
        "archive": str(destination),
        "archive_bytes": size,
        "archive_size": human_bytes(size),
        "archive_sha256": sha256(destination),
        "entry_count": len(manifest_entries) + 1,
        "within_150_mib_limit": size <= MAX_PACKAGE_BYTES,
        "entries": manifest_entries,
    }
    write_json(default_report_dir() / "package_manifest.json", report)
    if size > MAX_PACKAGE_BYTES:
        raise RuntimeError(
            f"server package is {human_bytes(size)}, exceeding the 150 MiB acceptance limit"
        )
    return report
