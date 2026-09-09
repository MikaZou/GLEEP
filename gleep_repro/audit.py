from __future__ import annotations

import ast
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import (
    PACKAGE_ROOT,
    cifar_source_checkpoint,
    default_report_dir,
    exp1_cache_root,
    exp1_metrics_root,
    exp2_results_root,
    load_config,
    runtime_data_root,
)
from .io_utils import (
    directory_inventory,
    human_bytes,
    markdown_table,
    portable_path,
    sha256,
    write_json,
    write_text,
)


TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "torch",
    "torchvision",
    "timm",
    "matplotlib",
    "Pillow",
)

KNOWN_OFFICIAL_PLACEHOLDERS = {
    "EXP1/datasets/imagenet.py": "Official unused ImageNet dataset stub is syntactically incomplete.",
    "EXP2/Ablation/test1.py": "Official out-of-scope ablation placeholder is empty.",
}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _torch_numpy_probe() -> dict[str, Any]:
    try:
        import numpy as np
        import torch

        value = torch.tensor([1.0]).numpy()
        return {"ok": bool(np.asarray(value)[0] == 1.0), "error": None}
    except BaseException as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _gpu_probe() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return {
            "ok": result.returncode == 0,
            "output": result.stdout.strip(),
            "error": result.stderr.strip() or None,
        }
    except BaseException as exc:
        return {"ok": False, "output": "", "error": f"{type(exc).__name__}: {exc}"}


def _gpu_compute_probe() -> dict[str, Any]:
    scripts = {
        "cudnn_conv": (
            "import torch; "
            "x=torch.randn(1,3,32,32,device='cuda'); "
            "m=torch.nn.Conv2d(3,8,3,padding=1).cuda(); "
            "print(tuple(m(x).shape))"
        ),
        "native_conv": (
            "import torch; torch.backends.cudnn.enabled=False; "
            "x=torch.randn(1,3,32,32,device='cuda'); "
            "m=torch.nn.Conv2d(3,8,3,padding=1).cuda(); "
            "print(tuple(m(x).shape))"
        ),
        "cublas_matmul": (
            "import torch; "
            "a=torch.randn(8,512,device='cuda'); b=torch.randn(512,100,device='cuda'); "
            "print(tuple((a@b).shape))"
        ),
    }
    results = {}
    for name, script in scripts.items():
        try:
            process = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            results[name] = {
                "ok": process.returncode == 0,
                "returncode": process.returncode,
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip(),
            }
        except BaseException as exc:
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return results


def _python_file_health(root: Path, workspace: Path) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    if not root.exists():
        return problems
    for path in sorted(root.rglob("*.py")):
        raw = path.read_bytes()
        if not raw or not any(raw):
            if path.name == "__init__.py":
                continue
            problems.append({"path": portable_path(path, workspace), "kind": "empty_or_all_nul", "bytes": len(raw)})
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            problems.append({"path": portable_path(path, workspace), "kind": "decode_error", "detail": str(exc)})
            continue
        try:
            ast.parse(text, filename=str(path))
        except (SyntaxError, ValueError) as exc:
            problems.append({
                "path": portable_path(path, workspace),
                "kind": "syntax_error",
                "detail": f"{type(exc).__name__}: {exc}",
            })
    return problems


def _exp1_cache_inventory(cache_root: Path) -> dict[str, dict[str, int]]:
    result = {name: {"files": 0, "bytes": 0} for name in ("feature", "output", "label", "other")}
    if not cache_root.exists():
        return result
    for path in cache_root.iterdir():
        if not path.is_file():
            continue
        kind = next((name for name in ("feature", "output", "label") if path.name.endswith(f"_{name}.npy")), "other")
        result[kind]["files"] += 1
        result[kind]["bytes"] += path.stat().st_size
    return result


def _exp2_json_counts(result_root: Path, workspace: Path) -> list[dict[str, Any]]:
    rows = []
    if not result_root.exists():
        return rows
    for path in sorted(result_root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            count = len(data) if isinstance(data, dict) else None
            error = None
        except BaseException as exc:
            count = None
            error = f"{type(exc).__name__}: {exc}"
        rows.append({"path": portable_path(path, workspace), "entries": count, "error": error})
    return rows


def _checkpoint_coverage(checkpoint_root: Path) -> list[dict[str, Any]]:
    rows = []
    for strategy in ("Finetune", "Retrain"):
        for model in ("ResNet18", "ResNet34"):
            for source in ("CIFAR10", "ImageNet"):
                base = checkpoint_root / strategy / model / source
                class_ids = set()
                if base.exists():
                    for path in base.glob("C*/30_64_0.001/best_model.pth"):
                        try:
                            class_ids.add(int(path.parents[1].name[1:]))
                        except ValueError:
                            pass
                rows.append({
                    "strategy": strategy,
                    "model": model,
                    "source": source,
                    "checkpoint_count": len(class_ids),
                    "missing_tasks": [value for value in range(2, 101) if value not in class_ids],
                })
    return rows


def _data_health(data_root: Path, workspace: Path) -> list[dict[str, Any]]:
    rows = []
    for relative in (
        Path("cifar-100-python") / "train",
        Path("cifar-100-python") / "test",
        Path("cifar-100-python") / "meta",
        Path("cifar-100-python.tar.gz"),
    ):
        path = data_root / relative
        if not path.exists():
            rows.append({"path": portable_path(path, workspace), "exists": False, "bytes": 0, "nul_prefix": None})
            continue
        with path.open("rb") as handle:
            prefix = handle.read(4096)
        rows.append({
            "path": portable_path(path, workspace),
            "exists": True,
            "bytes": path.stat().st_size,
            "nul_prefix": bool(prefix) and not any(prefix),
        })
    return rows


def _artifact_hashes(workspace: Path) -> list[dict[str, Any]]:
    paths = []
    paths.extend(sorted(exp1_metrics_root(workspace).rglob("*.json")))
    paths.extend(sorted(exp2_results_root(workspace).rglob("*.json")))
    for model in ("ResNet18", "ResNet34"):
        checkpoint = cifar_source_checkpoint(workspace, model)
        if checkpoint.exists():
            paths.append(checkpoint)
    rows = []
    for path in paths:
        rows.append({"path": portable_path(path, workspace), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def _audit_markdown(report: dict[str, Any]) -> str:
    cache = report["storage"]["exp1_cache_by_kind"]
    lines = [
        "# GLEEP workspace audit",
        "",
        f"Workspace: `{report['workspace']}`  ",
        f"Official source: `{report['official_commit']}`",
        "",
        "## Storage",
        "",
        markdown_table(
            ["Area", "Files", "Size"],
            [
                ("EXP1 feature cache", cache["feature"]["files"], human_bytes(cache["feature"]["bytes"])),
                ("EXP1 logits cache", cache["output"]["files"], human_bytes(cache["output"]["bytes"])),
                ("EXP1 labels", cache["label"]["files"], human_bytes(cache["label"]["bytes"])),
                ("EXP2 checkpoints", report["storage"]["exp2_checkpoints"]["files"], human_bytes(report["storage"]["exp2_checkpoints"]["bytes"])),
                ("EXP2 result JSON", report["storage"]["exp2_results"]["files"], human_bytes(report["storage"]["exp2_results"]["bytes"])),
            ],
        ),
        "",
        "## Python source health",
        "",
    ]
    if report["source_problems"]:
        lines.append(markdown_table(
            ["Kind", "Path", "Detail"],
            ((row["kind"], row["path"], row.get("detail", "")) for row in report["source_problems"]),
        ))
    else:
        lines.append("No source corruption was detected.")
    lines.extend([
        "",
        "### Official legacy source notes",
        "",
    ])
    if report["legacy_source_notes"]:
        lines.append(markdown_table(
            ["Path", "Note"],
            ((row["path"], row["note"]) for row in report["legacy_source_notes"]),
        ))
    else:
        lines.append("None.")
    lines.extend([
        "",
        "## EXP2 checkpoint coverage",
        "",
        markdown_table(
            ["Strategy", "Model", "Source", "Checkpoints", "Missing"],
            (
                (
                    row["strategy"], row["model"], row["source"], row["checkpoint_count"],
                    len(row["missing_tasks"]),
                )
                for row in report["checkpoint_coverage"]
            ),
        ),
        "",
        "## Runtime EXP2 data health",
        "",
        markdown_table(
            ["Path", "Exists", "Size", "First 4 KiB all NUL"],
            (
                (
                    row["path"], row["exists"], human_bytes(row["bytes"]), row["nul_prefix"]
                )
                for row in report["legacy_exp2_data_health"]
            ),
        ),
        "",
        "## Runtime",
        "",
        f"- Python: `{report['environment']['python']}`",
        f"- Torch/NumPy bridge: `{'OK' if report['environment']['torch_numpy']['ok'] else 'BROKEN'}`",
        f"- GPU: `{report['environment']['gpu']['output'] or 'not detected'}`",
        f"- cuDNN convolution: `{'OK' if report['environment']['gpu_compute']['cudnn_conv']['ok'] else 'BROKEN'}`",
        f"- Native CUDA convolution: `{'OK' if report['environment']['gpu_compute']['native_conv']['ok'] else 'BROKEN'}`",
        f"- cuBLAS matrix multiply: `{'OK' if report['environment']['gpu_compute']['cublas_matmul']['ok'] else 'BROKEN'}`",
        "",
        "Official `EXP1`/`EXP2` remain provenance sources. `gleep_repro` is the deterministic runnable implementation.",
        "",
    ])
    return "\n".join(lines)


def run_audit(workspace: Path, report_dir: Path | None = None) -> dict[str, Any]:
    config = load_config()
    cache_root = exp1_cache_root(workspace)
    exp2_checkpoint_root = PACKAGE_ROOT / "artifacts" / "checkpoints" / "downstream"
    result_root = exp2_results_root(workspace)
    detected = _python_file_health(workspace / "EXP1", workspace) + _python_file_health(
        workspace / "EXP2", workspace
    )
    legacy_source_notes = [
        {**row, "note": KNOWN_OFFICIAL_PLACEHOLDERS[row["path"]]}
        for row in detected
        if row["path"] in KNOWN_OFFICIAL_PLACEHOLDERS
    ]
    source_problems = [
        row for row in detected if row["path"] not in KNOWN_OFFICIAL_PLACEHOLDERS
    ]
    report = {
        "workspace": ".",
        "official_repository": config["official_repository"],
        "official_commit": config["official_commit"],
        "storage": {
            "exp1_cache_by_kind": _exp1_cache_inventory(cache_root),
            "exp2_checkpoints": directory_inventory(exp2_checkpoint_root),
            "exp2_results": directory_inventory(result_root),
        },
        "source_problems": source_problems,
        "legacy_source_notes": legacy_source_notes,
        "exp2_result_counts": _exp2_json_counts(result_root, workspace),
        "checkpoint_coverage": _checkpoint_coverage(exp2_checkpoint_root),
        "legacy_exp2_data_health": _data_health(runtime_data_root(), workspace),
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "torch_numpy": _torch_numpy_probe(),
            "gpu": _gpu_probe(),
            "gpu_compute": _gpu_compute_probe(),
        },
        "artifact_hashes": _artifact_hashes(workspace),
    }
    destination = report_dir or default_report_dir()
    write_json(destination / "audit.json", report)
    write_text(destination / "audit.md", _audit_markdown(report))
    return report
