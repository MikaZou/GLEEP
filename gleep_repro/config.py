from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict[str, Any]:
    with (PACKAGE_ROOT / "configs" / "published.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def workspace_root(override: str | Path | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env_value = os.environ.get("GLEEP_WORKSPACE_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return PACKAGE_ROOT.resolve()


def exp1_metrics_root(workspace: Path) -> Path:
    return workspace / "results" / "published" / "EXP1" / "results_metrics"


def exp1_cache_root(workspace: Path) -> Path:
    return workspace / "artifacts" / "cache" / "exp1" / "group1"


def exp2_results_root(workspace: Path) -> Path:
    return workspace / "results" / "published" / "EXP2" / "result"


def cifar_source_checkpoint(workspace: Path, model_name: str) -> Path:
    packaged = (
        PACKAGE_ROOT
        / "artifacts"
        / "checkpoints"
        / "Pretrain"
        / model_name
        / "40_64_0.001"
        / "best_model.pth"
    )
    return packaged


def published_path(name: str) -> Path:
    return PACKAGE_ROOT / "results" / "published" / name


def default_report_dir() -> Path:
    return PACKAGE_ROOT / "results" / "reproduced"


def default_result_dir() -> Path:
    return PACKAGE_ROOT / "results" / "research" / "recomputed"


def default_exp2_cache_dir() -> Path:
    """Persistent, unpackaged EXP2 activations used for metric development."""
    return PACKAGE_ROOT / "artifacts" / "cache" / "exp2"


def runtime_data_root() -> Path:
    value = os.environ.get("GLEEP_DATA_ROOT")
    return Path(value).expanduser().resolve() if value else PACKAGE_ROOT / "artifacts" / "cache" / "data"
