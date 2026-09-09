from __future__ import annotations

import contextlib
import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

from .config import (
    PACKAGE_ROOT,
    default_result_dir,
    exp1_cache_root,
    exp1_metrics_root,
    load_config,
)
from .io_utils import portable_path, write_json
from .metrics import gleep_from_logits, leep_from_logits


@contextlib.contextmanager
def _working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _numpy():
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy is required; install environment.yml first") from exc
    return np


def load_cached_logits(cache_root: Path, model: str, dataset: str):
    np = _numpy()
    output_path = cache_root / f"{model}_{dataset}_output.npy"
    label_path = cache_root / f"{model}_{dataset}_label.npy"
    if not output_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"missing cached logits/labels for {model}/{dataset}: {output_path}, {label_path}"
        )
    logits = np.load(output_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    return logits, labels, output_path, label_path


def score_logits(
    logits,
    labels,
    metrics: Iterable[str],
    *,
    seed: int,
    covariance_type: str,
    formula: str,
) -> dict[str, float]:
    np = _numpy()
    metric_names = [value.lower() for value in metrics]
    result: dict[str, float] = {}
    if "leep" in metric_names:
        result["leep"] = leep_from_logits(logits, labels, formula=formula)
    if "gleep" in metric_names:
        num_clusters = len(np.unique(labels))
        result["gleep"], _ = gleep_from_logits(
            logits,
            num_clusters=num_clusters,
            random_state=seed,
            covariance_type=covariance_type,
            formula=formula,
        )
    return result


def _legacy_exp1_root(workspace: Path) -> Path:
    packaged = PACKAGE_ROOT / "legacy_source" / "EXP1"
    if packaged.exists():
        return packaged
    return workspace / "EXP1"


def stream_logits(
    workspace: Path,
    model_name: str,
    dataset: str,
    *,
    batch_size: int,
    device: str,
):
    """Run one legacy EXP1 model/dataset pair without writing feature arrays."""
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required; install environment.yml first") from exc
    legacy_root = _legacy_exp1_root(workspace)
    if not (legacy_root / "get_dataloader.py").exists():
        raise FileNotFoundError(
            f"legacy EXP1 loaders were not found at {legacy_root}; use the packaged legacy_source"
        )
    legacy_text = str(legacy_root)
    inserted = legacy_text not in sys.path
    if inserted:
        sys.path.insert(0, legacy_text)
    try:
        with _working_directory(legacy_root):
            from get_dataloader import get_data, prepare_data
            import models.group1 as model_hub

            dset, data_dir, num_classes, _ = get_data(dataset)
            loaders = prepare_data(dset, data_dir, batch_size, 224, normalisation=True)
            score_loader = loaders[0]  # matches forward_feature.py: trainval_loader = train_loader
            kwargs = {"pretrained": True}
            if model_name in {"inception_v3", "googlenet"}:
                kwargs["aux_logits"] = False
            model = model_hub.__dict__[model_name](**kwargs).to(device).eval()
            logits_parts = []
            label_parts = []
            with torch.no_grad():
                for data, target in score_loader:
                    output = model(data.to(device, non_blocking=True))
                    if hasattr(output, "logits"):
                        output = output.logits
                    elif isinstance(output, (tuple, list)):
                        output = output[0]
                    logits_parts.append(output.detach().cpu())
                    label_parts.append(target.detach().cpu())
            logits = torch.cat(logits_parts).numpy()
            labels = torch.cat(label_parts)
            if dataset == "voc2007" and labels.ndim > 1:
                labels = labels.argmax(dim=1)
            return logits, labels.numpy(), num_classes
    finally:
        if inserted:
            try:
                sys.path.remove(legacy_text)
            except ValueError:
                pass


def run_exp1(
    workspace: Path,
    *,
    source: str = "cached",
    metrics: Iterable[str] = ("gleep", "leep"),
    models: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    seed: int = 0,
    covariance_type: str = "full",
    formula: str = "legacy",
    batch_size: int = 128,
    device: str = "cuda",
    output_dir: Path | None = None,
) -> dict[str, object]:
    config = load_config()["exp1"]
    selected_models = list(models or config["published_models"])
    selected_datasets = list(datasets or config["datasets"])
    metric_names = [value.lower() for value in metrics]
    unknown = set(metric_names).difference({"gleep", "leep"})
    if unknown:
        raise ValueError(f"EXP1 logits-only runner supports GLEEP/LEEP, not {sorted(unknown)}")
    destination = output_dir or default_result_dir() / "exp1"
    result: dict[str, object] = {
        "source": source,
        "formula": formula,
        "random_state": seed,
        "covariance_type": covariance_type,
        "records": [],
    }
    score_files: dict[tuple[str, str], dict[str, float]] = {}
    cache_root = exp1_cache_root(workspace)

    for dataset in selected_datasets:
        for model in selected_models:
            if source == "cached":
                logits, labels, output_path, label_path = load_cached_logits(cache_root, model, dataset)
                provenance = {
                    "logits": portable_path(output_path, workspace),
                    "labels": portable_path(label_path, workspace),
                }
            elif source == "stream":
                logits, labels, _ = stream_logits(
                    workspace, model, dataset, batch_size=batch_size, device=device
                )
                provenance = {"logits": "streamed", "labels": "streamed"}
            else:
                raise ValueError(f"unknown EXP1 source: {source}")
            scores = score_logits(
                logits,
                labels,
                metric_names,
                seed=seed,
                covariance_type=covariance_type,
                formula=formula,
            )
            record = {
                "dataset": dataset,
                "model": model,
                "sample_count": int(len(labels)),
                "source_class_count": int(logits.shape[1]),
                "target_class_count": int(len(_numpy().unique(labels))),
                "scores": scores,
                "provenance": provenance,
            }
            result["records"].append(record)
            for metric, score in scores.items():
                score_files.setdefault((dataset, metric), {})[model] = score
            del logits, labels
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    destination.mkdir(parents=True, exist_ok=True)
    for (dataset, metric), scores in score_files.items():
        write_json(destination / formula / metric / f"{dataset}_metrics.json", scores)
    write_json(destination / formula / "run_manifest.json", result)
    return result


def run_exp1_smoke(workspace: Path, seed: int = 0) -> dict[str, object]:
    """Fast cached-data smoke: exact LEEP plus a small deterministic GLEEP subtask."""
    np = _numpy()
    cache_root = exp1_cache_root(workspace)
    published_root = exp1_metrics_root(workspace) / "group2" / "leep"
    records = []
    for model in ("mobilenet_v2", "resnet34"):
        logits, labels, output_path, label_path = load_cached_logits(cache_root, model, "flowers")
        exact_leep = leep_from_logits(logits, labels, formula="legacy")
        historical = None
        historical_path = published_root / "flowers_metrics.json"
        if historical_path.exists():
            from .io_utils import load_json
            historical = float(load_json(historical_path)[model])

        # Use a real, deterministic 10-class slice and 32 source-logit dimensions.
        chosen_labels = np.unique(labels)[:10]
        indices = np.flatnonzero(np.isin(labels, chosen_labels))
        subset_logits = np.asarray(logits[indices, :32])
        subset_labels = np.asarray(labels[indices])
        first, _ = gleep_from_logits(
            subset_logits,
            num_clusters=len(chosen_labels),
            random_state=seed,
            covariance_type="diag",
            formula="legacy",
        )
        second, _ = gleep_from_logits(
            subset_logits,
            num_clusters=len(chosen_labels),
            random_state=seed,
            covariance_type="diag",
            formula="legacy",
        )
        records.append({
            "dataset": "flowers",
            "model": model,
            "full_sample_count": int(len(labels)),
            "leep": exact_leep,
            "historical_leep": historical,
            "leep_absolute_delta": abs(exact_leep - historical) if historical is not None else None,
            "gleep_smoke_score": first,
            "gleep_repeat_score": second,
            "gleep_deterministic": first == second,
            "gleep_subset_samples": int(len(indices)),
            "gleep_subset_target_classes": len(chosen_labels),
            "gleep_subset_source_dimensions": 32,
            "gleep_covariance_type": "diag",
            "logits_path": portable_path(output_path, workspace),
            "labels_path": portable_path(label_path, workspace),
        })
    return {"seed": seed, "records": records}
