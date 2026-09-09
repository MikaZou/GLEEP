from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable

from .config import (
    PACKAGE_ROOT,
    cifar_source_checkpoint,
    default_exp2_cache_dir,
    default_result_dir,
    exp2_results_root,
    load_config,
)
from .io_utils import load_json, portable_path, sha256, write_json, write_numpy
from .metrics import gleep_from_logits, leep_from_logits
from .models import build_cifar_resnet, build_imagenet_resnet
from .stats import common_ordered_values, kendall_tau_b, pearsonr


def parse_tasks(specification: str | None) -> list[int]:
    if not specification:
        return list(range(2, 101))
    values: set[int] = set()
    for part in specification.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_text, end_text = part.split(":", 1)
            start, end = int(start_text), int(end_text)
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    result = sorted(values)
    if not result or result[0] < 2 or result[-1] > 100:
        raise ValueError("EXP2 tasks must be between 2 and 100 classes")
    return result


def selected_classes(
    num_classes: int,
    seed: int = 42,
    semantics: str = "corrected",
) -> list[int]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required; install environment.yml first") from exc
    generator = torch.Generator().manual_seed(seed)
    if semantics == "published":
        # The official EXP2 scripts draw and discard one permutation before
        # prepare_data() draws the permutation that actually filters CIFAR100.
        # Reproduce that otherwise easy-to-miss historical behavior without
        # mutating PyTorch's process-global RNG state.
        torch.randperm(100, generator=generator)
    elif semantics != "corrected":
        raise ValueError(f"unknown EXP2 semantics: {semantics}")
    return torch.randperm(100, generator=generator)[:num_classes].sort()[0].tolist()


def _torch_load(path: Path, device: str):
    import torch

    # Always deserialize checkpoints on CPU first.  Some historical torchvision
    # checkpoints use legacy storages that cannot be remapped directly to CUDA
    # by newer PyTorch releases.  Callers move the fully constructed model to
    # ``device`` after loading the state dict, which is both safer and uses less
    # transient GPU memory.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older PyTorch
        return torch.load(path, map_location="cpu")


def _source_model(
    workspace: Path,
    model_name: str,
    source: str,
    device: str,
    semantics: str,
):
    if source == "CIFAR10":
        model = build_cifar_resnet(model_name, num_classes=100, semantics=semantics)
        checkpoint = cifar_source_checkpoint(workspace, model_name)
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing CIFAR10 source checkpoint: {checkpoint}")
        model.load_state_dict(_torch_load(checkpoint, device))
        return model.to(device), checkpoint
    if source == "ImageNet":
        local_names = {
            "ResNet18": "resnet18-f37072fd.pth",
            "ResNet34": "resnet34-333f7ec4.pth",
        }
        local_candidates = [
            workspace / "EXP1" / "models" / "group1" / "checkpoints" / local_names[model_name],
            Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / local_names[model_name],
        ]
        local_checkpoint = next((path for path in local_candidates if path.exists()), None)
        if local_checkpoint is not None:
            model = build_imagenet_resnet(model_name, pretrained=False)
            model.load_state_dict(_torch_load(local_checkpoint, device))
            return model.to(device), local_checkpoint
        return build_imagenet_resnet(model_name, pretrained=True).to(device), None
    raise ValueError(f"unsupported source dataset: {source}")


def _transforms(train: bool):
    from torchvision import transforms

    operations = []
    if train:
        operations.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transforms.Compose(operations)


def _filtered_dataset(
    data_root: Path,
    num_classes: int,
    *,
    train: bool,
    seed: int,
    semantics: str,
    download: bool = True,
):
    import torch
    from torchvision.datasets import CIFAR100

    base = CIFAR100(root=str(data_root), train=train, download=download, transform=_transforms(train))
    chosen = selected_classes(num_classes, seed, semantics)
    remap = {original: index for index, original in enumerate(chosen)}
    indices = [index for index, label in enumerate(base.targets) if label in remap]

    class FilteredDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(indices)

        def __getitem__(self, item):
            image, label = base[indices[item]]
            return image, remap[label]

    return FilteredDataset(), chosen


def _source_outputs(model, loader, device: str, *, max_samples: int | None = None):
    import torch

    feature_parts = []
    output_parts = []
    label_parts = []

    def capture(_module, inputs, _output):
        feature_parts.append(inputs[0].detach().cpu())

    handle = model.fc.register_forward_hook(capture)
    model.eval()
    seen = 0
    try:
        with torch.no_grad():
            for images, labels in loader:
                if max_samples is not None:
                    remaining = max_samples - seen
                    if remaining <= 0:
                        break
                    images = images[:remaining]
                    labels = labels[:remaining]
                output = model(images.to(device, non_blocking=True))
                output_parts.append(output.detach().cpu())
                label_parts.append(labels.detach().cpu())
                seen += len(labels)
    finally:
        handle.remove()
    return (
        torch.cat(feature_parts).numpy(),
        torch.cat(output_parts).numpy(),
        torch.cat(label_parts).numpy(),
    )


def _legacy_prediction_logits(workspace: Path, model_name: str, features, device: str):
    np = __import__("numpy")
    checkpoint = cifar_source_checkpoint(workspace, model_name)
    state = _torch_load(checkpoint, device="cpu")
    weight = state["fc.weight"].detach().cpu().numpy()
    bias = state["fc.bias"].detach().cpu().numpy()
    return np.asarray(features) @ weight.T + bias


def _cache_directory(cache_root: Path, semantics: str, model_name: str, source: str) -> Path:
    return cache_root / semantics / model_name / source


def _array_artifact(path: Path) -> dict[str, object]:
    import numpy as np

    values = np.load(path, mmap_mode="r", allow_pickle=False)
    finite = True if values.dtype.kind not in "fc" else bool(np.isfinite(values).all())
    return {
        "path": path.name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "finite": finite,
    }


def run_exp2_cache(
    workspace: Path,
    *,
    models: Iterable[str],
    sources: Iterable[str],
    data_root: Path,
    device: str,
    seed: int = 42,
    semantics: str = "published",
    batch_size: int = 128,
    cache_dir: Path | None = None,
) -> dict[str, object]:
    """Run each source model once on the complete CIFAR100 test set and persist its outputs."""
    import numpy as np
    import torch

    destination = cache_dir or default_exp2_cache_dir()
    records: list[dict[str, object]] = []
    task_classes = {
        f"{task:03d}": selected_classes(task, seed=seed, semantics=semantics)
        for task in range(2, 101)
    }
    for model_name in models:
        for source in sources:
            print(f"caching EXP2 outputs: model={model_name} source={source}", flush=True)
            dataset, classes = _filtered_dataset(
                data_root,
                100,
                train=False,
                seed=seed,
                semantics=semantics,
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=device.startswith("cuda"),
            )
            model, checkpoint = _source_model(workspace, model_name, source, device, semantics)
            features, clustering_logits, labels = _source_outputs(model, loader, device)
            if semantics == "published" and source == "ImageNet":
                prediction_logits = _legacy_prediction_logits(
                    workspace, model_name, features, device
                )
                prediction_rule = "cifar_checkpoint_head_on_imagenet_features"
            else:
                prediction_logits = clustering_logits
                prediction_rule = "source_model_logits"

            output_root = _cache_directory(destination, semantics, model_name, source)
            arrays = {
                "labels": np.ascontiguousarray(labels, dtype=np.int64),
                "features": np.ascontiguousarray(features, dtype=np.float32),
                "clustering_logits": np.ascontiguousarray(clustering_logits, dtype=np.float32),
                "prediction_logits": np.ascontiguousarray(prediction_logits, dtype=np.float32),
            }
            artifact_paths = {}
            for name, values in arrays.items():
                path = output_root / f"{name}.npy"
                write_numpy(path, values)
                artifact_paths[name] = path

            checkpoint_text = str(checkpoint) if checkpoint else "torchvision:IMAGENET1K_V1"
            checkpoint_digest = sha256(checkpoint) if checkpoint and checkpoint.exists() else None
            manifest = {
                "format_version": 1,
                "model": model_name,
                "source": source,
                "target_dataset": "CIFAR100",
                "split": "test",
                "sample_count": int(len(labels)),
                "source_checkpoint": checkpoint_text,
                "source_checkpoint_sha256": checkpoint_digest,
                "semantics": semantics,
                "selection_seed": seed,
                "all_classes": classes,
                "task_classes": task_classes,
                "prediction_rule": prediction_rule,
                "preprocessing": {
                    "to_tensor": True,
                    "normalization_mean": [0.4914, 0.4822, 0.4465],
                    "normalization_std": [0.2023, 0.1994, 0.2010],
                    "random_augmentation": False,
                },
                "arrays": {
                    name: _array_artifact(path) for name, path in artifact_paths.items()
                },
            }
            write_json(output_root / "manifest.json", manifest)
            records.append({
                "model": model_name,
                "source": source,
                "cache_directory": portable_path(output_root, workspace),
                "manifest": portable_path(output_root / "manifest.json", workspace),
                "sample_count": int(len(labels)),
                "arrays": manifest["arrays"],
            })
            del (
                model,
                loader,
                dataset,
                features,
                clustering_logits,
                prediction_logits,
                labels,
                arrays,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = {
        "format_version": 1,
        "semantics": semantics,
        "selection_seed": seed,
        "records": records,
    }
    write_json(destination / semantics / "cache_manifest.json", report)
    return report


def _cached_task_arrays(
    cache_dir: Path,
    *,
    model_name: str,
    source: str,
    num_classes: int,
    seed: int,
    semantics: str,
):
    import numpy as np

    root = _cache_directory(cache_dir, semantics, model_name, source)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing EXP2 logits cache manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    expected = {"model": model_name, "source": source, "semantics": semantics}
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"EXP2 cache {manifest_path} has {field}={manifest.get(field)!r}, expected {value!r}"
            )
    labels = np.load(
        root / manifest["arrays"]["labels"]["path"], mmap_mode="r", allow_pickle=False
    )
    clustering = np.load(
        root / manifest["arrays"]["clustering_logits"]["path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    prediction = np.load(
        root / manifest["arrays"]["prediction_logits"]["path"],
        mmap_mode="r",
        allow_pickle=False,
    )
    if not (len(labels) == len(clustering) == len(prediction) == manifest["sample_count"]):
        raise ValueError(f"EXP2 cache arrays have inconsistent sample counts: {manifest_path}")
    classes = selected_classes(num_classes, seed=seed, semantics=semantics)
    indices = np.flatnonzero(np.isin(labels, classes))
    remapped_labels = np.searchsorted(np.asarray(classes, dtype=np.int64), labels[indices])
    return (
        np.asarray(clustering[indices]),
        np.asarray(prediction[indices]),
        np.asarray(remapped_labels, dtype=np.int64),
        classes,
        manifest,
    )


def score_exp2_cached_task(
    *,
    cache_dir: Path,
    model_name: str,
    source: str,
    num_classes: int,
    metrics: Iterable[str],
    seed: int,
    random_state: int,
    covariance_type: str,
    semantics: str,
    formula: str,
) -> dict[str, object]:
    clustering_logits, prediction_logits, labels, classes, manifest = _cached_task_arrays(
        cache_dir,
        model_name=model_name,
        source=source,
        num_classes=num_classes,
        seed=seed,
        semantics=semantics,
    )
    metric_names = {value.upper() for value in metrics}
    scores: dict[str, float] = {}
    if "LEEP" in metric_names:
        scores["LEEP"] = leep_from_logits(prediction_logits, labels, formula=formula)
    if "GLEEP" in metric_names:
        scores["GLEEP"], _ = gleep_from_logits(
            clustering_logits,
            prediction_logits,
            num_clusters=num_classes,
            random_state=random_state,
            covariance_type=covariance_type,
            formula=formula,
        )
    return {
        "task": f"{num_classes:03d}",
        "selected_classes": classes,
        "sample_count": int(len(labels)),
        "clustering_dimensions": int(clustering_logits.shape[1]),
        "prediction_dimensions": int(prediction_logits.shape[1]),
        "source_checkpoint": (
            f"<external-artifact>/{Path(str(manifest['source_checkpoint'])).name}"
        ),
        "input_cache": portable_path(
            _cache_directory(cache_dir, semantics, model_name, source), PACKAGE_ROOT
        ),
        "scores": scores,
    }


def score_exp2_task(
    workspace: Path,
    *,
    model_name: str,
    source: str,
    num_classes: int,
    metrics: Iterable[str],
    data_root: Path,
    device: str,
    seed: int,
    random_state: int,
    covariance_type: str,
    semantics: str,
    formula: str,
    batch_size: int,
    max_samples: int | None = None,
    max_logit_dimensions: int | None = None,
) -> dict[str, object]:
    import numpy as np
    import torch

    dataset, classes = _filtered_dataset(
        data_root,
        num_classes,
        train=False,
        seed=seed,
        semantics=semantics,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.startswith("cuda")
    )
    model, checkpoint = _source_model(workspace, model_name, source, device, semantics)
    features, clustering_logits, labels = _source_outputs(
        model, loader, device, max_samples=max_samples
    )
    if semantics == "published" and source == "ImageNet":
        # Official code clusters ImageNet outputs, then applies the CIFAR checkpoint head to ImageNet features.
        prediction_logits = _legacy_prediction_logits(workspace, model_name, features, device)
    else:
        prediction_logits = clustering_logits
    if max_logit_dimensions is not None:
        clustering_logits = clustering_logits[:, :max_logit_dimensions]
        prediction_logits = prediction_logits[:, :max_logit_dimensions]

    metric_names = {value.upper() for value in metrics}
    scores: dict[str, float] = {}
    if "LEEP" in metric_names:
        scores["LEEP"] = leep_from_logits(prediction_logits, labels, formula=formula)
    if "GLEEP" in metric_names:
        scores["GLEEP"], _ = gleep_from_logits(
            clustering_logits,
            prediction_logits,
            num_clusters=num_classes,
            random_state=random_state,
            covariance_type=covariance_type,
            formula=formula,
        )
    result = {
        "task": f"{num_classes:03d}",
        "selected_classes": classes,
        "sample_count": int(len(labels)),
        "clustering_dimensions": int(clustering_logits.shape[1]),
        "prediction_dimensions": int(prediction_logits.shape[1]),
        "source_checkpoint": str(checkpoint) if checkpoint else "torchvision:IMAGENET1K_V1",
        "scores": scores,
    }
    del model, features, clustering_logits, prediction_logits, labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _historical_accuracy_path(root: Path, strategy: str, model: str, source: str, metric: str) -> Path:
    return root / strategy / model / source / "test_ACC" / f"{metric}_ACC.json"


def run_exp2_scores(
    workspace: Path,
    *,
    models: Iterable[str],
    sources: Iterable[str],
    tasks: Iterable[int],
    metrics: Iterable[str],
    strategies: Iterable[str],
    data_root: Path,
    device: str,
    seed: int = 42,
    random_state: int = 0,
    covariance_type: str = "full",
    semantics: str = "published",
    formula: str = "legacy",
    batch_size: int = 128,
    output_dir: Path | None = None,
    max_samples: int | None = None,
    max_logit_dimensions: int | None = None,
    logits_cache_dir: Path | None = None,
) -> dict[str, object]:
    destination = output_dir or default_result_dir() / "exp2" / "score"
    historical_root = exp2_results_root(workspace)
    records = []
    correlations = []
    metric_names = [value.upper() for value in metrics]
    for model in models:
        for source in sources:
            score_maps = {metric: {} for metric in metric_names}
            for task in tasks:
                print(
                    f"scoring EXP2 task: model={model} source={source} classes={task}",
                    flush=True,
                )
                if logits_cache_dir is None:
                    record = score_exp2_task(
                        workspace,
                        model_name=model,
                        source=source,
                        num_classes=task,
                        metrics=metric_names,
                        data_root=data_root,
                        device=device,
                        seed=seed,
                        random_state=random_state,
                        covariance_type=covariance_type,
                        semantics=semantics,
                        formula=formula,
                        batch_size=batch_size,
                        max_samples=max_samples,
                        max_logit_dimensions=max_logit_dimensions,
                    )
                else:
                    if max_samples is not None or max_logit_dimensions is not None:
                        raise ValueError(
                            "max_samples/max_logit_dimensions are smoke-only forward options"
                        )
                    record = score_exp2_cached_task(
                        cache_dir=logits_cache_dir,
                        model_name=model,
                        source=source,
                        num_classes=task,
                        metrics=metric_names,
                        seed=seed,
                        random_state=random_state,
                        covariance_type=covariance_type,
                        semantics=semantics,
                        formula=formula,
                    )
                records.append({"model": model, "source": source, **record})
                for metric, score in record["scores"].items():
                    score_maps[metric][record["task"]] = score
                    write_json(
                        destination / semantics / model / source / f"{metric}.json",
                        score_maps[metric],
                    )
                write_json(
                    destination / semantics / "progress_manifest.json",
                    {
                        "semantics": semantics,
                        "formula": formula,
                        "seed": seed,
                        "random_state": random_state,
                        "covariance_type": covariance_type,
                        "complete": False,
                        "completed_records": len(records),
                        "records": records,
                    },
                )
            for metric, scores in score_maps.items():
                score_path = destination / semantics / model / source / f"{metric}.json"
                write_json(score_path, scores)
                for strategy in strategies:
                    accuracy_path = _historical_accuracy_path(
                        historical_root, strategy, model, source, metric
                    )
                    if not accuracy_path.exists():
                        continue
                    accuracies = load_json(accuracy_path)
                    keys, score_values, accuracy_values = common_ordered_values(scores, accuracies)
                    if len(keys) >= 2:
                        correlations.append({
                            "strategy": strategy,
                            "model": model,
                            "source": source,
                            "metric": metric,
                            "task_count": len(keys),
                            "pearson": pearsonr(score_values, accuracy_values),
                            "kendall": kendall_tau_b(score_values, accuracy_values),
                            "accuracy_source": portable_path(accuracy_path, workspace),
                        })
    report = {
        "semantics": semantics,
        "formula": formula,
        "seed": seed,
        "random_state": random_state,
        "covariance_type": covariance_type,
        "records": records,
        "correlations": correlations,
    }
    write_json(destination / semantics / "run_manifest.json", report)
    write_json(
        destination / semantics / "progress_manifest.json",
        {
            **report,
            "complete": True,
            "completed_records": len(records),
        },
    )
    return report


def _target_model(workspace: Path, model_name: str, source: str, classes: int, strategy: str, device: str, semantics: str):
    import torch.nn as nn

    model, _ = _source_model(workspace, model_name, source, device, semantics)
    model.fc = nn.Linear(model.fc.in_features, classes).to(device)
    if strategy == "Retrain":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
    return model


def _evaluate(model, loader, device: str) -> float:
    import torch

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            output = model(images.to(device, non_blocking=True))
            prediction = output.argmax(dim=1).cpu()
            correct += int((prediction == labels).sum())
            total += len(labels)
    return 100.0 * correct / total


def run_exp2_training(
    workspace: Path,
    *,
    models: Iterable[str],
    sources: Iterable[str],
    strategies: Iterable[str],
    tasks: Iterable[int],
    data_root: Path,
    device: str,
    seed: int = 42,
    semantics: str = "published",
    epochs: int = 30,
    batch_size_train: int = 64,
    batch_size_test: int = 128,
    learning_rate: float = 0.001,
    retain_checkpoints: str = "none",
    output_dir: Path | None = None,
) -> dict[str, object]:
    import torch

    destination = output_dir or default_result_dir() / "exp2" / "train"
    records = []
    for strategy in strategies:
        for model_name in models:
            for source in sources:
                accuracy_map = {}
                for task in tasks:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    train_set, classes = _filtered_dataset(
                        data_root,
                        task,
                        train=True,
                        seed=seed,
                        semantics=semantics,
                    )
                    test_set, _ = _filtered_dataset(
                        data_root,
                        task,
                        train=False,
                        seed=seed,
                        semantics=semantics,
                    )
                    train_loader = torch.utils.data.DataLoader(
                        train_set, batch_size=batch_size_train, shuffle=True, num_workers=2
                    )
                    test_loader = torch.utils.data.DataLoader(
                        test_set, batch_size=batch_size_test, shuffle=False, num_workers=2
                    )
                    model = _target_model(
                        workspace, model_name, source, task, strategy, device, semantics
                    )
                    optimizer = torch.optim.SGD(
                        (p for p in model.parameters() if p.requires_grad),
                        lr=learning_rate,
                        momentum=0.9,
                        weight_decay=5e-4,
                    )
                    criterion = torch.nn.CrossEntropyLoss()
                    best_accuracy = 0.0
                    best_state = None
                    for _epoch in range(epochs):
                        model.train()
                        for images, labels in train_loader:
                            optimizer.zero_grad(set_to_none=True)
                            loss = criterion(model(images.to(device)), labels.to(device))
                            loss.backward()
                            optimizer.step()
                        accuracy = _evaluate(model, test_loader, device)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            if retain_checkpoints == "best":
                                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    task_key = f"{task:03d}"
                    accuracy_map[task_key] = best_accuracy
                    record = {
                        "strategy": strategy,
                        "implementation": "linear_probe" if strategy == "Retrain" else "full_finetune",
                        "model": model_name,
                        "source": source,
                        "task": task_key,
                        "selected_classes": classes,
                        "best_test_accuracy": best_accuracy,
                    }
                    if best_state is not None:
                        checkpoint_path = destination / semantics / strategy / model_name / source / f"C{task}" / "best_model.pth"
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(best_state, checkpoint_path)
                        record["checkpoint"] = portable_path(checkpoint_path, workspace)
                    records.append(record)
                    write_json(
                        destination / semantics / strategy / model_name / source / "accuracy.json",
                        accuracy_map,
                    )
                    del model, optimizer, train_loader, test_loader, train_set, test_set, best_state
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    report = {
        "semantics": semantics,
        "epochs": epochs,
        "retain_checkpoints": retain_checkpoints,
        "records": records,
    }
    write_json(destination / semantics / "run_manifest.json", report)
    return report


def run_exp2_smoke(workspace: Path, data_root: Path, device: str, seed: int = 42) -> dict[str, object]:
    records = []
    task_map = {"CIFAR10": [2, 100], "ImageNet": [2, 10]}
    for model in ("ResNet18", "ResNet34"):
        for source, tasks in task_map.items():
            result = run_exp2_scores(
                workspace,
                models=[model],
                sources=[source],
                tasks=tasks,
                metrics=["LEEP", "GLEEP"],
                strategies=["Finetune", "Retrain"],
                data_root=data_root,
                device=device,
                seed=seed,
                random_state=0,
                covariance_type="full",
                semantics="published",
                formula="legacy",
                batch_size=128,
                output_dir=default_result_dir() / "smoke" / "exp2",
                max_samples=512,
                max_logit_dimensions=32,
            )
            records.extend(result["records"])
    return {
        "note": "Smoke mode limits each task to 512 samples and 32 logit dimensions; it is not a paper-number run.",
        "records": records,
    }


def run_exp2_synthetic_smoke(workspace: Path, device: str, seed: int = 42) -> dict[str, object]:
    """Exercise source checkpoints, GPU forward, dynamic dimensions and GMM without a dataset download."""
    import numpy as np
    import torch

    records = []
    task_map = {"CIFAR10": [2, 100], "ImageNet": [2, 10]}
    for model_name in ("ResNet18", "ResNet34"):
        for source, tasks in task_map.items():
            print(f"synthetic EXP2 smoke forward: model={model_name} source={source}", flush=True)
            source_model, checkpoint = _source_model(
                workspace, model_name, source, device, "published"
            )
            one_image = torch.randn(1, 3, 32, 32)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(one_image, torch.zeros(1, dtype=torch.long)),
                batch_size=1,
                shuffle=False,
            )
            feature_probe, output_probe, _ = _source_outputs(source_model, loader, device)
            actual_feature_dimensions = int(feature_probe.shape[1])
            actual_output_dimensions = int(output_probe.shape[1])
            del source_model, feature_probe, output_probe, one_image, loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for task in tasks:
                sample_count = max(2 * task, 128)
                rng = np.random.default_rng(seed + task + (1000 if source == "ImageNet" else 0))
                clustering_dimensions = min(actual_output_dimensions, 32)
                prediction_dimensions = min(100 if source == "ImageNet" else actual_output_dimensions, 32)
                clustering_logits = rng.normal(size=(sample_count, clustering_dimensions))
                prediction_logits = rng.normal(size=(sample_count, prediction_dimensions))
                observed_labels = np.arange(sample_count) % task
                leep = leep_from_logits(prediction_logits, observed_labels, formula="legacy")
                first, _ = gleep_from_logits(
                    clustering_logits,
                    prediction_logits,
                    num_clusters=task,
                    random_state=0,
                    covariance_type="full",
                    formula="legacy",
                )
                second, _ = gleep_from_logits(
                    clustering_logits,
                    prediction_logits,
                    num_clusters=task,
                    random_state=0,
                    covariance_type="full",
                    formula="legacy",
                )
                records.append({
                    "model": model_name,
                    "source": source,
                    "task": f"{task:03d}",
                    "sample_count": sample_count,
                    "clustering_dimensions": clustering_dimensions,
                    "prediction_dimensions": prediction_dimensions,
                    "actual_feature_dimensions": actual_feature_dimensions,
                    "actual_output_dimensions": actual_output_dimensions,
                    "source_checkpoint": str(checkpoint) if checkpoint else "torchvision:IMAGENET1K_V1",
                    "scores": {"LEEP": leep, "GLEEP": first},
                    "gleep_deterministic": first == second,
                })
                del clustering_logits, prediction_logits, observed_labels
    return {
        "note": f"Synthetic smoke validates model/checkpoint/{device} forward wiring only; it is not an experimental result.",
        "records": records,
    }
