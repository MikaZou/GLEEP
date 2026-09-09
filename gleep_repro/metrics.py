from __future__ import annotations

from typing import Literal


def _numpy():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "NumPy is required for score recomputation. Create the environment from environment.yml."
        ) from exc
    return np


def softmax(logits):
    np = _numpy()
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"logits must be two-dimensional, got {values.shape}")
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def leep_from_probabilities(
    probabilities,
    labels,
    formula: Literal["legacy", "canonical"] = "legacy",
) -> float:
    """Compute the historical mean-probability LEEP or canonical log-LEEP."""
    np = _numpy()
    probs = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probs.ndim != 2 or len(probs) != len(target):
        raise ValueError("probabilities and labels must contain the same number of samples")
    unique = np.unique(target)
    remap = {int(value): index for index, value in enumerate(unique.tolist())}
    encoded = np.asarray([remap[int(value)] for value in target], dtype=np.int64)
    joint = np.zeros((len(unique), probs.shape[1]), dtype=np.float64)
    np.add.at(joint, encoded, probs)
    joint /= len(encoded)
    marginal = joint.sum(axis=0)
    conditional = np.divide(
        joint,
        marginal,
        out=np.zeros_like(joint),
        where=marginal > 0,
    )
    prediction = probs @ conditional.T
    selected = prediction[np.arange(len(encoded)), encoded]
    if formula == "legacy":
        return float(selected.mean())
    if formula == "canonical":
        return float(np.log(np.clip(selected, np.finfo(np.float64).tiny, None)).mean())
    raise ValueError(f"unknown LEEP formula: {formula}")


def leep_from_logits(logits, labels, formula: Literal["legacy", "canonical"] = "legacy") -> float:
    return leep_from_probabilities(softmax(logits), labels, formula=formula)


def gleep_from_logits(
    clustering_logits,
    prediction_logits=None,
    *,
    num_clusters: int,
    random_state: int = 0,
    covariance_type: str = "full",
    formula: Literal["legacy", "canonical"] = "legacy",
) -> tuple[float, object]:
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "scikit-learn is required for GLEEP. Create the environment from environment.yml."
        ) from exc
    np = _numpy()
    cluster_values = np.asarray(clustering_logits)
    prediction_values = cluster_values if prediction_logits is None else np.asarray(prediction_logits)
    if len(cluster_values) != len(prediction_values):
        raise ValueError("clustering and prediction logits must contain the same samples")
    if not 1 <= num_clusters <= len(cluster_values):
        raise ValueError("num_clusters must be between 1 and the sample count")
    model = GaussianMixture(
        n_components=num_clusters,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    pseudo_labels = model.fit_predict(cluster_values)
    score = leep_from_logits(prediction_values, pseudo_labels, formula=formula)
    return score, pseudo_labels
