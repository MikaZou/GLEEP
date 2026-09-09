"""Label-free UPR core. No target labels, class counts or accuracies enter here."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def project_features(features, seed=0, dimension=64):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("features must be a finite matrix")
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    omega = np.random.default_rng(seed).normal(size=(x.shape[1], dimension))
    return x @ (omega / np.sqrt(dimension))


def fourier_features(x, seed=0, dimension=64):
    """Cos/sin pairs ensure each row has exactly unit squared norm."""
    if dimension % 2:
        raise ValueError("Fourier dimension must be even")
    rng = np.random.default_rng(seed)
    n = len(x)
    a = rng.integers(n, size=min(4096, max(n, 1024)))
    b = (a + rng.integers(1, n, size=len(a))) % n
    distances = np.linalg.norm(x[a] - x[b], axis=1)
    positive = distances[distances > 1e-12]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    omega = rng.normal(size=(x.shape[1], dimension // 2)) / sigma
    angles = x @ omega
    v = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
    return v / np.sqrt(dimension // 2), sigma


class RidgeLOO:
    """Fixed-design PRESS operator, without an n by n matrix."""

    def __init__(self, projected, strength=0.01):
        if strength <= 0 or len(projected) < 2:
            raise ValueError("positive ridge strength and n >= 2 required")
        self.f = np.column_stack([projected, np.ones(len(projected))])
        n, d = self.f.shape
        self.alpha = strength * n / d
        gram = self.f.T @ self.f
        self.factor = cho_factor(gram + self.alpha * np.eye(d), lower=True)
        self.t = cho_solve(self.factor, self.f.T).T
        self.leverage = np.einsum("ij,ij->i", self.f, self.t)
        self.denominator = 1 - self.leverage
        if self.denominator.min() <= 1e-10:
            raise ValueError("unstable leave-one-out denominator")
        h2 = np.einsum("ij,ij->i", self.t @ gram, self.t)
        self.r_frobenius_squared = float(np.sum(
            (1 - 2 * self.leverage + h2) / self.denominator**2))

    def residual(self, targets, loo=True):
        residual = targets - self.f @ cho_solve(self.factor, self.f.T @ targets)
        return residual / self.denominator[:, None] if loo else residual


def linear_cka(x, v):
    x = x - x.mean(axis=0)
    v = v - v.mean(axis=0)
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(v.T @ v)
    return float(np.linalg.norm(x.T @ v)**2 / denominator) if denominator > 1e-20 else 0.0


def score_upr(projected_pool, candidate, seed=0):
    """Return score and diagnostics using aligned, preprojected features only."""
    keys = sorted(projected_pool)
    if candidate not in keys or len(keys) < 2:
        raise ValueError("candidate and at least one reference are required")
    n = len(projected_pool[candidate])
    if any(len(x) != n or not np.isfinite(x).all() for x in projected_pool.values()):
        raise ValueError("unaligned or nonfinite pool")
    ridge = RidgeLOO(projected_pool[candidate])
    refs = [key for key in keys if key != candidate]
    blocks, bandwidths = {}, {}
    for i, key in enumerate(keys):
        blocks[key], bandwidths[key] = fourier_features(projected_pool[key], seed + 1009*(i+1))
    v = np.concatenate([blocks[key] for key in refs], axis=1) / np.sqrt(len(refs))
    error = float(np.sum(ridge.residual(v)**2) / n)
    return {"score": -error, "risk_proxy": error, "references": refs,
            "bandwidths": bandwidths, "alpha": ridge.alpha,
            "min_loo_denominator": float(ridge.denominator.min()),
            "r_frobenius_squared": ridge.r_frobenius_squared}, ridge, v, blocks


def rankme(features):
    """RankMe on uncentered original features (singular values, not variances)."""
    x = np.asarray(features, dtype=np.float64)
    singular = np.sqrt(np.maximum(np.linalg.eigvalsh(x.T @ x), 0))
    total = singular.sum()
    if total <= 1e-20:
        return 0.0
    p = singular / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p*np.log(p))))


def structural_diagnostics(ridge, v, labels):
    """Evaluation ONLY: uses true labels; never called by the score function."""
    _, encoded = np.unique(labels, return_inverse=True)
    y = np.eye(int(encoded.max())+1)[encoded]
    n = len(y)
    residual = ridge.residual(y)
    true_risk = float(np.sum(residual**2) / n)
    predictions = y - residual
    accuracy = float(np.mean(predictions.argmax(axis=1) == encoded))
    proxy = float(np.sum(ridge.residual(v)**2)/n)
    # Nonzero eigenvalues of [Y,V] diag(I,-I) [Y,V]^T from a small Gram matrix.
    cross = y.T @ v
    gram = np.block([[y.T @ y, cross], [cross.T, v.T @ v]])
    w, u = np.linalg.eigh(gram)
    keep = w > max(float(w.max()), 1)*1e-12
    b = np.sqrt(w[keep])[:, None] * u[:, keep].T
    signs = np.r_[np.ones(y.shape[1]), -np.ones(v.shape[1])]
    epsilon = float(np.max(np.abs(np.linalg.eigvalsh((b*signs)@b.T))))
    delta = epsilon * ridge.r_frobenius_squared / n
    lower = max(0., 1 - 2*(proxy+delta))
    return {"ridge_loo_risk": true_risk, "ridge_loo_accuracy": accuracy,
            "epsilon_oracle": epsilon, "delta_oracle": delta,
            "accuracy_lower_bound": lower, "bound_nonvacuous": lower > 0,
            "risk_bound_holds": abs(true_risk-proxy) <= delta+1e-8,
            "classification_bound_holds": accuracy+1e-8 >= lower,
            "oracle_brier_accuracy_bound": max(0., 1-2*true_risk)}
