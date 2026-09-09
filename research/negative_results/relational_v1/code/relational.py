"""Exploratory, label-free relational scores; no model fitting or clustering."""
from __future__ import annotations

import numpy as np


def relational_score(probabilities, reference, *, draws=50000, seed=20260906, bandwidth2=None):
    p = np.asarray(probabilities, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if p.ndim != 2 or r.ndim != 2 or len(p) != len(r) or len(p) < 2:
        raise ValueError("Aligned probability and reference matrices with at least two rows required")
    if not np.isfinite(p).all() or not np.isfinite(r).all() or (p < 0).any():
        raise ValueError("Nonfinite inputs or negative probabilities")
    if not np.allclose(p.sum(1), 1) or draws < 2:
        raise ValueError("Normalized probabilities and at least two draws required")
    norms = np.linalg.norm(r, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Zero reference vectors")
    r = r / norms
    n, c = p.shape
    rng = np.random.default_rng(seed)
    # A separate fixed stream makes the bandwidth independent of candidate predictions.
    if bandwidth2 is None:
        brng = np.random.default_rng(901)
        i = brng.integers(n, size=min(10000, draws))
        j = brng.integers(n - 1, size=len(i)); j += j >= i
        distances = np.maximum(0, 2 - 2 * np.einsum('ij,ij->i', r[i], r[j]))
        bandwidth2 = float(np.median(distances))
    if not np.isfinite(bandwidth2) or bandwidth2 <= 0:
        raise ValueError("Reference bandwidth must be positive")
    s = p.sum(0)
    z = rng.choice(c, size=draws, p=s / n)
    i = np.empty(draws, dtype=np.int64)
    j = np.empty(draws, dtype=np.int64)
    # Cumulative-distribution sampler: O(n C + B log n), not alias sampling.
    order = np.argsort(z)
    counts = np.bincount(z, minlength=c)
    start = 0
    for k, count in enumerate(counts):
        if not count:
            continue
        idx = order[start:start + count]; start += count
        cdf = np.cumsum(p[:, k] / s[k]); cdf[-1] = 1
        i[idx] = np.searchsorted(cdf, rng.random(count))
        j[idx] = np.searchsorted(cdf, rng.random(count))
    u = rng.integers(n - 1, size=draws); u += u >= i
    values = np.empty(draws)
    raw = np.empty(draws)
    for start in range(0, draws, 4096):
        sl = slice(start, start + 4096)
        a = np.exp(-np.maximum(0, 2 - 2 * np.einsum('ij,ij->i', r[i[sl]], r[j[sl]])) / (2 * bandwidth2))
        baseline = np.exp(-np.maximum(0, 2 - 2 * np.einsum('ij,ij->i', r[i[sl]], r[u[sl]])) / (2 * bandwidth2))
        raw[sl] = a
        values[sl] = (i[sl] != j[sl]) * (a - baseline)
    return {"rpt": float(values.mean()), "rpt_se": float(values.std(ddof=1) / np.sqrt(draws)),
            "relation_raw": float(raw.mean()), "bandwidth2": bandwidth2,
            "self_mass": float(np.mean(i == j)), "draws": draws}


def projected_cka(features, reference, dimension=32):
    """A fixed random-projection CKA baseline; not exact full-feature CKA."""
    x, y = np.asarray(features, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    rng = np.random.default_rng(103)
    x = x @ (rng.standard_normal((x.shape[1], dimension)) / np.sqrt(dimension))
    rng = np.random.default_rng(107)
    y = y @ (rng.standard_normal((y.shape[1], dimension)) / np.sqrt(dimension))
    x -= x.mean(0); y -= y.mean(0)
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(np.square(x.T @ y).sum() / denominator) if denominator > 0 else None


def exact_cosine_relational_score(probabilities, reference):
    """Same centered functional with A=(1+cos)/2, exact sufficient statistics."""
    p = np.asarray(probabilities, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    n = len(p)
    if p.ndim != 2 or r.ndim != 2 or len(r) != n or n < 2:
        raise ValueError('Aligned matrices with n>=2 required')
    if not np.isfinite(p).all() or not np.isfinite(r).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError('Invalid probabilities or reference')
    norms = np.linalg.norm(r, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError('Zero reference vectors')
    r = r / norms
    s = p.sum(0); p = p[:, s > 0]; s = s[s > 0]
    aggregate = p.T @ r
    g = float((np.square(aggregate).sum(1)/s).sum())
    diagonal = (np.square(p)/s).sum(1)
    correction = ((1-diagonal)*(r@r.sum(0)-1)).sum()/(n-1)
    return {'rpt_cosine_exact': float((g-diagonal.sum()-correction)/(2*n)),
            'relation_cosine_raw': float(0.5+g/(2*n))}
