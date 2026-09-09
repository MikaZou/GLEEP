from __future__ import annotations

import math
from collections.abc import Sequence


def _validate_pair(x: Sequence[float], y: Sequence[float]) -> None:
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} != {len(y)}")
    if len(x) < 2:
        raise ValueError("at least two paired observations are required")
    if any(not math.isfinite(float(v)) for v in (*x, *y)):
        raise ValueError("correlation inputs must be finite")


def pearsonr(x: Sequence[float], y: Sequence[float]) -> float:
    _validate_pair(x, y)
    x_values = [float(v) for v in x]
    y_values = [float(v) for v in y]
    x_mean = math.fsum(x_values) / len(x_values)
    y_mean = math.fsum(y_values) / len(y_values)
    dx = [v - x_mean for v in x_values]
    dy = [v - y_mean for v in y_values]
    numerator = math.fsum(a * b for a, b in zip(dx, dy))
    denominator = math.sqrt(math.fsum(a * a for a in dx) * math.fsum(b * b for b in dy))
    if denominator == 0.0:
        raise ValueError("Pearson correlation is undefined for a constant input")
    return numerator / denominator


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Ordinary Kendall tau-b, including ties, with no SciPy dependency."""
    _validate_pair(x, y)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            delta_x = float(x[i]) - float(x[j])
            delta_y = float(y[i]) - float(y[j])
            if delta_x == 0.0 and delta_y == 0.0:
                continue
            if delta_x == 0.0:
                ties_x += 1
            elif delta_y == 0.0:
                ties_y += 1
            elif delta_x * delta_y > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x) *
        (concordant + discordant + ties_y)
    )
    if denominator == 0.0:
        raise ValueError("Kendall correlation is undefined for these inputs")
    return (concordant - discordant) / denominator


def common_ordered_values(
    first: dict[str, float],
    second: dict[str, float],
    preferred_order: Sequence[str] | None = None,
) -> tuple[list[str], list[float], list[float]]:
    common = set(first).intersection(second)
    if preferred_order is None:
        keys = sorted(common)
    else:
        keys = [key for key in preferred_order if key in common]
    return keys, [float(first[k]) for k in keys], [float(second[k]) for k in keys]

