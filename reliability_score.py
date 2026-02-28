"""
reliability_score.py
--------------------
Computes a real-time system reliability score ranging from 0 to 100.

Score interpretation:
  100 = perfectly healthy system (all metrics at baseline)
    0 = complete failure (all metrics severely deviated)

Formula (weighted sum of normalised deviations):

    deviation_i = |metric_i - baseline_mean_i| / baseline_std_i

    weighted_score = sum(w_i * deviation_i) for each metric i

    raw_reliability = 100 * exp(-k * weighted_score)
    reliability     = clip(raw_reliability, 0, 100)

Where:
  w_i are the per-metric weights (sum = 1)
  k   is a sensitivity constant controlling how fast the score drops

This gives a smooth, exponential decay relative to baseline deviations —
easier to interpret than a linear mapping and avoids hard cliffs.
"""

import numpy as np
import pandas as pd


# Per-metric importance weights (must sum to 1.0)
METRIC_WEIGHTS = {
    "cpu_usage": 0.20,
    "memory_mb": 0.30,
    "latency_ms": 0.25,
    "error_rate": 0.25,
}

# Sensitivity: higher k → score drops faster with deviations
SENSITIVITY = 0.5


def compute_reliability_scores(
    df: pd.DataFrame,
    baseline_stats: dict,
    sensitivity: float = SENSITIVITY,
    weights: dict | None = None,
) -> np.ndarray:
    """
    Compute per-step reliability score.

    Parameters
    ----------
    df             : Full (injected) metrics DataFrame
    baseline_stats : Dict with {"metric": {"mean": float, "std": float}}
                     from simulator.get_baseline_stats()
    sensitivity    : k constant in exp(-k * score)
    weights        : override default METRIC_WEIGHTS

    Returns
    -------
    scores : np.ndarray of shape (n_steps,), values in [0, 100]
    """
    w = {**METRIC_WEIGHTS, **(weights or {})}
    n = len(df)
    weighted_dev = np.zeros(n)

    for col, weight in w.items():
        mean = baseline_stats[col]["mean"]
        std = baseline_stats[col]["std"]
        deviation = np.abs(df[col].values - mean) / std
        weighted_dev += weight * deviation

    # Exponential decay: 100 when deviation=0, falls toward 0 with deviation
    scores = 100.0 * np.exp(-sensitivity * weighted_dev)
    return np.clip(scores, 0.0, 100.0)


def reliability_status(score: float) -> str:
    """
    Map a scalar score to a human-readable status label.

      >= 80 : Healthy
      >= 60 : Degraded
      >= 40 : Warning
      >= 20 : Critical
       < 20 : Failure
    """
    if score >= 80:
        return "Healthy"
    elif score >= 60:
        return "Degraded"
    elif score >= 40:
        return "Warning"
    elif score >= 20:
        return "Critical"
    else:
        return "Failure"
