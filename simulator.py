"""
simulator.py
------------
Generates realistic time-series service metrics under normal (baseline) operating conditions.

Emulates the telemetry a production monitoring agent would collect:
  - cpu_usage      : CPU utilization in percent (0-100)
  - memory_mb      : Memory consumption in MB
  - latency_ms     : Request response latency in milliseconds
  - error_rate     : Count of errors per collection interval

All values follow stable distributions with configurable noise levels,
representing a healthy service under typical load.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default baseline parameters (healthy service profile)
# ---------------------------------------------------------------------------
DEFAULT_BASELINE = {
    "cpu_mean": 30.0,       # %
    "cpu_std": 5.0,
    "memory_mean": 512.0,   # MB
    "memory_std": 20.0,
    "latency_mean": 80.0,   # ms
    "latency_std": 10.0,
    "error_lambda": 0.3,    # Poisson rate (errors per interval)
}


def generate_baseline(
    n_steps: int = 500,
    interval_seconds: int = 10,
    baseline: dict | None = None,
    noise_multiplier: float = 1.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate normal service behaviour for `n_steps` collection intervals.

    Parameters
    ----------
    n_steps : int
        Number of collection time steps.
    interval_seconds : int
        Duration of each time step in seconds.
    baseline : dict, optional
        Override default distribution parameters.
    noise_multiplier : float
        Scales the standard deviations to model different noise levels.
        1.0 = normal noise, 2.0 = double noise, 0.5 = low noise.
    random_seed : int
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, cpu_usage, memory_mb, latency_ms, error_rate
    """
    rng = np.random.default_rng(random_seed)
    params = DEFAULT_BASELINE.copy()
    if baseline:
        params.update(baseline)

    timestamps = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=n_steps,
        freq=f"{interval_seconds}s",
    )

    cpu = rng.normal(
        params["cpu_mean"],
        params["cpu_std"] * noise_multiplier,
        n_steps,
    ).clip(0, 100)

    memory = rng.normal(
        params["memory_mean"],
        params["memory_std"] * noise_multiplier,
        n_steps,
    ).clip(0, None)

    latency = rng.normal(
        params["latency_mean"],
        params["latency_std"] * noise_multiplier,
        n_steps,
    ).clip(0, None)

    # Error rate: Poisson-distributed counts (non-negative integers)
    error_rate = rng.poisson(
        params["error_lambda"] * noise_multiplier,
        n_steps,
    ).astype(float)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "cpu_usage": cpu,
            "memory_mb": memory,
            "latency_ms": latency,
            "error_rate": error_rate,
        }
    )
    df.set_index("timestamp", inplace=True)
    return df


def get_baseline_stats(df: pd.DataFrame) -> dict:
    """
    Compute the mean and standard deviation of each metric.
    Used by detectors and health index scorer for normalisation.
    """
    stats = {}
    for col in ["cpu_usage", "memory_mb", "latency_ms", "error_rate"]:
        stats[col] = {"mean": df[col].mean(), "std": max(df[col].std(), 1e-6)}
    return stats
