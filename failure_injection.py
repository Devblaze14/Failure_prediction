"""
failure_injection.py
--------------------
Models controlled degradation scenarios against a baseline metrics DataFrame.

Each degradation scenario:
  - Has a configurable start_step and severity
  - Records the ground-truth SLA breach timestamp for evaluation
  - Returns the modified DataFrame and the exact breach index

Degradation scenarios implemented:
  1. memory_leak          - monotonic memory growth (gradual)
  2. cpu_spike_burst      - sudden repeated CPU spikes
  3. latency_trend        - linearly increasing latency
  4. error_explosion      - sudden sharp rise in error rate
  5. combined_degradation - all metrics degrade simultaneously
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual degradation scenario functions
# ---------------------------------------------------------------------------

def inject_memory_leak(
    df: pd.DataFrame,
    start_step: int = 200,
    severity: float = 1.0,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, int]:
    """
    Model a gradual memory leak starting at `start_step`.

    Memory increases monotonically at rate `severity * 2 MB/step`
    with small Gaussian noise added on top.

    Returns (modified_df, breach_step) where breach_step marks when
    memory has grown enough to constitute an SLA violation
    (defined as 2x the initial memory baseline at start).
    """
    rng = np.random.default_rng(random_seed)
    df = df.copy()
    n = len(df)
    steps_after_start = np.arange(0, n - start_step)

    leak_rate = severity * 2.0  # MB per step
    leak = leak_rate * steps_after_start + rng.normal(0, 2, len(steps_after_start))

    df.iloc[start_step:, df.columns.get_loc("memory_mb")] += leak

    # Ground-truth SLA breach: when memory doubles from start baseline
    baseline_at_start = df.iloc[start_step]["memory_mb"]
    breach_step = start_step
    for i in range(start_step, n):
        if df.iloc[i]["memory_mb"] >= baseline_at_start * 2:
            breach_step = i
            break

    return df, breach_step


def inject_cpu_spike_burst(
    df: pd.DataFrame,
    start_step: int = 200,
    severity: float = 1.0,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, int]:
    """
    Inject recurring CPU spike bursts after `start_step`.

    Every ~20 steps, a spike of severity * 30% of headroom occurs.
    Ground-truth SLA breach: first sustained spike above 90%.
    """
    rng = np.random.default_rng(random_seed)
    df = df.copy()
    n = len(df)
    breach_step = n - 1

    for i in range(start_step, n):
        if (i - start_step) % 20 < 8:  # spike window of 8 steps
            spike = severity * rng.uniform(25, 45)
            df.iloc[i, df.columns.get_loc("cpu_usage")] = min(
                df.iloc[i]["cpu_usage"] + spike, 100.0
            )
            if df.iloc[i]["cpu_usage"] > 90 and breach_step == n - 1:
                breach_step = i

    return df, breach_step


def inject_latency_trend(
    df: pd.DataFrame,
    start_step: int = 200,
    severity: float = 1.0,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, int]:
    """
    Linear latency increase starting at start_step.

    Rate: severity * 1.5 ms/step.
    Ground-truth SLA breach: when latency exceeds 3x baseline latency.
    """
    rng = np.random.default_rng(random_seed)
    df = df.copy()
    n = len(df)
    steps_after = np.arange(0, n - start_step)

    rate = severity * 1.5  # ms per step
    increase = rate * steps_after + rng.normal(0, 5, len(steps_after))

    df.iloc[start_step:, df.columns.get_loc("latency_ms")] += increase

    baseline_latency = df.iloc[:start_step]["latency_ms"].mean()
    breach_step = n - 1
    for i in range(start_step, n):
        if df.iloc[i]["latency_ms"] >= baseline_latency * 3:
            breach_step = i
            break

    return df, breach_step


def inject_error_explosion(
    df: pd.DataFrame,
    start_step: int = 200,
    severity: float = 1.0,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, int]:
    """
    Sudden explosion in error rate starting at start_step.

    Error count jumps from near-zero to severity * Poisson(20).
    Ground-truth SLA breach: first step where error_rate > 10.
    """
    rng = np.random.default_rng(random_seed)
    df = df.copy()
    n = len(df)
    breach_step = n - 1

    for i in range(start_step, n):
        errors = rng.poisson(severity * 20)
        df.iloc[i, df.columns.get_loc("error_rate")] += errors
        if errors > 10 and breach_step == n - 1:
            breach_step = i

    return df, breach_step


def inject_combined_degradation(
    df: pd.DataFrame,
    start_step: int = 200,
    severity: float = 1.0,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, int]:
    """
    Apply all four degradation scenarios simultaneously with shared start_step.

    The ground-truth SLA breach step is the EARLIEST trigger across all modes.
    """
    df, fs1 = inject_memory_leak(df, start_step, severity * 0.8, random_seed)
    df, fs2 = inject_cpu_spike_burst(df, start_step, severity * 0.8, random_seed + 1)
    df, fs3 = inject_latency_trend(df, start_step, severity * 0.8, random_seed + 2)
    df, fs4 = inject_error_explosion(df, start_step, severity * 0.8, random_seed + 3)

    breach_step = min(fs1, fs2, fs3, fs4)
    return df, breach_step


# ---------------------------------------------------------------------------
# Registry for easy experiment lookup
# ---------------------------------------------------------------------------

DEGRADATION_SCENARIOS = {
    "memory_leak": inject_memory_leak,
    "cpu_spike_burst": inject_cpu_spike_burst,
    "latency_trend": inject_latency_trend,
    "error_explosion": inject_error_explosion,
    "combined_degradation": inject_combined_degradation,
}

# Backward-compatible alias
FAILURE_MODES = DEGRADATION_SCENARIOS


def inject_degradation(
    df: pd.DataFrame,
    mode: str,
    start_step: int = 200,
    severity: float = 1.0,
    random_seed: int = 0,
) -> tuple[pd.DataFrame, int]:
    """
    Dispatch to the appropriate degradation scenario by name.

    Parameters
    ----------
    df         : Baseline DataFrame from simulator.py
    mode       : One of the keys in DEGRADATION_SCENARIOS
    start_step : Time index where degradation begins
    severity   : Multiplier (1.0 = nominal, 2.0 = aggressive)
    random_seed: For reproducibility

    Returns
    -------
    (modified_df, breach_step)
    """
    if mode not in DEGRADATION_SCENARIOS:
        raise ValueError(
            f"Unknown degradation scenario '{mode}'. "
            f"Choose from: {list(DEGRADATION_SCENARIOS)}"
        )
    return DEGRADATION_SCENARIOS[mode](df, start_step, severity, random_seed)


# Backward-compatible alias
inject_failure = inject_degradation


def build_ground_truth_labels(n_steps: int, failure_step: int) -> np.ndarray:
    """
    Build a binary ground-truth array:
      0 = normal operation
      1 = degraded / SLA violation  (from failure_step onward)
    """
    labels = np.zeros(n_steps, dtype=int)
    labels[failure_step:] = 1
    return labels
