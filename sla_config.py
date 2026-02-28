"""
sla_config.py
-------------
Lightweight SLA (Service Level Agreement) definitions and breach detection.

Provides:
  - Default SLA thresholds for key service metrics
  - SLA breach detection on a metrics DataFrame
  - Early-warning margin calculation
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default SLA thresholds
# ---------------------------------------------------------------------------

DEFAULT_SLA_THRESHOLDS = {
    "latency_ms": 300.0,    # p99 latency must stay below 300 ms
    "error_rate": 5.0,      # errors per interval must stay below 5
    "cpu_usage": 90.0,      # sustained CPU above 90% violates SLA
}


def detect_sla_breach(
    df: pd.DataFrame,
    thresholds: dict | None = None,
) -> tuple[int, str]:
    """
    Find the first time step where any SLA threshold is violated.

    Parameters
    ----------
    df         : Metrics DataFrame (must contain columns matching threshold keys)
    thresholds : Override default SLA thresholds

    Returns
    -------
    (breach_step, breached_metric)
        breach_step    : index of first SLA violation (-1 if none)
        breached_metric: name of the metric that breached first ("" if none)
    """
    sla = {**DEFAULT_SLA_THRESHOLDS, **(thresholds or {})}
    earliest_step = len(df)
    breached_metric = ""

    for metric, limit in sla.items():
        if metric not in df.columns:
            continue
        values = df[metric].values
        breach_indices = np.where(values > limit)[0]
        if len(breach_indices) > 0 and breach_indices[0] < earliest_step:
            earliest_step = breach_indices[0]
            breached_metric = metric

    if earliest_step >= len(df):
        return -1, ""

    return int(earliest_step), breached_metric


def compute_early_warning_margin(
    sla_breach_step: int,
    alert_step: int,
) -> int:
    """
    Compute how many steps of advance warning the detector provided
    before an actual SLA breach.

    Parameters
    ----------
    sla_breach_step : Step where SLA was first breached (-1 if never)
    alert_step      : Step where the detector first fired (-1 if never)

    Returns
    -------
    margin : int
        Positive = alert came before breach (good).
        Zero     = alert at exact breach time.
        Negative = alert came after breach (late).
        Returns -1 if either step is -1 (no breach or no alert).
    """
    if sla_breach_step == -1 or alert_step == -1:
        return -1
    return sla_breach_step - alert_step
