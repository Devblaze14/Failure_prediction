"""
prediction.py
-------------
Converts per-step anomaly scores into an SLA violation prediction signal.

Strategy:
  - Maintain a sliding window of the last `window_size` anomaly scores.
  - Compute anomaly density = fraction of steps classified as anomalous.
  - Trigger early-warning when density >= `density_threshold`.

Outputs:
  - prediction_labels : binary array (1 = predicted degradation, 0 = normal)
  - trigger_step      : first step where early-warning fired (-1 if never)
  - early_warning_time: (breach_step - trigger_step) steps; > 0 means ahead of breach
"""

import numpy as np


def sliding_window_prediction(
    anomaly_scores: np.ndarray,
    window_size: int = 20,
    anomaly_threshold: float = 0.4,
    density_threshold: float = 0.5,
) -> tuple[np.ndarray, int]:
    """
    Predict SLA violation using sliding window anomaly density.

    Parameters
    ----------
    anomaly_scores    : float array in [0, 1] from any detector
    window_size       : number of steps to look back
    anomaly_threshold : score cutoff to mark a step as anomalous
    density_threshold : fraction of anomalous steps in window to fire warning

    Returns
    -------
    prediction_labels : binary np.array (1 = degradation predicted at this step)
    trigger_step      : index of first prediction trigger (-1 if none)
    """
    n = len(anomaly_scores)
    binary_anomaly = (anomaly_scores >= anomaly_threshold).astype(int)
    prediction_labels = np.zeros(n, dtype=int)
    trigger_step = -1
    triggered = False

    for i in range(window_size, n):
        window = binary_anomaly[i - window_size: i]
        density = window.mean()
        if density >= density_threshold:
            prediction_labels[i] = 1
            if not triggered:
                trigger_step = i
                triggered = True

    return prediction_labels, trigger_step


def compute_early_warning_time(
    trigger_step: int,
    failure_step: int,
) -> int:
    """
    Compute how many steps before the actual SLA breach the predictor fired.

    Positive value  -> early warning (good).
    Zero            -> detected exactly at breach.
    Negative value  -> detected after breach (late).
    -1 as trigger_step means the predictor never fired.
    """
    if trigger_step == -1:
        return int(-1e9)  # sentinel: never triggered
    return failure_step - trigger_step


def detection_latency(
    prediction_labels: np.ndarray,
    failure_step: int,
) -> int:
    """
    Steps between breach_step and the first positive prediction AT or AFTER breach_step.

    Returns -1 if the detector never fires after the breach.
    """
    for i in range(failure_step, len(prediction_labels)):
        if prediction_labels[i] == 1:
            return i - failure_step
    return -1
