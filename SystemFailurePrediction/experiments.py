"""
experiments.py
--------------
Runs a structured grid of experiments varying:
  - Failure mode
  - Noise level
  - Failure severity
  - Detection window size

For each configuration, all three detectors are evaluated and metrics
are collected into a results table.

Entry point: run_experiments() → returns pd.DataFrame of all results.
"""

import numpy as np
import pandas as pd

from simulator import generate_baseline, get_baseline_stats
from failure_injection import inject_failure, build_ground_truth_labels
from detectors import RuleBasedDetector, StatisticalDetector, MLDetector
from prediction import sliding_window_prediction, compute_early_warning_time
from evaluation import compute_metrics


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

FAILURE_MODES = ["memory_leak", "cpu_spike_burst", "latency_trend",
                 "error_explosion", "combined_degradation"]
NOISE_LEVELS = [0.5, 1.0, 2.0]
SEVERITIES = [0.5, 1.0, 2.0]
WINDOW_SIZES = [10, 20, 40]

N_STEPS = 500
TRAIN_SPLIT = 0.4          # first 40% = baseline / training data
FAILURE_START_FRAC = 0.5   # failure injected at 50% of timeline


def _single_experiment(
    failure_mode: str,
    noise_level: float,
    severity: float,
    window_size: int,
    random_seed: int = 42,
) -> list[dict]:
    """
    Run one experiment configuration for all three detectors.

    Returns a list of metric dicts (one per detector).
    """
    # 1. Generate baseline data
    df_baseline = generate_baseline(
        n_steps=N_STEPS,
        noise_multiplier=noise_level,
        random_seed=random_seed,
    )

    # 2. Split: use early portion as clean training set
    n_train = int(N_STEPS * TRAIN_SPLIT)
    df_train = df_baseline.iloc[:n_train]

    # 3. Inject failure into full timeline
    start_step = int(N_STEPS * FAILURE_START_FRAC)
    df_injected, failure_step = inject_failure(
        df_baseline, mode=failure_mode,
        start_step=start_step, severity=severity,
        random_seed=random_seed,
    )

    baseline_stats = get_baseline_stats(df_train)

    # 4. Ground truth labels
    y_true = build_ground_truth_labels(N_STEPS, failure_step)

    # 5. Evaluate each detector
    detectors = {
        "Rule-Based": RuleBasedDetector(),
        "Statistical": StatisticalDetector(window=window_size),
        "ML (IsoForest)": MLDetector(random_state=random_seed),
    }

    results = []
    for det_name, detector in detectors.items():
        try:
            detector.fit(df_train)
            scores = detector.predict(df_injected)
            y_pred, trigger_step = sliding_window_prediction(
                scores,
                window_size=window_size,
                density_threshold=0.5,
            )

            metrics = compute_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_scores=scores,
                failure_step=failure_step,
                trigger_step=trigger_step,
            )
            metrics.update({
                "detector": det_name,
                "failure_mode": failure_mode,
                "noise_level": noise_level,
                "severity": severity,
                "window_size": window_size,
                "failure_step": failure_step,
                "trigger_step": trigger_step,
            })
        except Exception as exc:
            metrics = {
                "detector": det_name,
                "failure_mode": failure_mode,
                "noise_level": noise_level,
                "severity": severity,
                "window_size": window_size,
                "error": str(exc),
            }
        results.append(metrics)

    return results


def run_experiments(
    failure_modes: list[str] | None = None,
    noise_levels: list[float] | None = None,
    severities: list[float] | None = None,
    window_sizes: list[int] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Execute the full experiment grid and return a results DataFrame.

    Parameters are lists of values to test; passing None uses the defaults.
    """
    failure_modes = failure_modes or FAILURE_MODES
    noise_levels = noise_levels or NOISE_LEVELS
    severities = severities or SEVERITIES
    window_sizes = window_sizes or WINDOW_SIZES

    all_results = []
    total = len(failure_modes) * len(noise_levels) * len(severities) * len(window_sizes)
    done = 0

    for mode in failure_modes:
        for noise in noise_levels:
            for severity in severities:
                for window in window_sizes:
                    seed = int(noise * 10 + severity * 100 + window)
                    batch = _single_experiment(mode, noise, severity, window, seed)
                    all_results.extend(batch)
                    done += 1
                    if verbose:
                        pct = 100 * done / total
                        print(f"  [{pct:5.1f}%] mode={mode}, noise={noise}, "
                              f"sev={severity}, win={window}")

    df_results = pd.DataFrame(all_results)
    return df_results


def summarise_by_detector(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Average performance metrics across all experiment configurations,
    grouped by detector name.
    """
    numeric_cols = ["precision", "recall", "f1", "fpr", "auc_roc",
                    "detection_latency_steps"]
    # Filter to rows with numeric data
    df_clean = df_results[["detector"] + numeric_cols].dropna()
    summary = df_clean.groupby("detector")[numeric_cols].mean().round(4)
    return summary
