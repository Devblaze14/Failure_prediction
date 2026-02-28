"""
main.py
-------
Entry point for the System Failure Prediction and Reliability Evaluation Framework.

Usage:
    python main.py                          # default: combined_degradation demo
    python main.py --mode memory_leak       # specific failure mode
    python main.py --experiments            # run full experiment grid
    python main.py --mode latency_trend --severity 2.0 --noise 1.5

All plots are saved to ./outputs/ directory.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend for file saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from simulator import generate_baseline, get_baseline_stats
from failure_injection import inject_failure, build_ground_truth_labels, FAILURE_MODES
from detectors import RuleBasedDetector, StatisticalDetector, MLDetector
from prediction import sliding_window_prediction, compute_early_warning_time
from reliability_score import compute_reliability_scores, reliability_status
from evaluation import (
    compute_metrics,
    plot_roc_comparison,
    plot_confusion_matrix,
    build_comparison_table,
)
from experiments import run_experiments, summarise_by_detector


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# Visualization helpers
# ===========================================================================

def plot_metrics_overview(
    df: pd.DataFrame,
    failure_step: int,
    trigger_step: int,
    reliability: np.ndarray,
    save_path: str,
) -> None:
    """
    5-panel figure:
      1. CPU usage
      2. Memory MB
      3. Latency ms
      4. Error rate
      5. Reliability score
    With vertical lines for failure injection and early-warning trigger.
    """
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle("System Metrics, Failure Injection & Reliability Score", fontsize=14, y=0.98)

    metrics = [
        ("cpu_usage", "CPU Usage (%)", "steelblue"),
        ("memory_mb", "Memory (MB)", "darkorange"),
        ("latency_ms", "Latency (ms)", "seagreen"),
        ("error_rate", "Error Rate (count/interval)", "tomato"),
    ]

    gs = gridspec.GridSpec(5, 1, hspace=0.45)
    x = np.arange(len(df))

    for idx, (col, label, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx])
        ax.plot(x, df[col].values, color=color, linewidth=1.0, alpha=0.85)
        ax.axvline(failure_step, color="red", linestyle="--", linewidth=1.2,
                   label="Failure injection" if idx == 0 else "")
        if trigger_step != -1:
            ax.axvline(trigger_step, color="gold", linestyle=":", linewidth=1.5,
                       label="Early warning" if idx == 0 else "")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    # Reliability score panel
    ax_r = fig.add_subplot(gs[4])
    ax_r.plot(x, reliability, color="mediumpurple", linewidth=1.2)
    ax_r.axvline(failure_step, color="red", linestyle="--", linewidth=1.2)
    if trigger_step != -1:
        ax_r.axvline(trigger_step, color="gold", linestyle=":", linewidth=1.5)
    ax_r.axhline(60, color="orange", linestyle="-.", linewidth=0.8, alpha=0.6, label="Degraded")
    ax_r.axhline(40, color="red", linestyle="-.", linewidth=0.8, alpha=0.6, label="Warning")
    ax_r.set_ylim(0, 105)
    ax_r.set_ylabel("Reliability (0–100)", fontsize=8)
    ax_r.set_xlabel("Time Step", fontsize=9)
    ax_r.legend(fontsize=7, loc="upper right")
    ax_r.grid(alpha=0.25)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_anomaly_scores(
    scores_dict: dict,
    failure_step: int,
    trigger_steps: dict,
    save_path: str,
) -> None:
    """
    Plot anomaly scores from all detectors in stacked subplots.
    """
    n_det = len(scores_dict)
    fig, axes = plt.subplots(n_det, 1, figsize=(13, 3 * n_det), sharex=True)
    if n_det == 1:
        axes = [axes]

    colors = ["steelblue", "tomato", "seagreen"]
    fig.suptitle("Anomaly Scores by Detector", fontsize=13)

    for ax, (det_name, scores), color in zip(axes, scores_dict.items(), colors):
        x = np.arange(len(scores))
        ax.fill_between(x, scores, alpha=0.3, color=color)
        ax.plot(x, scores, color=color, linewidth=1.0)
        ax.axvline(failure_step, color="red", linestyle="--", linewidth=1.2, label="Failure")
        ts = trigger_steps.get(det_name, -1)
        if ts != -1:
            ax.axvline(ts, color="gold", linestyle=":", linewidth=1.5, label="Early Warning")
        ax.set_ylabel("Score [0–1]", fontsize=8)
        ax.set_title(det_name, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Time Step", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_experiment_summary(df_results: pd.DataFrame, save_path: str) -> None:
    """
    Bar chart of mean F1 and AUC-ROC per detector across all experiment runs.
    """
    summary = df_results.groupby("detector")[["f1", "auc_roc"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Experiment Summary: Average Performance by Detector", fontsize=13)

    colors = ["steelblue", "tomato", "seagreen"]

    for ax, metric, ylabel in zip(axes, ["f1", "auc_roc"], ["Mean F1", "Mean AUC-ROC"]):
        bars = ax.bar(summary.index, summary[metric], color=colors, width=0.5, edgecolor="white")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, summary[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# Single-mode demo pipeline
# ===========================================================================

def run_demo(
    mode: str = "combined_degradation",
    noise: float = 1.0,
    severity: float = 1.0,
    window_size: int = 20,
    n_steps: int = 500,
    random_seed: int = 42,
) -> None:
    print(f"\n{'='*60}")
    print(f"  DEMO — Failure Mode : {mode}")
    print(f"  Noise={noise}, Severity={severity}, Window={window_size}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Simulate baseline
    # ------------------------------------------------------------------
    print("[1/5] Generating baseline metrics...")
    df_baseline = generate_baseline(
        n_steps=n_steps,
        noise_multiplier=noise,
        random_seed=random_seed,
    )
    n_train = int(n_steps * 0.4)
    df_train = df_baseline.iloc[:n_train]
    baseline_stats = get_baseline_stats(df_train)

    # ------------------------------------------------------------------
    # 2. Inject failure
    # ------------------------------------------------------------------
    print(f"[2/5] Injecting failure: {mode} ...")
    start_step = int(n_steps * 0.5)
    df_injected, failure_step = inject_failure(
        df_baseline, mode=mode,
        start_step=start_step, severity=severity,
        random_seed=random_seed,
    )
    y_true = build_ground_truth_labels(n_steps, failure_step)
    print(f"       Ground-truth failure at step {failure_step}")

    # ------------------------------------------------------------------
    # 3. Fit detectors and compute scores
    # ------------------------------------------------------------------
    print("[3/5] Running detectors...")
    detectors = {
        "Rule-Based": RuleBasedDetector(),
        "Statistical": StatisticalDetector(window=window_size),
        "ML (IsoForest)": MLDetector(random_state=random_seed),
    }

    scores_dict = {}
    trigger_steps = {}
    metrics_list = []
    roc_inputs = []

    for det_name, detector in detectors.items():
        detector.fit(df_train)
        scores = detector.predict(df_injected)
        y_pred, trigger_step = sliding_window_prediction(
            scores, window_size=window_size, density_threshold=0.5
        )
        scores_dict[det_name] = scores
        trigger_steps[det_name] = trigger_step

        m = compute_metrics(y_true, y_pred, scores, failure_step, trigger_step)
        m["detector"] = det_name
        m["failure_mode"] = mode
        metrics_list.append(m)
        roc_inputs.append({"name": det_name, "y_true": y_true, "y_scores": scores})

        ew = m.get("early_warning_steps")
        ew_str = f"{ew} steps" if ew is not None else "N/A"
        print(f"       {det_name:20s} | F1={m['f1']:.3f} | AUC={m['auc_roc']:.3f} "
              f"| EarlyWarn={ew_str}")

    # Primary early-warning trigger (best trigger across detectors)
    valid = {k: v for k, v in trigger_steps.items() if v != -1}
    best_trigger = min(valid.values()) if valid else -1

    # ------------------------------------------------------------------
    # 4. Reliability score (using Statistical detector scores as proxy)
    # ------------------------------------------------------------------
    print("[4/5] Computing reliability score...")
    reliability = compute_reliability_scores(df_injected, baseline_stats)
    final_status = reliability_status(reliability[-1])
    print(f"       Final reliability: {reliability[-1]:.1f} / 100 — {final_status}")

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    print("[5/5] Generating plots...")

    plot_metrics_overview(
        df_injected, failure_step, best_trigger, reliability,
        os.path.join(OUTPUT_DIR, f"metrics_overview_{mode}.png"),
    )

    plot_anomaly_scores(
        scores_dict, failure_step, trigger_steps,
        os.path.join(OUTPUT_DIR, f"anomaly_scores_{mode}.png"),
    )

    roc_fig = plot_roc_comparison(
        roc_inputs,
        save_path=os.path.join(OUTPUT_DIR, f"roc_comparison_{mode}.png"),
    )
    plt.close(roc_fig)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, f'roc_comparison_{mode}.png')}")

    # Confusion matrix for each detector
    for det_name, detector in detectors.items():
        scores = scores_dict[det_name]
        y_pred, _ = sliding_window_prediction(scores, window_size=window_size, density_threshold=0.5)
        cm_path = os.path.join(OUTPUT_DIR, f"conf_matrix_{mode}_{det_name.replace(' ','_')}.png")
        cm_fig = plot_confusion_matrix(y_true, y_pred, det_name, save_path=cm_path)
        plt.close(cm_fig)
        print(f"  Saved: {cm_path}")

    # Results table
    table = build_comparison_table(metrics_list)
    print("\n" + "-" * 80)
    print("  RESULTS TABLE")
    print("-" * 80)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)
    print(table.to_string(index=False))
    csv_path = os.path.join(OUTPUT_DIR, f"results_{mode}.csv")
    table.to_csv(csv_path, index=False)
    print(f"\n  Table saved: {csv_path}")


# ===========================================================================
# Full experiment pipeline
# ===========================================================================

def run_full_experiments() -> None:
    print("\n" + "=" * 60)
    print("  FULL EXPERIMENT GRID")
    print("=" * 60 + "\n")
    print("  Running all configurations (this may take a few minutes)...\n")

    df_results = run_experiments(verbose=True)

    csv_path = os.path.join(OUTPUT_DIR, "experiment_results_all.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n  All results saved: {csv_path}")

    summary = summarise_by_detector(df_results)
    print("\n  DETECTOR SUMMARY (averages across all configs):")
    print(summary.to_string())
    summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
    summary.to_csv(summary_path)
    print(f"  Summary saved: {summary_path}")

    # Plot summary bar chart
    df_clean = df_results.dropna(subset=["f1", "auc_roc"])
    if not df_clean.empty:
        plot_experiment_summary(df_clean, os.path.join(OUTPUT_DIR, "experiment_summary_chart.png"))


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="System Failure Prediction & Reliability Evaluation Framework",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=list(FAILURE_MODES.keys()),
        default="combined_degradation",
        help="Failure mode to simulate (default: combined_degradation)\n"
             f"Options: {', '.join(FAILURE_MODES.keys())}",
    )
    parser.add_argument("--noise", type=float, default=1.0,
                        help="Noise multiplier for baseline simulation (default: 1.0)")
    parser.add_argument("--severity", type=float, default=1.0,
                        help="Failure severity multiplier (default: 1.0)")
    parser.add_argument("--window", type=int, default=20,
                        help="Sliding window size for prediction (default: 20)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Total simulation time steps (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--experiments", action="store_true",
                        help="Run full experiment grid instead of single demo")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.experiments:
        run_full_experiments()
    else:
        run_demo(
            mode=args.mode,
            noise=args.noise,
            severity=args.severity,
            window_size=args.window,
            n_steps=args.steps,
            random_seed=args.seed,
        )

    print("\nDone. All outputs saved to:", OUTPUT_DIR)
