"""
evaluation.py
-------------
Evaluation utilities for degradation detectors.

Functions:
  - compute_metrics()     : Precision, Recall, F1, FPR, Detection Latency, Early-Warning Time,
                            SLA breach metrics
  - plot_roc_curve()      : ROC curve for a single detector
  - plot_roc_comparison() : Overlay ROC curves for multiple detectors
  - plot_confusion_matrix(): Visualise TP/FP/TN/FN
  - build_comparison_table(): DataFrame comparing all detectors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay,
)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    failure_step: int,
    trigger_step: int,
    sla_breach_step: int = -1,
    degradation_start_step: int = -1,
) -> dict:
    """
    Compute a complete set of evaluation metrics including SLA-aware results.

    Parameters
    ----------
    y_true       : Ground-truth binary labels (1=degraded)
    y_pred       : Binary predictions from classifier
    y_scores     : Continuous anomaly scores (for AUC)
    failure_step : Actual degradation breach index from ground truth
    trigger_step : First step where early-warning fired
    sla_breach_step : First step where an SLA threshold was violated (-1 if never)
    degradation_start_step : Step where degradation injection began (-1 if unknown)

    Returns
    -------
    dict with keys: precision, recall, f1, fpr, auc_roc,
                    detection_latency, early_warning_steps,
                    sla_breach_step, sla_alert_lead_time,
                    false_pre_sla_alerts, time_to_sla_breach
    """
    # Guard: avoid division-by-zero when no positives predicted
    if y_pred.sum() == 0:
        precision = 0.0
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)

    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # False positive rate = FP / (FP + TN)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # AUC-ROC
    try:
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_scores)
        auc_roc = auc(fpr_arr, tpr_arr)
    except Exception:
        auc_roc = float("nan")

    # Detection latency: steps from failure_step to first prediction at/after it
    latency = -1
    for i in range(failure_step, len(y_pred)):
        if y_pred[i] == 1:
            latency = i - failure_step
            break

    # Early warning: steps BEFORE failure_step that the trigger fired
    if trigger_step == -1:
        early_warning = None  # never fired
    else:
        early_warning = failure_step - trigger_step  # positive = before failure

    # -----------------------------------------------------------------------
    # SLA-aware metrics
    # -----------------------------------------------------------------------

    # SLA alert lead time: how many steps before SLA breach the alert fired
    if sla_breach_step == -1 or trigger_step == -1:
        sla_alert_lead_time = None
    else:
        sla_alert_lead_time = sla_breach_step - trigger_step

    # False pre-SLA alerts: predictions that fired before the SLA breach
    # in the baseline period (before degradation started)
    if sla_breach_step == -1:
        false_pre_sla = 0
    else:
        # Count prediction=1 steps that occur before both degradation start
        # and the SLA breach — these are false pre-SLA warnings
        cutoff = min(sla_breach_step, failure_step)
        pre_region = y_pred[:cutoff]
        false_pre_sla = int(pre_region.sum())

    # Time to SLA breach from degradation start
    if sla_breach_step == -1 or degradation_start_step == -1:
        time_to_sla = None
    else:
        time_to_sla = sla_breach_step - degradation_start_step

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "auc_roc": round(auc_roc, 4),
        "detection_latency_steps": latency,
        "early_warning_steps": early_warning,
        "sla_breach_step": sla_breach_step,
        "sla_alert_lead_time": sla_alert_lead_time,
        "false_pre_sla_alerts": false_pre_sla,
        "time_to_sla_breach": time_to_sla,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    detector_name: str,
    ax: plt.Axes | None = None,
    color: str = "steelblue",
) -> plt.Axes:
    """Plot a single ROC curve onto `ax`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr_arr, tpr_arr)

    ax.plot(fpr_arr, tpr_arr, color=color, lw=2,
            label=f"{detector_name}  (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    return ax


def plot_roc_comparison(
    results: list[dict],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Overlay ROC curves for multiple detectors.

    Parameters
    ----------
    results : list of dicts, each must have keys:
              'name', 'y_true', 'y_scores'
    """
    colors = ["steelblue", "tomato", "seagreen", "darkorange", "purple"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, r in enumerate(results):
        try:
            fpr_arr, tpr_arr, _ = roc_curve(r["y_true"], r["y_scores"])
            roc_auc = auc(fpr_arr, tpr_arr)
            ax.plot(fpr_arr, tpr_arr, color=colors[i % len(colors)], lw=2,
                    label=f"{r['name']}  AUC={roc_auc:.3f}")
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    detector_name: str,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Render a labelled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax_ = (plt.subplots(figsize=(4, 3)) if ax is None else (ax.get_figure(), ax))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Degraded"])
    disp.plot(ax=ax_, colorbar=False, cmap="Blues")
    ax_.set_title(f"Confusion Matrix - {detector_name}")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def build_comparison_table(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Build a summary DataFrame from a list of metric dicts.

    Each dict should have a 'detector' key plus the standard metric keys.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "Detector": m.get("detector", "unknown"),
            "Scenario": m.get("failure_mode", ""),
            "Precision": m.get("precision", float("nan")),
            "Recall": m.get("recall", float("nan")),
            "F1": m.get("f1", float("nan")),
            "FPR": m.get("fpr", float("nan")),
            "AUC-ROC": m.get("auc_roc", float("nan")),
            "Det. Latency (steps)": m.get("detection_latency_steps", -1),
            "Early Warning (steps)": m.get("early_warning_steps", None),
            "SLA Breach Step": m.get("sla_breach_step", -1),
            "SLA Lead Time": m.get("sla_alert_lead_time", None),
            "False Pre-SLA Alerts": m.get("false_pre_sla_alerts", 0),
        })
    return pd.DataFrame(rows)
