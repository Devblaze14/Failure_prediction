"""
evaluation.py
-------------
Evaluation utilities for anomaly detectors.

Functions:
  - compute_metrics()     : Precision, Recall, F1, FPR, Detection Latency, Early-Warning Time
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
) -> dict:
    """
    Compute a complete set of evaluation metrics.

    Parameters
    ----------
    y_true       : Ground-truth binary labels (1=failure)
    y_pred       : Binary predictions from classifier
    y_scores     : Continuous anomaly scores (for AUC)
    failure_step : Actual failure index from ground truth
    trigger_step : First step where early-warning fired

    Returns
    -------
    dict with keys: precision, recall, f1, fpr, auc_roc,
                    detection_latency, early_warning_steps
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

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "auc_roc": round(auc_roc, 4),
        "detection_latency_steps": latency,
        "early_warning_steps": early_warning,
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
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Failure"])
    disp.plot(ax=ax_, colorbar=False, cmap="Blues")
    ax_.set_title(f"Confusion Matrix — {detector_name}")

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
            "Failure Mode": m.get("failure_mode", ""),
            "Precision": m.get("precision", float("nan")),
            "Recall": m.get("recall", float("nan")),
            "F1": m.get("f1", float("nan")),
            "FPR": m.get("fpr", float("nan")),
            "AUC-ROC": m.get("auc_roc", float("nan")),
            "Det. Latency (steps)": m.get("detection_latency_steps", -1),
            "Early Warning (steps)": m.get("early_warning_steps", None),
        })
    return pd.DataFrame(rows)
