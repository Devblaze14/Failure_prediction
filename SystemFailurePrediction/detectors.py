"""
detectors.py
------------
Three families of anomaly detectors with a common interface.

Interface contract for every detector class:
    fit(df_train)            -> self          (learn baseline)
    predict(df)              -> np.ndarray   (anomaly scores, 1 per step)
    classify(df, threshold)  -> np.ndarray   (binary: 1=anomaly, 0=normal)

Detector families:
  A) RuleBasedDetector   — static thresholds + slope detection
  B) StatisticalDetector — rolling mean/std + Z-score
  C) MLDetector          — Isolation Forest on multivariate metrics
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


METRIC_COLS = ["cpu_usage", "memory_mb", "latency_ms", "error_rate"]

# ---------------------------------------------------------------------------
# A) Rule-Based Detector
# ---------------------------------------------------------------------------

class RuleBasedDetector:
    """
    Combines two strategies:
      1. Static threshold breach — any metric exceeds a fixed absolute limit
      2. Slope detection         — rolling slope of each metric exceeds a rate limit

    Anomaly score: count of rules triggered (0–N), normalised to [0, 1].
    """

    DEFAULT_THRESHOLDS = {
        "cpu_usage": 80.0,      # %
        "memory_mb": 900.0,     # MB
        "latency_ms": 300.0,    # ms
        "error_rate": 5.0,      # errors/interval
    }

    DEFAULT_SLOPE_LIMITS = {
        "cpu_usage": 3.0,       # % per step
        "memory_mb": 8.0,       # MB per step
        "latency_ms": 5.0,      # ms per step
        "error_rate": 2.0,      # errors per step
    }

    def __init__(
        self,
        thresholds: dict | None = None,
        slope_limits: dict | None = None,
        slope_window: int = 10,
    ):
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.slope_limits = {**self.DEFAULT_SLOPE_LIMITS, **(slope_limits or {})}
        self.slope_window = slope_window
        self._n_rules = len(METRIC_COLS) * 2  # threshold + slope per metric

    def fit(self, df_train: pd.DataFrame) -> "RuleBasedDetector":
        # Rule-based is parameter-free; fit is a no-op kept for API consistency.
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return anomaly score in [0, 1] (fraction of rules fired)."""
        n = len(df)
        scores = np.zeros(n)

        for col in METRIC_COLS:
            series = df[col].values

            # Rule 1: static threshold breach
            threshold_breach = (series > self.thresholds[col]).astype(float)

            # Rule 2: slope too steep (rolling linear slope via diff / window)
            slopes = np.zeros(n)
            for i in range(self.slope_window, n):
                window = series[i - self.slope_window: i]
                slopes[i] = (window[-1] - window[0]) / self.slope_window
            slope_breach = (np.abs(slopes) > self.slope_limits[col]).astype(float)

            scores += threshold_breach + slope_breach

        return scores / self._n_rules  # normalise to [0, 1]

    def classify(self, df: pd.DataFrame, threshold: float = 0.2) -> np.ndarray:
        return (self.predict(df) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# B) Statistical Detector
# ---------------------------------------------------------------------------

class StatisticalDetector:
    """
    Uses rolling statistics over the training baseline:
      - Rolling mean ± k*std (control-chart style)
      - Per-step Z-score relative to global baseline

    Anomaly score: mean absolute Z-score across all metrics, capped at 5.
    """

    def __init__(self, window: int = 30, z_threshold: float = 3.0):
        self.window = window
        self.z_threshold = z_threshold
        self._baseline_mean: dict = {}
        self._baseline_std: dict = {}

    def fit(self, df_train: pd.DataFrame) -> "StatisticalDetector":
        for col in METRIC_COLS:
            self._baseline_mean[col] = df_train[col].mean()
            self._baseline_std[col] = max(df_train[col].std(), 1e-6)
        return self

    def _compute_z_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Per-step Z-score matrix: shape (n_steps, n_metrics)."""
        z_matrix = np.zeros((len(df), len(METRIC_COLS)))
        for j, col in enumerate(METRIC_COLS):
            z_matrix[:, j] = (
                (df[col].values - self._baseline_mean[col]) / self._baseline_std[col]
            )
        return z_matrix

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Anomaly score = mean |Z| across all metrics, normalised:
          score = min(mean_abs_z / 5, 1.0)
        """
        z = self._compute_z_scores(df)
        mean_abs_z = np.abs(z).mean(axis=1)
        return np.minimum(mean_abs_z / 5.0, 1.0)

    def classify(self, df: pd.DataFrame, threshold: float = 0.3) -> np.ndarray:
        return (self.predict(df) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# C) ML Detector — Isolation Forest
# ---------------------------------------------------------------------------

class MLDetector:
    """
    Multivariate anomaly detection using scikit-learn's Isolation Forest.

    Scores are converted from sklearn's raw scores (negative: more anomalous)
    to [0, 1] where 1 = most anomalous.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, df_train: pd.DataFrame) -> "MLDetector":
        X = self._scaler.fit_transform(df_train[METRIC_COLS].values)
        self._model.fit(X)
        # Store the score range observed during training for normalisation
        raw_scores = self._model.score_samples(X)
        self._train_score_min = raw_scores.min()
        self._train_score_max = raw_scores.max()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly scores in [0, 1]:
          0 = very normal, 1 = very anomalous.
        """
        X = self._scaler.transform(df[METRIC_COLS].values)
        raw = self._model.score_samples(X)

        # Lower score_samples → more anomalous → invert and normalise
        score_range = max(self._train_score_max - self._train_score_min, 1e-6)
        normalised = (self._train_score_max - raw) / score_range
        return np.clip(normalised, 0, 1)

    def classify(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict(df) >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DETECTORS = {
    "rule_based": RuleBasedDetector,
    "statistical": StatisticalDetector,
    "ml_isolation_forest": MLDetector,
}


def build_detector(name: str, **kwargs):
    """Instantiate a detector by name with optional kwargs."""
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector '{name}'. Available: {list(DETECTORS)}")
    return DETECTORS[name](**kwargs)
