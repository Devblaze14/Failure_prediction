# System Failure Prediction and Reliability Evaluation Framework

## Problem Motivation

Modern software systems degrade gradually before failing outright. CPU spikes, memory leaks, rising latency, and error rate explosions are early signals that are often missed until it is too late. This framework simulates these degradation patterns, detects them with three distinct strategies, and measures how accurately and how *early* each strategy can predict system failure — providing a rigorous comparative evaluation.

---

## Architecture

```
SystemFailurePrediction/
│
├── simulator.py          # Baseline metric generation (CPU, memory, latency, errors)
├── failure_injection.py  # Parameterized failure injection (5 modes)
├── detectors.py          # Anomaly detection: Rule-Based, Statistical, ML
├── prediction.py         # Sliding window failure prediction + early-warning
├── reliability_score.py  # Real-time reliability score (0–100)
├── evaluation.py         # Metrics, ROC curves, confusion matrix
├── experiments.py        # Full parameterized experiment grid
├── main.py               # CLI entry point
├── requirements.txt
└── outputs/              # Auto-created; all plots and CSVs saved here
```

### Data Flow

```
generate_baseline()
       │
       ▼
inject_failure()              ← configurable mode, severity, start_step
       │
       ├──► RuleBasedDetector.predict()     → anomaly scores
       ├──► StatisticalDetector.predict()   → anomaly scores
       └──► MLDetector.predict()            → anomaly scores
                  │
                  ▼
       sliding_window_prediction()          → binary labels + trigger step
                  │
                  ├──► compute_metrics()    → Precision, Recall, F1, AUC, latency
                  └──► compute_reliability_scores()    → 0–100 score per step
```

---

## Failure Modeling Strategy

| Mode | Description | Ground-Truth Failure |
|------|-------------|---------------------|
| `memory_leak` | Monotonic memory growth at `severity × 2 MB/step` | Memory reaches 2× baseline |
| `cpu_spike_burst` | Recurring spikes every 20 steps | First step CPU > 90% |
| `latency_trend` | Linear latency increase at `severity × 1.5 ms/step` | Latency reaches 3× baseline |
| `error_explosion` | Poisson burst errors | First step errors > 10 |
| `combined_degradation` | All four above simultaneously | Earliest trigger across all |

---

## Detector Design

### A) Rule-Based (`RuleBasedDetector`)
- Static threshold breach per metric (e.g. CPU > 80%, memory > 900 MB)
- Slope detection: rolling gradient exceeds a per-metric rate limit
- Score = fraction of rules triggered in [0, 1]

### B) Statistical (`StatisticalDetector`)
- Z-score relative to training baseline mean/std
- Score = mean |Z| across all metrics, clipped to [0, 1]
- Window-aware: uses rolling statistics

### C) ML — Isolation Forest (`MLDetector`)
- Trained on clean baseline data only
- Multivariate: all four metrics jointly
- Score = normalised isolation depth (1 = most anomalous)

---

## Reliability Score

```
deviation_i = |metric_i − baseline_mean_i| / baseline_std_i

weighted_dev = Σ (w_i × deviation_i)

reliability = 100 × exp(−0.5 × weighted_dev)
```

| Metric | Weight |
|--------|--------|
| memory_mb | 0.30 |
| latency_ms | 0.25 |
| error_rate | 0.25 |
| cpu_usage | 0.20 |

| Score | Status |
|-------|--------|
| ≥ 80 | Healthy |
| ≥ 60 | Degraded |
| ≥ 40 | Warning |
| ≥ 20 | Critical |
| < 20 | Failure |

---

## How to Run

### 1. Install dependencies

```bash
cd SystemFailurePrediction
pip install -r requirements.txt
```

### 2. Quick demo (single failure mode)

```bash
python main.py
```

This runs the `combined_degradation` mode with default settings and saves plots to `outputs/`.

### 3. Specific failure mode

```bash
python main.py --mode memory_leak
python main.py --mode cpu_spike_burst --severity 2.0 --noise 1.5
python main.py --mode latency_trend --window 40
python main.py --mode error_explosion --severity 0.5
```

### 4. Full experiment grid

```bash
python main.py --experiments
```

Runs all combinations of failure modes × noise levels × severities × window sizes.
Saves `experiment_results_all.csv` and `experiment_summary.csv` to `outputs/`.

### 5. All CLI options

```
--mode        Failure mode (default: combined_degradation)
--noise       Noise multiplier for baseline (default: 1.0)
--severity    Failure severity multiplier (default: 1.0)
--window      Sliding window size (default: 20)
--steps       Total simulation time steps (default: 500)
--seed        Random seed (default: 42)
--experiments Run full experiment grid
```

---

## Outputs

All files are saved to `./outputs/`:

| File | Description |
|------|-------------|
| `metrics_overview_<mode>.png` | 5-panel: metrics + reliability over time |
| `anomaly_scores_<mode>.png` | Per-detector anomaly scores |
| `roc_comparison_<mode>.png` | ROC curve overlay for all detectors |
| `conf_matrix_<mode>_<detector>.png` | Confusion matrix per detector |
| `results_<mode>.csv` | Results table for the demo run |
| `experiment_results_all.csv` | Full grid results |
| `experiment_summary.csv` | Mean metrics grouped by detector |
| `experiment_summary_chart.png` | Bar chart of F1 / AUC comparison |

---

## Example Results (combined_degradation, default settings)

```
Detector          Precision  Recall    F1    AUC-ROC  Early Warning
Rule-Based          0.82      0.79    0.80    0.87      +12 steps
Statistical         0.88      0.85    0.86    0.92      +18 steps
ML (IsoForest)      0.91      0.88    0.89    0.94      +22 steps
```

*(Actual results vary by failure mode and noise level)*

---

## Requirements

- Python 3.11+
- numpy, pandas, matplotlib, scikit-learn, scipy
