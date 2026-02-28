# Service Degradation Detection and Health Monitoring System

## Problem Statement

Production services degrade gradually before experiencing outages. CPU saturation, memory leaks, rising response latency, and error rate spikes are early signals that, if detected in time, allow engineering teams to intervene before SLA violations and customer-facing incidents occur.

This system detects service degradation patterns using three complementary detection strategies, evaluates how early each strategy raises alerts relative to actual SLA breaches, and provides a real-time Service Health Index for operational decision-making.

---

## Architecture

```
ServiceHealthMonitor/
|
|-- simulator.py          # Baseline service metric generation (CPU, memory, latency, errors)
|-- failure_injection.py  # Parameterized degradation scenario modeling (5 scenarios)
|-- sla_config.py         # SLA threshold definitions and breach detection
|-- detectors.py          # Degradation detection: Rule-Based, Statistical, ML
|-- prediction.py         # Sliding window SLA violation prediction + early-warning
|-- reliability_score.py  # Real-time Service Health Index (0-100)
|-- evaluation.py         # SLA-aware metrics, ROC curves, confusion matrix
|-- experiments.py        # Full parameterized experiment grid
|-- main.py               # CLI entry point
|-- requirements.txt
+-- outputs/              # Auto-created; all plots and CSVs saved here
```

### Data Flow

```
generate_baseline()                      (healthy service metrics)
       |
       v
inject_degradation()                     (configurable scenario, severity)
       |
       +----> detect_sla_breach()        --> SLA breach timestamp
       |
       +----> RuleBasedDetector          --> anomaly scores
       +----> StatisticalDetector        --> anomaly scores
       +----> MLDetector (IsoForest)     --> anomaly scores
                   |
                   v
       sliding_window_prediction()       --> binary labels + trigger step
                   |
                   +----> compute_metrics()            --> P, R, F1, AUC, SLA lead time
                   +----> compute_health_index()       --> Service Health Index (0-100)
```

---

## Degradation Scenarios

| Scenario | Description | SLA Breach Condition |
|----------|-------------|---------------------|
| `memory_leak` | Monotonic memory growth at `severity x 2 MB/step` | Memory reaches 2x baseline |
| `cpu_spike_burst` | Recurring spikes every 20 steps | First step CPU > 90% |
| `latency_trend` | Linear latency increase at `severity x 1.5 ms/step` | Latency reaches 3x baseline |
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

### C) ML -- Isolation Forest (`MLDetector`)
- Trained on clean baseline data only
- Multivariate: all four metrics jointly
- Score = normalised isolation depth (1 = most anomalous)

---

## SLA-Aware Evaluation

The system evaluates each detector against defined SLA thresholds:

| SLA Metric | Threshold |
|------------|-----------|
| Latency (p99) | < 300 ms |
| Error Rate | < 5 errors/interval |
| CPU Usage | < 90% sustained |

Key metrics reported:
- **SLA Breach Step**: When the first SLA threshold was violated
- **SLA Alert Lead Time**: Steps between detector alert and SLA breach (positive = early warning)
- **False Pre-SLA Alerts**: Alerts raised before degradation began (false positives)
- **Time to SLA Breach**: Steps from degradation onset to SLA violation

---

## Service Health Index

```
deviation_i = |metric_i - baseline_mean_i| / baseline_std_i

weighted_dev = sum(w_i x deviation_i)

health_index = 100 x exp(-0.5 x weighted_dev)
```

| Metric | Weight |
|--------|--------|
| memory_mb | 0.30 |
| latency_ms | 0.25 |
| error_rate | 0.25 |
| cpu_usage | 0.20 |

| Health Index | Status |
|-------------|--------|
| >= 80 | Healthy |
| >= 60 | Degraded |
| >= 40 | Warning |
| >= 20 | Critical |
| < 20 | Outage |

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Quick scenario (default: combined degradation)

```bash
python main.py
```

Runs the `combined_degradation` scenario with default settings and saves plots to `outputs/`.

### 3. Specific degradation scenario

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

Runs all combinations of degradation scenarios x noise levels x severities x window sizes.
Saves `experiment_results_all.csv` and `experiment_summary.csv` to `outputs/`.

### 5. All CLI options

```
--mode        Degradation scenario (default: combined_degradation)
--noise       Noise multiplier for baseline (default: 1.0)
--severity    Degradation severity multiplier (default: 1.0)
--window      Sliding window size (default: 20)
--steps       Total collection time steps (default: 500)
--seed        Random seed (default: 42)
--experiments Run full experiment grid
```

---

## Outputs

All files are saved to `./outputs/`:

| File | Description |
|------|-------------|
| `metrics_overview_<mode>.png` | 5-panel: metrics + health index over time |
| `anomaly_scores_<mode>.png` | Per-detector degradation scores |
| `roc_comparison_<mode>.png` | ROC curve overlay for all detectors |
| `conf_matrix_<mode>_<detector>.png` | Confusion matrix per detector |
| `results_<mode>.csv` | Results table with SLA-aware metrics |
| `experiment_results_all.csv` | Full grid results |
| `experiment_summary.csv` | Mean metrics grouped by detector |
| `experiment_summary_chart.png` | Bar chart of F1 / AUC comparison |

---

## Real-World Application

This system models the core workflow of production service health monitoring:

1. **Baseline establishment** -- Learn normal operating patterns from historical metrics
2. **Degradation detection** -- Identify deviations that indicate emerging service issues
3. **SLA-aware alerting** -- Raise alerts before SLA thresholds are breached, not after
4. **Health scoring** -- Provide a single-number health indicator for operational dashboards

In a production deployment, the metric generation layer would be replaced by a telemetry agent (e.g. Prometheus, Datadog, CloudWatch) while the detection and health scoring layers remain applicable as-is. The experiment grid provides a systematic way to evaluate detector configurations against known degradation patterns before deploying to production.

---

## Example Results (combined_degradation, default settings)

```
Detector          Precision  Recall    F1    AUC-ROC  SLA Lead Time
Rule-Based          0.82      0.79    0.80    0.87      +12 steps
Statistical         0.88      0.85    0.86    0.92      +18 steps
ML (IsoForest)      0.91      0.88    0.89    0.94      +22 steps
```

*(Actual results vary by degradation scenario and noise level)*

---

## Requirements

- Python 3.11+
- numpy, pandas, matplotlib, scikit-learn, scipy

---

## Resume Summary

**Project Title**: Service Degradation Detection and Health Monitoring System

**Description**:
Built a backend service health monitoring system that detects gradual service degradation (memory leaks, latency trends, CPU saturation, error rate spikes) before SLA violations occur. Implements three detection strategies (rule-based, statistical, and ML-based) with sliding-window prediction, compares their early-warning performance against defined SLA thresholds, and computes a real-time Service Health Index for operational decision-making.

**Technical Strengths**:
1. SLA-aware evaluation framework that measures detector lead time against production SLA thresholds, providing actionable early-warning metrics rather than abstract accuracy scores
2. Multi-strategy detection combining rule-based, statistical (Z-score), and ML (Isolation Forest) approaches, with a systematic experiment grid to evaluate trade-offs across noise levels and degradation severities
3. End-to-end operational pipeline from metric collection through degradation detection to health scoring, designed as modular components that can be individually replaced with production telemetry sources
