<div align="center">

# 🛡️ DQAL — Data-Quality-Aware Learning

**Production-grade runtime guardrails, real-time quality scoring, and automated retraining orchestration for machine learning pipelines.**

[![PyPI version](https://img.shields.io/pypi/v/dqal.svg?color=blue)](https://pypi.org/project/dqal/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/dqal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![DQAL CI](https://github.com/Pratham1227-beep/dqal/actions/workflows/ci.yml/badge.svg)](https://github.com/Pratham1227-beep/dqal/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Quickstart](#-quickstart-in-60-seconds) •
[CLI & Shift-Left](#-cli--shift-left-data-quality) •
[Architecture](#-architecture) •
[Key Capabilities](#-key-capabilities) •
[Benchmarks](#-production-benchmarks) •
[Configuration](#-configuration) •
[Documentation](#-framework-integrations)

</div>

---
## Introduction

Machine learning models often perform well during development but can become unreliable after they are deployed in the real world. Changes in incoming data, missing or incorrect values, data drift, and unexpected outliers can cause a model to make confident but incorrect predictions without producing an obvious error. DQAL (Data-Quality-Aware Learning) is designed to address this problem by acting as a safety layer between incoming data and a deployed ML model. It continuously checks the quality of incoming data, detects issues such as missing values, distribution changes, and abnormal data, and assigns a quality score to each batch. Based on this score, DQAL decides whether the model should **SERVE** the prediction, **FLAG** the input for increased risk, or **ABSTAIN** when the data is considered unsafe. It also provides controlled retraining mechanisms to help prevent unreliable or corrupted data from being used to retrain the production model. In this way, DQAL helps make machine learning systems more reliable, explainable, and safer in production.

> [!TIP]
> **Mental Model — Data Quality vs. Code Quality (Shift-Left):**
> Just as linters, pre-commit hooks, and AI code reviewers (like CodeRabbit) inspect and format source code before it gets pushed to GitHub, **DQAL inspects data distributions, missingness, and drift** before inputs reach your ML models. It brings automated, shift-left quality gates to data pipelines.

---

## Overview

Deployed machine learning models fail silently. Even stationary models begin producing overconfident, incorrect predictions when:
- **Upstream schema disruptions occur** (missing payloads, schema drift, API timeouts).
- **Distributions shift** (covariate drift, seasonal changes, demographic variance).
- **Extreme noise penetrates pipelines** (sensor faults, corrupt joins, adversarial perturbations).

Post-hoc monitoring systems identify these failures days or weeks after damage is done. **DQAL** operates **inline** as an intelligent proxy between incoming data streams and your model. It computes a real-time, interpretable Quality Score $Q \in [0, 1]$, gates batch execution through a hysteresis-stabilized state machine, and orchestrates validation-checked retraining.

### Raw Serving vs. DQAL-Gated Serving

| Production Hazard | Unguarded Pipeline | DQAL-Guarded Pipeline |
| :--- | :--- | :--- |
| **Missing / Null Features** | Silent imputation failures, corrupted outputs | Immediate `BLOCKED` fallback, null-rate attribution |
| **Covariate & Feature Drift** | Degraded accuracy with high model confidence | `WARNING` state + automated retrain trigger |
| **Outlier & Sensor Noise** | Extreme prediction spikes, invalid decisions | Outlier isolation via Isolation Forest & Mahalanobis |
| **Boundary Flapping** | Rapidly toggling alerts near decision boundaries | Hysteresis dead-bands & confirmation windows |
| **Unsafe Retraining** | Poisoning active model with unlabeled/dirty data | Gated promotion: requires verified labels + validation gain |

---

## ⚡ Installation

Install DQAL directly from PyPI:

```bash
pip install dqal
```

### Optional Extras

```bash
# With PyTorch adapter support
pip install "dqal[torch]"

# With development and benchmarking tools
pip install "dqal[all]"
```

---

## 🚀 Quickstart in 60 Seconds

DQAL wraps any trained model with zero modifications to your model's internal logic.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from dqal import DQAL

# 1. Prepare data and train model
data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# 2. Wrap model and capture clean baseline distributions
dqal = DQAL(model=model, model_version="v1.0.0")
dqal.fit_baseline(X_train)

# 3. Predict on incoming production batches
result = dqal.predict(X_test.iloc[:50])

# 4. Inspect runtime decisions and sub-signal attribution
print(result.summary())
# Or print(result.summary(compact=True)) for a compact summary
```

### Expected Terminal Output:
```text
==============================================================
                  DQAL Batch Quality Report                   
==============================================================
Gating Decision : [PASSED] Safe - Quality check passed
Quality Score Q : 0.942 / 1.000 (94.2% - High Quality)
Model Version   : v1.0.0
Batch ID        : batch_7a8f1e20

Sub-Signals (1.0 = Pristine, 0.0 = Degraded):
  • Data Completeness     : 100.0% (1.000) - No missing values
  • Covariate Stability   :  91.2% (0.912) - Minimal drift
  • Inlier Adherence      :  93.8% (0.938) - Normal distribution

Sample Predictions: [0, 1, 0, 1, 0] (first 5 of 50 samples)
==============================================================
```

---

## 🔍 CLI & Shift-Left Data Quality

DQAL enables local, shift-left dataset validation in the terminal, in Git pre-commit hooks, and in CI/CD pull-request checks—without requiring an instantiated ML model.

### 1. Terminal CLI (`dqal check`)

Validate any newly collected or ingested dataset (.csv, .parquet, .json) against a baseline distribution:

```bash
# Basic validation against a training baseline
dqal check data/test_batch.csv --baseline data/train_baseline.csv

# Single-line compact output
dqal check data/test_batch.csv --baseline data/train_baseline.csv --compact

# JSON output for automated scripting
dqal check data/test_batch.csv --baseline data/train_baseline.csv --json

# Fail build on warning (exit code 1 for WARNING, 2 for BLOCKED)
dqal check data/test_batch.csv --baseline data/train_baseline.csv --fail-on-warning
```

#### Exit Codes for CI/CD Automation
| Exit Code | Status | Meaning |
| :---: | :--- | :--- |
| `0` | **`PASSED`** | Dataset quality checks passed cleanly |
| `1` | **`WARNING`** / Error | Mild data degradation (with `--fail-on-warning`) or file/argument error |
| `2` | **`BLOCKED`** | Critical data degradation (severe nulls, extreme drift, or outlier explosion) |

---

### 2. Standalone Python API (No Model Required)

Validate data quality during exploratory data analysis (EDA), data wrangling, or pre-flight pipelines:

```python
import pandas as pd
from dqal import check_quality

# Load datasets
current_df = pd.read_csv("data/batch_incoming.csv")
train_df = pd.read_csv("data/train_baseline.csv")

# Run comprehensive data quality assessment
report = check_quality(current_df, baseline=train_df)

# Print human-readable report card
print(report.summary())

# Programmatic checks
if report.blocked:
    raise ValueError(f"Batch rejected! Quality score Q={report.Q:.3f}")
```

---

### 3. Git Pre-Commit Hook Integration

Catch corrupted datasets locally before pushing to remote repositories or data stores:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: dqal-check
        name: DQAL Data Quality Gate
        entry: dqal check
        language: python
        files: ^data/.*\.csv$
        args: ["--baseline", "data/baseline.csv", "--fail-on-warning"]
```

---

### 4. GitHub Actions CI Quality Gate

Add automated dataset quality gates to your pull request workflows:

```yaml
# .github/workflows/data-quality.yml
name: Data Quality Gate
on: [pull_request, push]

jobs:
  validate-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install dqal
      - name: Validate dataset quality
        run: dqal check data/test_batch.csv --baseline data/train_baseline.csv --fail-on-warning
```

---

## 🏗️ Architecture

```
                                INCOMING BATCH DATA
                                        │
                                        ▼
             ┌──────────────────────────────────────────────────────┐
             │               1. QUALITY SCORING ENGINE              │
             │   • Missingness Scorer  (Null rates, schema mask)    │
             │   • Covariate Drift     (Continuous PSI / Cat JS)    │
             │   • Anomaly Scorer      (Isolation Forest / Mahala)  │
             └──────────────────────────┬───────────────────────────┘
                                        │ Composite Q ∈ [0, 1] + Sub-signals
                                        ▼
             ┌──────────────────────────────────────────────────────┐
             │               2. HYSTERESIS TRIGGER GATE             │
             │   • Q > 0.80        →  PASSED   (Green light / Safe)     │
             │   • 0.50 < Q ≤ 0.80  →  WARNING  (Yellow light / Caution) │
             │   • Q ≤ 0.50        →  BLOCKED  (Red light / Safe block) │
             │   * Stabilized with dead-bands & confirmation count  │
             └──────────────────────────┬───────────────────────────┘
                                        │ Decision
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
      ┌─────────────────────────────┐       ┌─────────────────────────────┐
      │      3. WRAPPED MODEL       │       │    4. TELEMETRY LOGGER      │
      │   Executes or blocks;       │       │   Privacy-safe SQLite sync; │
      │   returns safe defaults.    │       │   zero raw PII by default.  │
      └─────────────────────────────┘       └──────────────┬──────────────┘
                                                           │
                                                           ▼
                                            ┌─────────────────────────────┐
                                            │  5. RETRAIN ORCHESTRATOR    │
                                            │   • Data volume checks      │
                                            │   • Ground-truth gating     │
                                            │   • Validation promotion    │
                                            │   • Automated rollback      │
                                            └─────────────────────────────┘
```

---

## 🎯 Key Capabilities

### 1. Model-Agnostic Adapter
Wraps any callable:
- **`scikit-learn`** Estimators and Pipelines
- **`PyTorch`** `nn.Module` networks
- **`XGBoost`** / **`LightGBM`** classifiers and regressors
- Custom Python inference functions

### 2. Decomposable Quality Score ($Q$)
DQAL evaluates quality across three orthogonal axes:

$$Q = w_{\text{missing}} \cdot S_{\text{missing}} + w_{\text{drift}} \cdot S_{\text{drift}} + w_{\text{outlier}} \cdot S_{\text{outlier}}$$

- **Missingness ($S_{\text{missing}}$)**: Penalizes feature null rates relative to baseline tolerance.
- **Drift ($S_{\text{drift}}$)**: Uses **Population Stability Index (PSI)** for continuous features and **Jensen-Shannon (JS) divergence** for categorical attributes.
- **Outlier ($S_{\text{outlier}}$)**: Measures anomaly densities using an **Isolation Forest** or **Mahalanobis distance**.

### 3. Anti-Flapping Hysteresis State Machine
Standard thresholding causes rapid oscillation when batches hover near decision boundaries. DQAL uses:
- **Dead-Bands ($\Delta$)**: Requires score transitions to exceed a buffer zone before switching states.
- **Consecutive Confirmation**: Enforces $k$ consecutive batches in a new zone before triggering state transitions.

### 4. Enterprise Retraining Orchestration
Unlike naive triggers that retrain indiscriminately, DQAL enforces:
- **Sample Sufficiency**: Requires minimum threshold of flagged samples ($N \ge N_{\min}$).
- **Ground-Truth Verification**: Blocks retraining if verified labels are unavailable, preventing label-pollution attacks.
- **Champion-Challenger Validation**: Evaluates candidate models against an out-of-time validation split. Only models exceeding active performance by $\ge \delta_{\min}$ are promoted.
- **Automatic Rollback**: Restores previous active checkpoints instantly if validation criteria fail.

### 5. Privacy-Preserving Observability
Telemetry records runtime statistics, $Q$ trajectories, and retrain logs in high-performance local SQLite storage. **Raw feature logging is disabled by default**, ensuring full compliance with GDPR, HIPAA, and data residency policies.

---

## 📊 Production Benchmarks

DQAL was evaluated across a 60-batch progressive degradation stress-test (`benchmarks/run_benchmark.py`):
1. **Pristine Baseline** (Batches 0–11)
2. **Mild Covariate Shift** (Batches 12–23)
3. **Missing Feature Spikes** (Batches 24–35)
4. **Severe Noise & Anomaly Ingestion** (Batches 36–47)
5. **Catastrophic Distribution Drift** (Batches 48–59)

### SLA Verification Summary

| Evaluation Metric | Production Target | DQAL Empirical Result | Validation Status |
| :--- | :--- | :--- | :--- |
| **Accuracy Correlation ($r$)** | $|r| > 0.70$ | **$r = 0.8502$** ($p = 8.45 \times 10^{-18}$) | Passed |
| **Runtime Overhead** | $< 50$ ms / batch | **$49.79$ ms / batch** | Passed |
| **Early Warning Lead Time** | $\ge 1$ batch | **$9$ batches advance warning** | Passed |
| **Retraining Safety Gate** | $100\%$ compliance | **Blocked $100\%$ of unlabelled retrain attempts** | Passed |

---

## 🛠️ Framework Integrations

### PyTorch Integration

```python
import torch
import torch.nn as nn
from dqal import DQAL

# Define standard PyTorch model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(30, 16), nn.ReLU(), nn.Linear(16, 2))
    def forward(self, x):
        return self.fc(x)

torch_model = Net()

# Wrap and fit baseline on PyTorch or NumPy tensors
dqal = DQAL(model=torch_model, model_version="v1.0.0-torch")
dqal.fit_baseline(X_train)

# Inference returns structured DQALResult
result = dqal.predict(X_test.iloc[:25])
```

### Terminal Telemetry Inspection

Query your telemetry logs directly from Python or shell without leaving your terminal:

```python
from dqal.logger import TelemetryLogger

logger = TelemetryLogger("dqal_telemetry.db")
telemetry = logger.get_telemetry_df()
print(telemetry[["batch_id", "Q", "decision", "missing_signal", "drift_signal", "outlier_signal"]].tail(10))
```

Or query via SQLite CLI:

```bash
sqlite3 dqal_telemetry.db "SELECT batch_id, Q, decision, model_version FROM predictions_telemetry ORDER BY id DESC LIMIT 10;"
```

---

## ⚙️ Configuration

Tune thresholds and weights using `configs/default_config.yaml` or programmatically via `DQALConfig`:

```yaml
# Sub-signal weighting (must sum to 1.0)
weights:
  missingness: 0.30
  drift: 0.40
  outlier: 0.30

# Decision boundaries
thresholds:
  abstain_below: 0.50
  flag_below: 0.80

# Anti-flapping hysteresis controls
hysteresis:
  deadband: 0.03
  consecutive_batches: 2

# Storage & Privacy
logging:
  db_path: "dqal_telemetry.db"
  log_raw_features: false  # Strict opt-in only

# Retraining policy
retrain:
  min_flagged_batches: 5
  min_samples: 100
  require_labels: true
  min_improvement_delta: 0.01
```

Programmatic configuration:

```python
from dqal import DQAL, DQALConfig

config = DQALConfig(
    abstain_threshold=0.45,
    flag_threshold=0.75,
    deadband=0.05,
    consecutive_batches=3
)
dqal = DQAL(model=model, config=config)
```

---

## 🧪 Testing

Run the full automated test suite:

```bash
pytest -v
```

Execute the 60-batch benchmark simulation:

```bash
python benchmarks/run_benchmark.py
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
