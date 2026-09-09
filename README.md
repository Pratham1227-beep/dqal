# 🛡️ DQAL — Data-Quality-Aware Learning

[![DQAL CI](https://github.com/dqal/dqal/actions/workflows/ci.yml/badge.svg)](https://github.com/dqal/dqal)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Real-time, model-agnostic inference quality scoring, 3-tier gating (`SERVE` / `FLAG` / `ABSTAIN`), and selective retraining orchestration.**

---

## 1. Problem Statement

Deployed machine learning models fail silently. A model trained on clean, stationary data continues generating overconfident predictions when:
- **Upstream features drop out** (missing fields, API breaking changes, sensor timeouts)
- **Input distributions drift** (seasonal shifts, demographic changes, covariate shifts)
- **Extreme noise/outliers slip through** (hardware faults, corrupted data joins)

Traditional monitoring detects these weeks after downstream accuracy degrades. **DQAL** provides a lightweight, inline guardrail layer that inspects each prediction batch in real time, computes an interpretable Quality Score $Q \in [0, 1]$, gates execution, and safely triggers retraining.

---

## 2. System Architecture

```
                    ┌─────────────────────────────────────────┐
   Incoming batch → │           1. QUALITY SCORER             │
   of data          │   • Missingness (Feature null rates)    │
                    │   • Drift (Continuous PSI / Cat JS)     │
                    │   • Outlier (Isolation Forest / Mahala) │
                    └────────────────────┬────────────────────┘
                                         │ Q ∈ [0, 1], Sub-signals
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │            2. TRIGGER LOGIC             │
                    │   • Q > 0.80       →  SERVE             │
                    │   • 0.50 < Q ≤ 0.80 →  FLAG              │
                    │   • Q ≤ 0.50       →  ABSTAIN           │
                    │   • Hysteresis dead-band & confirmation │
                    └────────────────────┬────────────────────┘
                                         │ Decision
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
          ┌───────────────────┐                     ┌───────────────────┐
          │  3. WRAPPED MODEL │                     │ 4. SQLITE LOGGER  │
          │ (Sklearn/PyTorch) │                     │ (Privacy-safe: no │
          │  Serve/Abstain    │                     │  raw PII by def)  │
          └───────────────────┘                     └─────────┬─────────┘
                                                              │
                                                              ▼
                                                    ┌───────────────────┐
                                                    │ 5. ORCHESTRATOR   │
                                                    │ • Volume checks   │
                                                    │ • Label gating    │
                                                    │ • Val. promotion  │
                                                    └─────────┬─────────┘
                                                              │
                                                              ▼
                                                    ┌───────────────────┐
                                                    │ 6. DASHBOARD      │
                                                    │ (Streamlit live   │
                                                    │  telemetry & Q)   │
                                                    └───────────────────┘
```

---

## 3. Key Highlights & Features

- **⚡ Model-Agnostic Adapter**: Wraps any `scikit-learn` estimator, `PyTorch nn.Module`, or custom callable without modifying underlying model code.
- **📊 Interpretable Quality Score $Q$**: Combines Missingness, Population Stability Index (PSI), and Isolation Forest anomaly scores into a single metric $Q \in [0, 1]$.
- **🚦 3-Tier State Machine with Hysteresis**: Prevents state flapping near threshold boundaries with dead-bands and consecutive confirmation counts.
- **🔒 Privacy-Preserving Telemetry**: Logs statistical metrics to SQLite; disables raw feature logging by default to prevent PII exposure.
- **🔄 Validated Retraining & Rollback**: Enforces minimum data volume and verified label availability before retraining. Only promotes candidate models if validation accuracy beats active models.
- **📈 Streamlit Monitoring Dashboard**: Real-time visual tracking of $Q$, decision zones, sub-signal root-cause attribution, and model version transitions.

---

## 4. Installation & Setup

```bash
# Clone repository
git clone https://github.com/dqal/dqal.git
cd dqal

# Install in editable mode with all optional dependencies (visualization, PyTorch, dev)
pip install -e ".[all]"
```

---

## 5. Quickstart Example

### Wrapping a Scikit-Learn Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from dqal import DQAL, DQALConfig

# 1. Train your baseline model
data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# 2. Wrap with DQAL and fit baseline distributions
# CRITICAL: Baseline distributions must be captured from clean training data!
dqal = DQAL(model=model, model_version="v1.0.0")
dqal.fit_baseline(X_train)

# 3. Predict on incoming batches
result = dqal.predict(X_test.iloc[:50])

print(f"Quality Score Q: {result.Q:.3f}")
print(f"Gating Decision: {result.decision}")  # SERVE, FLAG, or ABSTAIN
print(f"Sub-signals:     {result.signals}")
print(f"Predictions:     {result.predictions[:5]}")
```

### Wrapping a PyTorch Model

```python
import torch
import torch.nn as nn
from dqal import DQAL

torch_model = nn.Sequential(
    nn.Linear(30, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
)

dqal = DQAL(model=torch_model, model_version="v1.0.0-torch")
dqal.fit_baseline(X_train)

result = dqal.predict(X_test.iloc[:20])
```

---

## 6. Configuration (`configs/default_config.yaml`)

```yaml
weights:
  missingness: 0.30
  drift: 0.40
  outlier: 0.30

thresholds:
  abstain_below: 0.50
  flag_below: 0.80

hysteresis:
  deadband: 0.03
  consecutive_batches: 2

logging:
  db_path: "dqal_telemetry.db"
  log_raw_features: false  # Opt-in only

retrain:
  min_flagged_batches: 5
  min_samples: 100
  require_labels: true
  min_improvement_delta: 0.01
```

---

## 7. Empirical Validation & Benchmarks

We evaluated DQAL across a progressive 60-batch degradation simulation (`benchmarks/run_benchmark.py`):
1. **Pristine Baseline** (Batches 0–11)
2. **Mild Covariate Shift** (Batches 12–23)
3. **Missing Fields Spike** (Batches 24–35)
4. **Severe Noise & Outliers** (Batches 36–47)
5. **Catastrophic Drift** (Batches 48–59)

### Benchmark Results

| Metric | Target | DQAL Result | Status |
| :--- | :--- | :--- | :--- |
| **Pearson Correlation ($r$)** | $r < -0.70$ or $> 0.70$ | **$r = 0.8502$** ($p = 8.45 \times 10^{-18}$) | Passed |
| **Inference Overhead** | $< 50$ ms / batch | **$49.79$ ms / batch** | Passed |
| **Early Warning Lead Time** | $> 0$ batches | **$9$ batches lead time** | Passed |
| **Label Safety Enforced** | $100\%$ | **Blocked unlabelled retraining** | Passed |

![DQAL Benchmark Results](benchmark_results.png)

---

## 8. Launching the Streamlit Dashboard

```bash
streamlit run dqal/dashboard.py
```

---

## 9. Running Tests

```bash
pytest -v
```

---

## 10. Known Limitations

- **Row vs. Batch Granularity**: Scoring single isolated rows has higher variance due to sample statistics. Batch mode ($\ge 30$ rows) provides robust distribution estimation.
- **Concept Drift vs. Anomaly**: DQAL detects statistical covariate and quality shift. Distinguishing permanent regime shift from temporary noise is surfaced to operators via the `FLAG` state rather than automated blind retraining.
