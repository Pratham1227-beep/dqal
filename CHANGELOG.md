# Changelog

All notable changes to DQAL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-09-09

### Added

- **Quality Scorer** with three sub-signals: Missingness, Drift (PSI/JS divergence), and Outlier (Isolation Forest / Mahalanobis).
- **3-tier gating state machine** (`PASSED` / `WARNING` / `BLOCKED`) with hysteresis dead-band and consecutive confirmation to prevent flapping.
- **Model-agnostic adapter** wrapping scikit-learn estimators, PyTorch `nn.Module`, and custom callables via `DQAL.predict()`.
- **Privacy-preserving SQLite telemetry logger** — logs statistical metrics only; raw feature logging disabled by default.
- **Retraining orchestrator** with minimum data volume, label verification, and validation-gated model promotion with rollback.
- **Learned Quality Scorer** for training a meta-model on historical quality signals.
- **Terminal telemetry reporting & SQLite logger** for real-time batch metrics, gating decisions, and selective retraining history.
- **60-batch degradation benchmark** (`benchmarks/run_benchmark.py`) validating correlation, latency, early warning, and label safety.
- **CI pipeline** via GitHub Actions across Python 3.10 / 3.11 / 3.12.
- **YAML-based configuration** (`configs/default_config.yaml`) for weights, thresholds, hysteresis, logging, and retrain policy.
