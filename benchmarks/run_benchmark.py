"""
Benchmark script for DQAL.
Simulates production data degradation, measures Quality Score Q vs. Real Accuracy correlation,
and generates the benchmark visualization plot.
"""

from __future__ import annotations
import os
import sys
import time

# Ensure workspace root is in python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import seaborn as sns

from dqal.config import DQALConfig
from dqal.wrapper import DQAL
from dqal.orchestrator import RetrainOrchestrator
from benchmarks.simulator import DegradationSimulator, DegradationPhase


def run_full_benchmark(output_plot_path: str = "benchmark_results.png") -> None:
    print("=" * 70)
    print(">> Running DQAL Degradation & Quality Correlation Benchmark")
    print("=" * 70)

    # 1. Load Data
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test_pool, y_train, y_test_pool = train_test_split(
        X, y, test_size=0.6, random_state=42, stratify=y
    )

    print(f"Loaded dataset with {X.shape[1]} features. Train: {len(X_train)} rows, Test pool: {len(X_test_pool)} rows.")

    # 2. Train baseline model with imputer in pipeline for robust prediction
    baseline_model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    baseline_model.fit(X_train.to_numpy(), y_train)
    val_acc_initial = accuracy_score(y_test_pool, baseline_model.predict(X_test_pool.to_numpy()))
    print(f"Clean Baseline Model Validation Accuracy: {val_acc_initial:.4f}")

    # 3. Setup DQAL
    db_path = "benchmark_telemetry.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    config = DQALConfig()
    config.logging.db_path = db_path
    config.weights.missingness = 0.35
    config.weights.drift = 0.40
    config.weights.outlier = 0.25

    dqal = DQAL(model=baseline_model, config=config, model_version="v1.0.0")
    dqal.fit_baseline(X_train)
    print("Fitted DQAL baseline distributions on clean training data.")

    # 4. Generate Degradation Timeline
    simulator = DegradationSimulator(
        X_clean=X_test_pool,
        y_clean=y_test_pool,
        batch_size=40,
        random_state=42
    )

    phases = [
        DegradationPhase(name="Pristine (Clean)", num_batches=12, missing_prob=0.0, drift_mean_shift=0.0, noise_outlier_prob=0.0),
        DegradationPhase(name="Mild Sensor Drift", num_batches=12, missing_prob=0.03, drift_mean_shift=0.7, noise_outlier_prob=0.03),
        DegradationPhase(name="Missing Fields Spike", num_batches=12, missing_prob=0.25, drift_mean_shift=1.4, noise_outlier_prob=0.08),
        DegradationPhase(name="Severe Noise & Outliers", num_batches=12, missing_prob=0.35, drift_mean_shift=2.2, noise_outlier_prob=0.30, noise_scale=4.0),
        DegradationPhase(name="Catastrophic Drift", num_batches=12, missing_prob=0.55, drift_mean_shift=3.5, noise_outlier_prob=0.45, noise_scale=6.0),
    ]

    batches = simulator.generate_timeline(phases)
    print(f"Generated {len(batches)} sequential test batches across {len(phases)} degradation phases.")

    # 5. Execute Simulation
    results = []
    latencies = []

    for i, (batch_X, batch_y, phase_name) in enumerate(batches):
        t0 = time.perf_counter()
        res = dqal.predict(batch_X, batch_id=f"batch_{i:03d}")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        # Real accuracy of the model on this batch
        real_preds = baseline_model.predict(batch_X.to_numpy())
        real_acc = accuracy_score(batch_y, real_preds)

        results.append({
            "batch_idx": i,
            "phase": phase_name,
            "Q": res.Q,
            "real_accuracy": real_acc,
            "decision": res.decision,
            "missing_quality": res.signals["missing_quality"],
            "drift_quality": res.signals["drift_quality"],
            "outlier_quality": res.signals["outlier_quality"],
            "latency_ms": elapsed_ms,
        })

    df_res = pd.DataFrame(results)

    # 6. Compute Metrics
    # Pearson correlation between Q and accuracy
    r_val, p_val = pearsonr(df_res["Q"], df_res["real_accuracy"])
    avg_latency = np.mean(latencies)

    print("\n" + "=" * 70)
    print("[*] BENCHMARK METRIC SUMMARY")
    print("=" * 70)
    print(f"- Pearson Correlation (Q vs True Accuracy): r = {r_val:.4f} (p-value = {p_val:.2e})")
    print(f"- Average DQAL Inference Scoring Overhead: {avg_latency:.2f} ms / batch (Target: < 50ms)")
    print(f"- Gating Decision Breakdown:")
    for decision, count in df_res["decision"].value_counts().items():
        pct = (count / len(df_res)) * 100
        print(f"   * {decision:<8}: {count:2d} batches ({pct:.1f}%)")

    # Early-warning lead time calculation:
    # First batch where DQAL flags (WARNING or BLOCKED) vs first batch where accuracy drops below 80%
    first_flag_batch = df_res[df_res["decision"].isin(["FLAG", "WARNING", "ABSTAIN", "BLOCKED"])]["batch_idx"].min()
    first_acc_drop_batch = df_res[df_res["real_accuracy"] < 0.80]["batch_idx"].min()
    lead_time = first_acc_drop_batch - first_flag_batch

    print(f"- First DQAL Degradation Warning: Batch #{first_flag_batch}")
    print(f"- First True Accuracy Drop (<80%): Batch #{first_acc_drop_batch}")
    print(f"- Early Warning Lead Time: {lead_time} batch(es) ahead of severe failure!")

    # 7. Plotting Publication-Quality Artifact
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False, gridspec_kw={"height_ratios": [2.2, 1.8, 1.8]})

    # Panel 1: Timeline of Q vs Real Accuracy
    ax1 = axes[0]
    ax1.axhspan(0.8, 1.05, color="#22c55e", alpha=0.10, label="PASSED Zone (Q > 0.8)")
    ax1.axhspan(0.5, 0.8, color="#eab308", alpha=0.10, label="WARNING Zone (0.5 < Q <= 0.8)")
    ax1.axhspan(0.0, 0.5, color="#ef4444", alpha=0.10, label="BLOCKED Zone (Q <= 0.5)")

    ax1.plot(df_res["batch_idx"], df_res["Q"], color="#4f46e5", linewidth=2.8, marker="o", markersize=4, label="DQAL Quality Score Q", zorder=4)
    ax1.plot(df_res["batch_idx"], df_res["real_accuracy"], color="#059669", linewidth=2.4, linestyle="--", marker="s", markersize=4, label="True Model Accuracy", zorder=3)

    # Phase vertical lines
    phase_starts = df_res.groupby("phase")["batch_idx"].min().sort_values()
    for phase_name, start_idx in phase_starts.items():
        if start_idx > 0:
            ax1.axvline(x=start_idx - 0.5, color="#94a3b8", linestyle=":", linewidth=1.2)
            ax1.text(start_idx, 0.05, phase_name, fontsize=8, color="#475569", rotation=90, va="bottom")

    ax1.set_title(f"DQAL Real-Time Quality Score Q vs. Ground-Truth Model Accuracy (Pearson r = {r_val:.3f})", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Score / Accuracy", fontsize=10, fontweight="bold")
    ax1.set_ylim(-0.02, 1.08)
    ax1.set_xlim(-0.5, len(df_res) - 0.5)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    # Panel 2: Sub-signal Breakdown
    ax2 = axes[1]
    ax2.plot(df_res["batch_idx"], df_res["missing_quality"], color="#0284c7", linewidth=2.0, label="Missingness Quality (1 - nulls)")
    ax2.plot(df_res["batch_idx"], df_res["drift_quality"], color="#e11d48", linewidth=2.0, label="Covariate Stability (1 - PSI)")
    ax2.plot(df_res["batch_idx"], df_res["outlier_quality"], color="#d97706", linewidth=2.0, label="Inlier Score (1 - noise)")
    ax2.set_title("Sub-Signal Telemetry Breakdown (Root-Cause Attribution)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Sub-Score", fontsize=10, fontweight="bold")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlim(-0.5, len(df_res) - 0.5)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)

    # Panel 3: Correlation Scatter Plot
    ax3 = axes[2]
    scatter = ax3.scatter(
        df_res["Q"],
        df_res["real_accuracy"],
        c=df_res["batch_idx"],
        cmap="coolwarm",
        s=50,
        edgecolor="black",
        linewidth=0.5,
        zorder=3
    )
    # Regression line
    m, b = np.polyfit(df_res["Q"], df_res["real_accuracy"], 1)
    q_range = np.linspace(df_res["Q"].min(), df_res["Q"].max(), 100)
    ax3.plot(q_range, m * q_range + b, color="#b91c1c", linestyle="--", linewidth=2.0, label=f"Fit (Slope: {m:.2f}, r = {r_val:.3f})")

    ax3.set_title(f"Correlation: Quality Score Q tracks Accuracy Drop (r = {r_val:.3f}, p = {p_val:.1e})", fontsize=11, fontweight="bold")
    ax3.set_xlabel("DQAL Quality Score Q", fontsize=10, fontweight="bold")
    ax3.set_ylabel("True Ground-Truth Accuracy", fontsize=10, fontweight="bold")
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

    cbar = plt.colorbar(scatter, ax=ax3, orientation="vertical", pad=0.02)
    cbar.set_label("Batch Sequence #", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    print(f"\n[+] Benchmark plot successfully saved to: {output_plot_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_benchmark()
