"""
DQAL Real-Time Telemetry and Degradation Monitoring Dashboard (Streamlit).
Displays live quality scores, SERVE/FLAG/ABSTAIN decision bands, sub-signal breakdowns,
and retraining event histories.
"""

from __future__ import annotations
import os
import sqlite3
import pandas as pd
import numpy as np

try:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def load_telemetry_data(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(db_path):
        return pd.DataFrame(), pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        telemetry_df = pd.read_sql_query("SELECT * FROM predictions_telemetry ORDER BY id ASC", conn)
    except Exception:
        telemetry_df = pd.DataFrame()

    try:
        retrain_df = pd.read_sql_query("SELECT * FROM retrain_events ORDER BY id ASC", conn)
    except Exception:
        retrain_df = pd.DataFrame()

    conn.close()
    return telemetry_df, retrain_df


def main():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit or Plotly is not installed. Please install them to run the dashboard.")
        return

    st.set_page_config(
        page_title="DQAL — Data Quality Monitor",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .metric-card {
            background-color: #1e1e2f;
            border-radius: 10px;
            padding: 16px;
            border: 1px solid #2e2e42;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🛡️ DQAL — Data-Quality-Aware Learning</div>', unsafe_allow_html=True)
    st.markdown("Real-time inference quality scoring, gating (SERVE / FLAG / ABSTAIN), and selective retraining telemetry.")

    # Sidebar
    st.sidebar.title("⚙️ Telemetry Settings")
    db_path = st.sidebar.text_input("SQLite Database Path", value="dqal_telemetry.db")
    refresh_rate = st.sidebar.slider("Auto-refresh interval (s)", 1, 30, 5)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Decision Thresholds**")
    st.sidebar.markdown("- 🟢 **SERVE**: Q > 0.80")
    st.sidebar.markdown("- 🟡 **FLAG**: 0.50 < Q ≤ 0.80")
    st.sidebar.markdown("- 🔴 **ABSTAIN**: Q ≤ 0.50")

    telemetry_df, retrain_df = load_telemetry_data(db_path)

    if telemetry_df.empty:
        st.warning(f"No telemetry data found in `{db_path}`. Run a simulation or benchmark script first!")
        st.info("Run `python benchmarks/run_benchmark.py` to generate sample degradation telemetry.")
        return

    # Metrics row
    latest_row = telemetry_df.iloc[-1]
    latest_q = latest_row["Q"]
    latest_decision = latest_row["decision"]
    active_version = latest_row["model_version"]
    total_batches = len(telemetry_df)
    avg_q = telemetry_df["Q"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Batches", f"{total_batches}")
    col2.metric("Latest Quality Q", f"{latest_q:.3f}", delta=f"{latest_q - avg_q:+.3f} vs avg")
    
    state_color = "🟢" if latest_decision == "SERVE" else ("🟡" if latest_decision == "FLAG" else "🔴")
    col3.metric("Current Gating State", f"{state_color} {latest_decision}")
    col4.metric("Active Model Version", f"{active_version}")
    col5.metric("Retrain Events", f"{len(retrain_df)}")

    st.markdown("---")

    # Time-Series Chart with Bands
    st.subheader("📈 Quality Score Timeline with Decision Bands")

    fig = go.Figure()

    # Decision background bands
    fig.add_hrect(y0=0.8, y1=1.0, fillcolor="rgba(34, 197, 94, 0.12)", line_width=0, annotation_text="SERVE (Q > 0.8)", annotation_position="top left")
    fig.add_hrect(y0=0.5, y1=0.8, fillcolor="rgba(234, 179, 8, 0.12)", line_width=0, annotation_text="FLAG (0.5 < Q ≤ 0.8)", annotation_position="top left")
    fig.add_hrect(y0=0.0, y1=0.5, fillcolor="rgba(239, 68, 68, 0.12)", line_width=0, annotation_text="ABSTAIN (Q ≤ 0.5)", annotation_position="top left")

    # Q curve
    fig.add_trace(go.Scatter(
        x=telemetry_df["id"],
        y=telemetry_df["Q"],
        mode="lines+markers",
        name="Quality Score Q",
        line=dict(color="#6366f1", width=3),
        marker=dict(
            size=6,
            color=telemetry_df["decision"].map({"SERVE": "#22c55e", "FLAG": "#eab308", "ABSTAIN": "#ef4444"}),
        ),
    ))

    # Add retrain event vertical markers
    if not retrain_df.empty:
        for _, r in retrain_df.iterrows():
            promoted = bool(r["promoted"])
            line_color = "#10b981" if promoted else "#f43f5e"
            label = f"Retrain ({r['new_version']})" if promoted else "Retrain Rejected"
            fig.add_vline(
                x=len(telemetry_df), # Approximate marker if timestamp matches
                line_width=2,
                line_dash="dash",
                line_color=line_color,
                annotation_text=label,
            )

    fig.update_layout(
        xaxis_title="Batch Sequence #",
        yaxis_title="Quality Score Q ∈ [0, 1]",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Breakdown of sub-signals
    st.subheader("🔍 Sub-Signal Breakdown (Root-Cause Diagnosis)")

    col_chart, col_donut = st.columns([2, 1])

    with col_chart:
        fig_signals = go.Figure()
        fig_signals.add_trace(go.Scatter(
            x=telemetry_df["id"],
            y=telemetry_df["missing_signal"],
            mode="lines",
            name="Missingness Quality (1 - nulls)",
            line=dict(color="#38bdf8", width=2),
        ))
        fig_signals.add_trace(go.Scatter(
            x=telemetry_df["id"],
            y=telemetry_df["drift_signal"],
            mode="lines",
            name="Covariate Stability (1 - drift)",
            line=dict(color="#f43f5e", width=2),
        ))
        fig_signals.add_trace(go.Scatter(
            x=telemetry_df["id"],
            y=telemetry_df["outlier_signal"],
            mode="lines",
            name="Inlier Score (1 - noise)",
            line=dict(color="#fbbf24", width=2),
        ))
        fig_signals.update_layout(
            xaxis_title="Batch Sequence #",
            yaxis_title="Sub-Score Quality",
            yaxis=dict(range=[0, 1.05]),
            template="plotly_dark",
            height=320,
            margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig_signals, use_container_width=True)

    with col_donut:
        decision_counts = telemetry_df["decision"].value_counts()
        colors = {"SERVE": "#22c55e", "FLAG": "#eab308", "ABSTAIN": "#ef4444"}
        fig_donut = go.Figure(data=[go.Pie(
            labels=decision_counts.index,
            values=decision_counts.values,
            hole=.5,
            marker=dict(colors=[colors.get(x, "#888") for x in decision_counts.index]),
        )])
        fig_donut.update_layout(
            title="Decision Distribution",
            template="plotly_dark",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Telemetry Log Inspector
    st.subheader("📋 Inference Telemetry Logs")
    display_cols = ["id", "timestamp", "batch_id", "Q", "decision", "missing_signal", "drift_signal", "outlier_signal", "model_version", "n_samples"]
    st.dataframe(
        telemetry_df[display_cols].sort_values(by="id", ascending=False),
        use_container_width=True,
        height=260,
    )

    if not retrain_df.empty:
        st.subheader("🔄 Selective Retraining & Model Promotion History")
        st.dataframe(retrain_df, use_container_width=True)


if __name__ == "__main__":
    main()
