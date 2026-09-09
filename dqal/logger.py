"""
Telemetry Logger for DQAL.
Stores quality scores, sub-signals, gating decisions, and retrain events in SQLite.
Enforces privacy safeguards (never logs raw features unless explicitly enabled).
"""

from __future__ import annotations
import sqlite3
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np


class TelemetryLogger:
    """
    SQLite telemetry logger for tracking DQAL health across prediction batches.
    """

    def __init__(self, db_path: str = "dqal_telemetry.db", log_raw_features: bool = False):
        self.db_path = db_path
        self.log_raw_features = log_raw_features
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create telemetry tables if they don't already exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    Q REAL NOT NULL,
                    missing_signal REAL NOT NULL,
                    drift_signal REAL NOT NULL,
                    outlier_signal REAL NOT NULL,
                    missing_rate REAL NOT NULL,
                    drift_magnitude REAL NOT NULL,
                    outlier_rate REAL NOT NULL,
                    decision TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    n_samples INTEGER NOT NULL,
                    raw_features_json TEXT,
                    metadata_json TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retrain_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    old_version TEXT NOT NULL,
                    new_version TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    val_score_old REAL NOT NULL,
                    val_score_new REAL NOT NULL,
                    promoted INTEGER NOT NULL,
                    details_json TEXT
                )
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_time ON predictions_telemetry (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_batch_decision ON predictions_telemetry (decision)")
            conn.commit()

    def log_batch(
        self,
        batch_id: str,
        Q: float,
        signals: Dict[str, float],
        decision: str,
        model_version: str,
        n_samples: int,
        raw_features: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a batch evaluation in the SQLite telemetry database.
        """
        now_ts = datetime.now(timezone.utc).isoformat()

        raw_json: Optional[str] = None
        if self.log_raw_features and raw_features is not None:
            if isinstance(raw_features, pd.DataFrame):
                raw_json = raw_features.to_json(orient="records")
            elif isinstance(raw_features, np.ndarray):
                raw_json = json.dumps(raw_features.tolist())

        meta_json = json.dumps(extra_meta or {})

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions_telemetry (
                    timestamp, batch_id, Q, missing_signal, drift_signal, outlier_signal,
                    missing_rate, drift_magnitude, outlier_rate, decision,
                    model_version, n_samples, raw_features_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_ts,
                str(batch_id),
                float(Q),
                float(signals.get("missing_quality", 1.0)),
                float(signals.get("drift_quality", 1.0)),
                float(signals.get("outlier_quality", 1.0)),
                float(signals.get("missing_rate", 0.0)),
                float(signals.get("drift_magnitude", 0.0)),
                float(signals.get("outlier_rate", 0.0)),
                str(decision),
                str(model_version),
                int(n_samples),
                raw_json,
                meta_json,
            ))
            conn.commit()

    def log_retrain_event(
        self,
        old_version: str,
        new_version: str,
        trigger_reason: str,
        val_score_old: float,
        val_score_new: float,
        promoted: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record retraining attempt and validation outcome."""
        now_ts = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO retrain_events (
                    timestamp, old_version, new_version, trigger_reason,
                    val_score_old, val_score_new, promoted, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_ts,
                str(old_version),
                str(new_version),
                str(trigger_reason),
                float(val_score_old),
                float(val_score_new),
                1 if promoted else 0,
                json.dumps(details or {}),
            ))
            conn.commit()

    def get_telemetry_df(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch logged telemetry as a pandas DataFrame."""
        with self._get_connection() as conn:
            query = "SELECT * FROM predictions_telemetry ORDER BY id ASC"
            if limit:
                query += f" LIMIT {limit}"
            df = pd.read_sql_query(query, conn)
            return df

    def get_retrain_events_df(self) -> pd.DataFrame:
        """Fetch logged retrain events."""
        with self._get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM retrain_events ORDER BY id ASC", conn)

    def count_flagged_or_abstained_since(self, since_id: int = 0) -> Tuple[int, int]:
        """Count (flagged_batches, abstained_batches) since a given log ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN decision = 'FLAG' THEN 1 ELSE 0 END) as flags,
                    SUM(CASE WHEN decision = 'ABSTAIN' THEN 1 ELSE 0 END) as abstains
                FROM predictions_telemetry WHERE id > ?
            """, (since_id,))
            row = cursor.fetchone()
            flags = row[0] or 0
            abstains = row[1] or 0
            return int(flags), int(abstains)

    def clear(self) -> None:
        """Clear all tables (for clean benchmark/testing runs)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions_telemetry")
            cursor.execute("DELETE FROM retrain_events")
            conn.commit()
