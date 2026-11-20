"""
SQLite Database Layer for Smart Healthcare Disease Prediction System
Maintains:
1. prediction_logs: Non-identifiable educational demonstration records
2. model_metadata: Versioned registry of trained models, parameters, and benchmark metrics
3. user_feedback: Clinical demonstration feedback (Accept, Modify, Reject)

Strictly adheres to privacy: NO PII, no patient names, phone numbers, or identifiable data.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

DEFAULT_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "healthcare.db"))

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a thread-safe SQLite connection with dictionary-like row factory."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes tables for prediction logs, model metadata, and feedback."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # 1. Prediction Logs Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            disease_type TEXT NOT NULL,
            input_summary TEXT NOT NULL,
            prediction_label TEXT NOT NULL,
            probability REAL NOT NULL,
            risk_category TEXT NOT NULL,
            top_factors TEXT,
            logging_enabled INTEGER DEFAULT 1
        )
    """)
    
    # 2. Model Metadata Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease_type TEXT NOT NULL,
            model_name TEXT NOT NULL,
            accuracy REAL,
            precision_score REAL,
            recall_score REAL,
            f1_score REAL,
            roc_auc REAL,
            hyperparameters TEXT,
            trained_at TEXT NOT NULL
        )
    """)
    
    # 3. Demonstration Feedback Table (for interactive clinician review)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS demonstration_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_id INTEGER,
            disease_type TEXT NOT NULL,
            decision TEXT NOT NULL,
            clinician_notes TEXT,
            submitted_at TEXT NOT NULL,
            FOREIGN KEY (log_id) REFERENCES prediction_logs (id)
        )
    """)
    
    conn.commit()
    conn.close()

def log_prediction(
    disease_type: str,
    input_dict: Dict[str, Any],
    prediction_label: str,
    probability: float,
    risk_category: str,
    top_factors: Optional[List[Dict[str, Any]]] = None,
    allow_logging: bool = True,
    db_path: str = DEFAULT_DB_PATH
) -> Optional[int]:
    """
    Safely logs a non-identifiable prediction demonstration record using parameterized queries.
    Respects user privacy: if allow_logging is False, returns None immediately without persisting.
    """
    if not allow_logging:
        return None
        
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Serialize data cleanly
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    input_summary_json = json.dumps(input_dict)
    top_factors_json = json.dumps(top_factors) if top_factors else "[]"
    
    cursor.execute("""
        INSERT INTO prediction_logs (
            timestamp, disease_type, input_summary, prediction_label, probability, risk_category, top_factors, logging_enabled
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
    """, (
        timestamp,
        disease_type,
        input_summary_json,
        prediction_label,
        float(probability),
        risk_category,
        top_factors_json
    ))
    
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return log_id

def log_model_metadata(
    disease_type: str,
    model_name: str,
    metrics: Dict[str, float],
    hyperparameters: Optional[Dict[str, Any]] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Logs model training run results and evaluation metrics."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    params_json = json.dumps(hyperparameters) if hyperparameters else "{}"
    
    cursor.execute("""
        INSERT INTO model_metadata (
            disease_type, model_name, accuracy, precision_score, recall_score, f1_score, roc_auc, hyperparameters, trained_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        disease_type,
        model_name,
        metrics.get("accuracy", 0.0),
        metrics.get("precision", 0.0),
        metrics.get("recall", 0.0),
        metrics.get("f1", 0.0),
        metrics.get("roc_auc", 0.0),
        params_json,
        timestamp
    ))
    
    meta_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return meta_id

def log_feedback(
    log_id: Optional[int],
    disease_type: str,
    decision: str,
    notes: str = "",
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Logs clinician feedback (Accept / Modify / Reject) on demonstration predictions."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO demonstration_feedback (log_id, disease_type, decision, clinician_notes, submitted_at)
        VALUES (?, ?, ?, ?, ?)
    """, (log_id, disease_type, decision, notes, timestamp))
    
    fb_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return fb_id

def get_recent_predictions(limit: int = 20, disease_type: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent non-identifiable prediction demonstration records."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    if disease_type:
        cursor.execute("""
            SELECT * FROM prediction_logs 
            WHERE disease_type = ? 
            ORDER BY id DESC LIMIT ?
        """, (disease_type, limit))
    else:
        cursor.execute("""
            SELECT * FROM prediction_logs 
            ORDER BY id DESC LIMIT ?
        """, (limit,))
        
    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results

def get_model_metadata_history(disease_type: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves recorded training runs for models."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    if disease_type:
        cursor.execute("""
            SELECT * FROM model_metadata 
            WHERE disease_type = ? 
            ORDER BY id DESC
        """, (disease_type,))
    else:
        cursor.execute("""
            SELECT * FROM model_metadata 
            ORDER BY id DESC
        """)
        
    rows = cursor.fetchall()
    results = [dict(row) for row in rows]
    conn.close()
    return results

def clear_prediction_logs(db_path: str = DEFAULT_DB_PATH) -> None:
    """Clears demonstration logs for privacy or fresh demonstrations."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_logs")
    cursor.execute("DELETE FROM demonstration_feedback")
    conn.commit()
    conn.close()

# Auto-initialize database on import
init_database()
