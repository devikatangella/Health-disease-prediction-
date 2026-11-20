"""
Re-export database methods within the src package.
"""
from database import (
    get_connection,
    init_database,
    log_prediction,
    log_model_metadata,
    log_feedback,
    get_recent_predictions,
    get_model_metadata_history,
    clear_prediction_logs,
    DEFAULT_DB_PATH
)

__all__ = [
    "get_connection",
    "init_database",
    "log_prediction",
    "log_model_metadata",
    "log_feedback",
    "get_recent_predictions",
    "get_model_metadata_history",
    "clear_prediction_logs",
    "DEFAULT_DB_PATH"
]
