"""
Data Loading and Validation Module for Smart Healthcare Disease Prediction System
Loads authentic datasets for:
1. Heart Disease (UCI Cleveland)
2. Diabetes Health Indicators (CDC BRFSS)
3. Chronic Kidney Disease (UCI CKD)
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from src.config import (
    HEART_DATA_PATH,
    DIABETES_DATA_PATH,
    KIDNEY_DATA_PATH,
    HEART_ALL_FEATURES,
    DIABETES_ALL_FEATURES,
    KIDNEY_ALL_FEATURES
)

def load_heart_data(csv_path: str = HEART_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads UCI Heart Disease (Cleveland) dataset.
    Converts multi-class target (0 = absence, 1, 2, 3, 4 = presence)
    into standard documented binary target:
        0: No Heart Disease
        1: Heart Disease Present
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Heart disease dataset not found at {csv_path}. Run download_datasets.py first.")
        
    df = pd.read_csv(csv_path)
    
    # Verify expected columns
    expected_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in heart dataset: {col}")
            
    # Binary conversion of target as documented in UCI repository:
    # 0 = absence of disease, values 1-4 indicate presence
    y = (df["target"] > 0).astype(int)
    X = df[HEART_ALL_FEATURES].copy()
    
    return X, y

def load_diabetes_data(csv_path: str = DIABETES_DATA_PATH, sample_size: int = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads CDC Diabetes Health Indicators dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Diabetes dataset not found at {csv_path}. Run download_datasets.py first.")
        
    df = pd.read_csv(csv_path)
    
    target_col = "target" if "target" in df.columns else "Diabetes_binary"
    if target_col not in df.columns:
        raise ValueError("Missing diabetes target column.")
        
    if sample_size and sample_size < len(df):
        # Stratified sample for rapid tuning or SHAP background
        pos = df[df[target_col] == 1].sample(n=min(sample_size // 2, (df[target_col] == 1).sum()), random_state=42)
        neg = df[df[target_col] == 0].sample(n=min(sample_size // 2, (df[target_col] == 0).sum()), random_state=42)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        
    y = df[target_col].astype(int)
    X = df[DIABETES_ALL_FEATURES].copy()
    
    return X, y

def load_kidney_data(csv_path: str = KIDNEY_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads UCI Chronic Kidney Disease (CKD) dataset.
    Cleans messy string labels (e.g. 'ckd\t', 'notckd', extra whitespace, '\tno', '\tyes').
    Converts target to binary:
        0: notckd
        1: ckd
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Kidney disease dataset not found at {csv_path}. Run download_datasets.py first.")
        
    df = pd.read_csv(csv_path)
    
    target_col = "target" if "target" in df.columns else "class"
    if target_col not in df.columns:
        raise ValueError("Missing kidney target column.")
        
    # Clean target string column
    y_raw = df[target_col].astype(str).str.strip().str.lower()
    y_clean = y_raw.apply(
        lambda val: 1 if "ckd" in val and "notckd" not in val else (0 if "notckd" in val else np.nan)
    )
    
    valid_idx = y_clean.dropna().index
    y = y_clean.loc[valid_idx].astype(int)
    X = df[KIDNEY_ALL_FEATURES].loc[valid_idx].copy()
    
    # Clean string columns: strip tabs and spaces, normalize values
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype).startswith("str"):
            X[col] = (
                X[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace("\t", "", regex=False)
                .replace({"nan": np.nan, "none": np.nan, "?": np.nan, "": np.nan})
            )
            # Try converting to numeric if it contains numbers
            try:
                X[col] = pd.to_numeric(X[col])
            except (ValueError, TypeError):
                pass
                
    return X, y

def get_dataset_summary(disease_type: str) -> Dict[str, Any]:
    """Provides metadata summary for dataset exploration."""
    if disease_type == "heart":
        X, y = load_heart_data()
        name = "UCI Cleveland Heart Disease"
    elif disease_type == "diabetes":
        X, y = load_diabetes_data()
        name = "CDC Diabetes Health Indicators"
    elif disease_type == "kidney":
        X, y = load_kidney_data()
        name = "UCI Chronic Kidney Disease"
    else:
        raise ValueError(f"Unknown disease type: {disease_type}")
        
    return {
        "name": name,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "target_distribution": y.value_counts(normalize=True).to_dict(),
        "missing_values": X.isnull().sum().to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in X.dtypes.items()}
    }
