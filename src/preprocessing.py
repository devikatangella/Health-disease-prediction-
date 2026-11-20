"""
Data Preprocessing and Pipeline Construction Module
Implements leakage-free Scikit-learn Transformers and Pipelines.
Fitting occurs strictly on training subsets to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from typing import Tuple, List, Dict, Any, Optional

from src.config import (
    RANDOM_STATE,
    HEART_NUMERIC_FEATURES,
    HEART_CATEGORICAL_FEATURES,
    DIABETES_NUMERIC_FEATURES,
    DIABETES_CATEGORICAL_FEATURES,
    KIDNEY_NUMERIC_FEATURES,
    KIDNEY_CATEGORICAL_FEATURES
)

def split_data_stratified(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.20, 
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train-test split to preserve disease prevalence ratio.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def build_heart_preprocessor() -> ColumnTransformer:
    """
    Constructs ColumnTransformer for Heart Disease features.
    """
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, HEART_NUMERIC_FEATURES),
            ("cat", cat_transformer, HEART_CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

def build_diabetes_preprocessor() -> ColumnTransformer:
    """
    Constructs ColumnTransformer for CDC Diabetes Health Indicators.
    """
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, DIABETES_NUMERIC_FEATURES),
            ("cat", cat_transformer, DIABETES_CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

def build_kidney_preprocessor() -> ColumnTransformer:
    """
    Constructs ColumnTransformer for UCI Chronic Kidney Disease features.
    Uses RobustScaler on numeric labs to handle extreme lab outliers (e.g. serum creatinine).
    """
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, KIDNEY_NUMERIC_FEATURES),
            ("cat", cat_transformer, KIDNEY_CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
