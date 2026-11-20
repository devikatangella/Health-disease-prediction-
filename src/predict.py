"""
Real-Time Model Inference, Risk Stratification, and Explainability Orchestrator
Supports:
1. Heart Disease Prediction (UCI Cleveland)
2. Diabetes Risk Prediction (CDC BRFSS)
3. Chronic Kidney Disease Prediction (UCI CKD)
Provides:
- Schema validation & range compliance
- Probability estimation
- Multi-tier clinical risk classification
- SHAP feature attributions
- Privacy-preserving SQLite demonstration logging
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from src.config import (
    HEART_MODEL_PATH,
    HEART_DATA_PATH,
    HEART_ALL_FEATURES,
    HEART_FEATURE_METADATA,
    DIABETES_MODEL_PATH,
    DIABETES_DATA_PATH,
    DIABETES_ALL_FEATURES,
    DIABETES_FEATURE_METADATA,
    KIDNEY_MODEL_PATH,
    KIDNEY_DATA_PATH,
    KIDNEY_ALL_FEATURES,
    KIDNEY_FEATURE_METADATA
)
from src.data_loader import load_heart_data, load_diabetes_data, load_kidney_data
from src.explainability import DiseaseModelExplainer
from database import log_prediction

# Global lazy cache for models and explainers
_LOADED_MODELS: Dict[str, Any] = {}
_LOADED_EXPLAINERS: Dict[str, Any] = {}

def get_pipeline_and_explainer(disease_type: str) -> Tuple[Any, DiseaseModelExplainer]:
    """Retrieves or lazily loads pipeline and SHAP explainer for the chosen disease."""
    disease_type = disease_type.lower()
    if disease_type in _LOADED_MODELS:
        return _LOADED_MODELS[disease_type], _LOADED_EXPLAINERS[disease_type]
        
    if disease_type == "heart":
        model_path = HEART_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train models first.")
        pipeline = joblib.load(model_path)
        X_bg, _ = load_heart_data(HEART_DATA_PATH)
        explainer = DiseaseModelExplainer(pipeline, X_bg, feature_names=HEART_ALL_FEATURES)
        
    elif disease_type == "diabetes":
        model_path = DIABETES_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train models first.")
        pipeline = joblib.load(model_path)
        # Sample 200 background instances for efficient SHAP computation
        X_bg, _ = load_diabetes_data(DIABETES_DATA_PATH, sample_size=200)
        explainer = DiseaseModelExplainer(pipeline, X_bg, feature_names=DIABETES_ALL_FEATURES)
        
    elif disease_type == "kidney":
        model_path = KIDNEY_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train models first.")
        pipeline = joblib.load(model_path)
        X_bg, _ = load_kidney_data(KIDNEY_DATA_PATH)
        explainer = DiseaseModelExplainer(pipeline, X_bg, feature_names=KIDNEY_ALL_FEATURES)
        
    else:
        raise ValueError(f"Unknown disease type: '{disease_type}'")
        
    _LOADED_MODELS[disease_type] = pipeline
    _LOADED_EXPLAINERS[disease_type] = explainer
    return pipeline, explainer

def validate_disease_input(disease_type: str, input_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validates inputs against disease schema, physiological ranges, and valid categories."""
    disease_type = disease_type.lower()
    if disease_type == "heart":
        features = HEART_ALL_FEATURES
        metadata = HEART_FEATURE_METADATA
    elif disease_type == "diabetes":
        features = DIABETES_ALL_FEATURES
        metadata = DIABETES_FEATURE_METADATA
    elif disease_type == "kidney":
        features = KIDNEY_ALL_FEATURES
        metadata = KIDNEY_FEATURE_METADATA
    else:
        return False, f"Unknown disease type: '{disease_type}'"
        
    for feat in features:
        if feat not in input_data:
            return False, f"Missing required parameter: '{feat}'"
            
        val = input_data[feat]
        meta = metadata.get(feat, {})
        
        # Numeric range validation
        if "min" in meta and "max" in meta:
            if val is not None and not pd.isna(val):
                try:
                    num_val = float(val)
                    if not (meta["min"] <= num_val <= meta["max"]):
                        return False, f"Value for '{meta.get('label', feat)}' ({val}) is outside plausible range [{meta['min']}, {meta['max']}]."
                except (ValueError, TypeError):
                    return False, f"Expected numeric value for '{feat}', got '{val}'."
                
        # Categorical options validation
        if "options" in meta:
            if val is not None and not pd.isna(val):
                opts = meta["options"]
                # Allow matching either keys or string representations
                valid_keys = list(opts.keys())
                str_keys = [str(k) for k in valid_keys]
                if val not in valid_keys and str(val) not in str_keys:
                    return False, f"Invalid value for '{meta.get('label', feat)}'. Allowed: {valid_keys}"
                
    return True, None

def predict_disease(
    disease_type: str,
    input_data: Dict[str, Any],
    allow_logging: bool = True
) -> Dict[str, Any]:
    """
    Unified disease prediction pipeline.
    """
    disease_type = disease_type.lower()
    is_valid, err = validate_disease_input(disease_type, input_data)
    if not is_valid:
        raise ValueError(err)
        
    pipeline, explainer = get_pipeline_and_explainer(disease_type)
    
    if disease_type == "heart":
        feature_order = HEART_ALL_FEATURES
        disease_title = "Heart Disease"
        label_pos = "Heart Disease Detected (High Risk)"
        label_neg = "Low Cardiovascular Risk"
    elif disease_type == "diabetes":
        feature_order = DIABETES_ALL_FEATURES
        disease_title = "Diabetes"
        label_pos = "Elevated Diabetes Risk"
        label_neg = "Low Diabetes Risk"
    elif disease_type == "kidney":
        feature_order = KIDNEY_ALL_FEATURES
        disease_title = "Chronic Kidney Disease"
        label_pos = "Chronic Kidney Disease Risk"
        label_neg = "Normal Renal Profile"
        
    # Build single-row DataFrame
    df_sample = pd.DataFrame([input_data])[feature_order]
    
    # Inference
    raw_pred = pipeline.predict(df_sample)[0]
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df_sample)[0]
        prob_positive = float(probs[1])
    else:
        prob_positive = 1.0 if raw_pred == 1 else 0.0
        
    # Clinical Risk Stratification
    if prob_positive < 0.35:
        risk_tier = "Low Risk"
        risk_color = "#10b981" # Green
        prediction_label = label_neg
    elif prob_positive < 0.65:
        risk_tier = "Moderate / Borderline Risk"
        risk_color = "#f59e0b" # Amber
        prediction_label = f"Borderline / {disease_title} Risk"
    else:
        risk_tier = "High Risk"
        risk_color = "#ef4444" # Red
        prediction_label = label_pos
        
    # Local Explainability (SHAP)
    top_factors = explainer.explain_instance(df_sample, top_k=5)
    
    # Audit Logging (Non-PII)
    log_id = log_prediction(
        disease_type=disease_title,
        input_dict=input_data,
        prediction_label=prediction_label,
        probability=prob_positive,
        risk_category=risk_tier,
        top_factors=top_factors,
        allow_logging=allow_logging
    )
    
    return {
        "disease_type": disease_title,
        "disease_key": disease_type,
        "prediction_label": prediction_label,
        "raw_prediction": int(raw_pred),
        "probability": round(prob_positive, 4),
        "probability_percent": round(prob_positive * 100, 1),
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "top_factors": top_factors,
        "log_id": log_id,
        "input_summary": input_data,
        "disclaimer": (
            "Educational and Research Demonstration Only: This prediction is generated by a statistical machine learning "
            "model trained on public benchmark research datasets. It does NOT constitute medical advice, diagnosis, or clinical evaluation. "
            "Always consult a licensed medical professional for personal health concerns."
        )
    }

# Backward compatibility alias
def predict_heart_disease(input_data: Dict[str, Any], allow_logging: bool = True) -> Dict[str, Any]:
    return predict_disease("heart", input_data, allow_logging)

def validate_heart_input(input_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    return validate_disease_input("heart", input_data)
