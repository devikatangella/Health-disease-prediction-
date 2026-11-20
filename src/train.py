"""
Model Training, Cross-Validation, Hyperparameter Tuning, and Pipeline Persistence Module
Supports:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier
For 3 Diseases:
1. Heart Disease (UCI Cleveland)
2. Diabetes Health Indicators (CDC BRFSS)
3. Chronic Kidney Disease (UCI CKD)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE,
    HEART_DATA_PATH,
    HEART_MODEL_PATH,
    HEART_METRICS_PATH,
    DIABETES_DATA_PATH,
    DIABETES_MODEL_PATH,
    DIABETES_METRICS_PATH,
    KIDNEY_DATA_PATH,
    KIDNEY_MODEL_PATH,
    KIDNEY_METRICS_PATH,
    REPORTS_DIR
)
from src.data_loader import load_heart_data, load_diabetes_data, load_kidney_data
from src.preprocessing import (
    split_data_stratified,
    build_heart_preprocessor,
    build_diabetes_preprocessor,
    build_kidney_preprocessor
)
from src.evaluate import (
    compute_evaluation_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_model_comparison
)
from database import log_model_metadata

def get_candidate_models(disease_type: str = "heart") -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Defines candidate algorithms and their hyperparameter search grids.
    Adjusted slightly by dataset size for optimal performance.
    """
    if disease_type == "diabetes":
        # For large CDC dataset (70k samples), keep grids focused and computationally efficient
        return {
            "Logistic Regression": (
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                {
                    "classifier__C": [0.1, 1.0],
                    "classifier__solver": ["lbfgs"]
                }
            ),
            "Decision Tree": (
                DecisionTreeClassifier(random_state=RANDOM_STATE),
                {
                    "classifier__max_depth": [6, 10],
                    "classifier__min_samples_split": [10, 20]
                }
            ),
            "Random Forest": (
                RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
                {
                    "classifier__min_samples_split": [5, 10]
                }
            ),
            "XGBoost": (
                XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
                {
                    "classifier__n_estimators": [100],
                    "classifier__max_depth": [4, 6],
                    "classifier__learning_rate": [0.05, 0.1]
                }
            )
        }
    else:
        # For Heart (303 rows) and Kidney (400 rows)
        return {
            "Logistic Regression": (
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                {
                    "classifier__C": [0.01, 0.1, 1.0, 10.0],
                    "classifier__solver": ["lbfgs"]
                }
            ),
            "Decision Tree": (
                DecisionTreeClassifier(random_state=RANDOM_STATE),
                {
                    "classifier__max_depth": [3, 5, 8, None],
                    "classifier__min_samples_split": [2, 5, 10],
                    "classifier__criterion": ["gini", "entropy"]
                }
            ),
            "Random Forest": (
                RandomForestClassifier(random_state=RANDOM_STATE),
                {
                    "classifier__n_estimators": [50, 100, 200],
                    "classifier__max_depth": [4, 6, 10, None],
                    "classifier__min_samples_split": [2, 5]
                }
            ),
            "XGBoost": (
                XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
                {
                    "classifier__n_estimators": [50, 100, 150],
                    "classifier__max_depth": [3, 4, 6],
                    "classifier__learning_rate": [0.01, 0.05, 0.1],
                    "classifier__subsample": [0.8, 1.0]
                }
            )
        }

def _train_module_generic(
    disease_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    model_save_path: str,
    metrics_save_path: str,
    reports_subfolder: str,
    class_names: Tuple[str, str] = ("Negative", "Positive"),
    disease_key: str = "heart"
) -> Dict[str, Any]:
    """Helper to train, tune, evaluate, and save any disease model."""
    print("\n" + "=" * 65)
    print(f"TRAINING MODULE: {disease_name.upper()}")
    print("=" * 65)
    print(f"Dataset Size: {X.shape[0]} samples, {X.shape[1]} features.")
    print("Class Balance:", y.value_counts(normalize=True).to_dict())

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = split_data_stratified(X, y, test_size=0.20, random_state=RANDOM_STATE)
    print(f"Train Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples (Untouched)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = get_candidate_models(disease_type=disease_key)
    benchmark_results = {}
    fitted_pipelines = {}
    
    reports_dir = os.path.join(REPORTS_DIR, reports_subfolder)
    os.makedirs(reports_dir, exist_ok=True)
    
    best_model_name = None
    best_score = -1.0
    
    for model_name, (classifier, param_grid) in candidates.items():
        print(f"\n--- Tuning & Training: {model_name} ---")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        fitted_pipelines[model_name] = best_pipeline
        
        # Test Set Evaluation
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1] if hasattr(best_pipeline, "predict_proba") else None
        
        metrics = compute_evaluation_metrics(y_test, y_pred, y_prob)
        metrics["best_cv_roc_auc"] = round(float(grid_search.best_score_), 4)
        metrics["best_params"] = {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}
        benchmark_results[model_name] = metrics
        
        print(f"Test Accuracy: {metrics['accuracy']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Optimal Hyperparameters: {metrics['best_params']}")
        
        # Log into SQLite registry
        log_model_metadata(
            disease_type=disease_name,
            model_name=model_name,
            metrics=metrics,
            hyperparameters=metrics["best_params"]
        )
        
        # Save plots
        plot_confusion_matrix(
            y_test, y_pred,
            class_names=[class_names[0], class_names[1]],
            title=f"{model_name} - {disease_name} Confusion Matrix",
            save_path=os.path.join(reports_dir, f"{model_name.lower().replace(' ', '_')}_cm.png")
        )
        
        if y_prob is not None:
            plot_roc_curve(
                y_test, y_prob,
                model_name=model_name,
                title=f"{model_name} - {disease_name} ROC Curve",
                save_path=os.path.join(reports_dir, f"{model_name.lower().replace(' ', '_')}_roc.png")
            )
            plot_precision_recall_curve(
                y_test, y_prob,
                model_name=model_name,
                title=f"{model_name} - {disease_name} PR Curve",
                save_path=os.path.join(reports_dir, f"{model_name.lower().replace(' ', '_')}_pr.png")
            )
            
        composite_score = metrics["roc_auc"] * 0.6 + metrics["f1"] * 0.4
        if composite_score > best_score:
            best_score = composite_score
            best_model_name = model_name
            
    print(f"\nModel Selection for {disease_name}: Selected {best_model_name} (Composite Score: {best_score:.4f})")
    
    # Save overall comparison chart
    plot_model_comparison(
        benchmark_results,
        title=f"{disease_name} Algorithm Benchmark Comparison",
        save_path=os.path.join(reports_dir, "model_comparison.png")
    )
    
    # Save best pipeline
    final_pipeline = fitted_pipelines[best_model_name]
    joblib.dump(final_pipeline, model_save_path)
    print(f"Selected pipeline saved to: {model_save_path}")
    
    # Save metrics JSON
    with open(metrics_save_path, "w") as f:
        json.dump({
            "disease_name": disease_name,
            "selected_model": best_model_name,
            "benchmark_results": benchmark_results,
            "best_model_metrics": benchmark_results[best_model_name]
        }, f, indent=4)
    print(f"Metrics saved to: {metrics_save_path}")
    
    return {
        "disease_name": disease_name,
        "selected_model": best_model_name,
        "metrics": benchmark_results[best_model_name],
        "all_benchmarks": benchmark_results
    }

def train_heart_disease_module() -> Dict[str, Any]:
    X, y = load_heart_data(HEART_DATA_PATH)
    preprocessor = build_heart_preprocessor()
    return _train_module_generic(
        disease_name="Heart Disease",
        X=X,
        y=y,
        preprocessor=preprocessor,
        model_save_path=HEART_MODEL_PATH,
        metrics_save_path=HEART_METRICS_PATH,
        reports_subfolder="heart",
        class_names=("No Heart Disease", "Heart Disease Present"),
        disease_key="heart"
    )

def train_diabetes_module() -> Dict[str, Any]:
    X, y = load_diabetes_data(DIABETES_DATA_PATH)
    preprocessor = build_diabetes_preprocessor()
    return _train_module_generic(
        disease_name="Diabetes",
        X=X,
        y=y,
        preprocessor=preprocessor,
        model_save_path=DIABETES_MODEL_PATH,
        metrics_save_path=DIABETES_METRICS_PATH,
        reports_subfolder="diabetes",
        class_names=("No Diabetes", "Diabetes Risk"),
        disease_key="diabetes"
    )

def train_kidney_disease_module() -> Dict[str, Any]:
    X, y = load_kidney_data(KIDNEY_DATA_PATH)
    preprocessor = build_kidney_preprocessor()
    return _train_module_generic(
        disease_name="Chronic Kidney Disease",
        X=X,
        y=y,
        preprocessor=preprocessor,
        model_save_path=KIDNEY_MODEL_PATH,
        metrics_save_path=KIDNEY_METRICS_PATH,
        reports_subfolder="kidney",
        class_names=("No CKD", "CKD Present"),
        disease_key="kidney"
    )

def train_all_modules():
    print("\n" + "#" * 70)
    print("STARTING COMPLETE TRAINING PIPELINE FOR ALL 3 DISEASES")
    print("#" * 70)
    
    heart_res = train_heart_disease_module()
    diabetes_res = train_diabetes_module()
    kidney_res = train_kidney_disease_module()
    
    print("\n" + "#" * 70)
    print("ALL 3 DISEASE PIPELINES SUCCESSFULLY TRAINED, EVALUATED, AND PERSISTED!")
    print("#" * 70)
    return {
        "heart": heart_res,
        "diabetes": diabetes_res,
        "kidney": kidney_res
    }

if __name__ == "__main__":
    train_all_modules()
