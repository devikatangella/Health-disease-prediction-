"""
Interactive Command Line Interface (CLI) for Smart Healthcare Disease Prediction System
Allows clinicians, researchers, and interviewers to run disease risk assessments directly in the Terminal.
Safe for all terminal encodings (Windows CP1252, UTF-8, etc.)
"""

import sys
import os
import json

# Ensure stdout handles unicode gracefully on all Windows terminal encodings
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import (
    HEART_ALL_FEATURES, HEART_FEATURE_METADATA,
    DIABETES_ALL_FEATURES, DIABETES_FEATURE_METADATA,
    KIDNEY_ALL_FEATURES, KIDNEY_FEATURE_METADATA,
    HEART_METRICS_PATH, DIABETES_METRICS_PATH, KIDNEY_METRICS_PATH
)
from src.predict import predict_disease
from database import get_recent_predictions, get_model_metadata_history

def print_banner():
    print("=" * 72)
    print("  SMART HEALTHCARE DISEASE PREDICTION SYSTEM - TERMINAL CLI")
    print("  AI Clinical Decision Support Demonstration (Machine Learning & SHAP)")
    print("=" * 72)
    print("  [!] DISCLAIMER: For Educational, Research & Portfolio Demonstration Only.")
    print("  Does NOT constitute medical advice or certified clinical diagnosis.")
    print("=" * 72)

def prompt_features(disease_key: str, feature_list: list, metadata_dict: dict) -> dict:
    """Interactively prompts user for features with sensible defaults."""
    print(f"\n--- Enter Clinical Parameters (Press [Enter] to use default value) ---")
    user_inputs = {}
    
    for feat in feature_list:
        meta = metadata_dict.get(feat, {})
        label = meta.get("label", feat)
        default_val = meta.get("default", 0)
        unit = f" ({meta['unit']})" if "unit" in meta else ""
        
        if "options" in meta:
            opts_str = ", ".join([f"{k}: {v}" for k, v in meta["options"].items()])
            prompt = f"  * {label}{unit} [{opts_str}] (default: {default_val}): "
        elif "min" in meta and "max" in meta:
            prompt = f"  * {label}{unit} [range {meta['min']} - {meta['max']}] (default: {default_val}): "
        else:
            prompt = f"  * {label}{unit} (default: {default_val}): "
            
        raw = input(prompt).strip()
        if not raw:
            user_inputs[feat] = default_val
        else:
            # Cast type appropriately
            if isinstance(default_val, int):
                try:
                    user_inputs[feat] = int(raw)
                except ValueError:
                    user_inputs[feat] = float(raw)
            elif isinstance(default_val, float):
                user_inputs[feat] = float(raw)
            else:
                user_inputs[feat] = raw.lower()
                
    return user_inputs

def display_prediction_results(result: dict):
    """Prints diagnostic assessment and SHAP explainability in the terminal."""
    print("\n" + "#" * 60)
    print(f"  DIAGNOSTIC ASSESSMENT RESULT: {result['disease_type'].upper()}")
    print("#" * 60)
    print(f"  * Prediction Status  : {result['prediction_label']}")
    print(f"  * Clinical Risk Tier : {result['risk_tier']}")
    print(f"  * Estimated Risk Prob: {result['probability_percent']}% (Probability: {result['probability']})")
    print(f"  * Audit Log ID       : #{result['log_id']} (Logged in SQLite: healthcare.db)")
    
    print("\n  [+] TOP INFLUENCING BIOMARKERS (SHAP Explainable AI):")
    print("  " + "-" * 56)
    print(f"  {'Biomarker Feature':<28} | {'Impact':<16} | {'SHAP Value'}")
    print("  " + "-" * 56)
    for factor in result.get("top_factors", []):
        feat = factor["feature"]
        impact = factor["impact"]
        val = f"{factor['shap_value']:+.4f}"
        print(f"  {feat:<28} | {impact:<16} | {val}")
    print("  " + "-" * 56)
    print("  * Note: Statistical attribution reflects model weights, not clinical causation.")
    print("#" * 60 + "\n")

def run_quick_demo():
    """Runs pre-configured clinical patient profiles through all 3 disease models."""
    print("\n" + "=" * 65)
    print("  RUNNING AUTOMATED MULTI-DISEASE CLINICAL DEMONSTRATION")
    print("=" * 65)
    
    # 1. Heart Disease Demo (High Risk Case)
    heart_sample = {
        "age": 63, "sex": 1, "cp": 4, "trestbps": 150, "chol": 270,
        "fbs": 1, "restecg": 2, "thalach": 125, "exang": 1,
        "oldpeak": 2.6, "slope": 2, "ca": 2, "thal": 7
    }
    print("\n[Case 1/3] Cardiovascular Disease Evaluation (63yo Male with Angina):")
    res_heart = predict_disease("heart", heart_sample)
    display_prediction_results(res_heart)
    
    # 2. Diabetes Demo (High Risk Case)
    diab_sample = {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 34, "Smoker": 1,
        "Stroke": 0, "HeartDiseaseorAttack": 1, "PhysActivity": 0, "Fruits": 0,
        "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
        "GenHlth": 4, "MentHlth": 8, "PhysHlth": 15, "DiffWalk": 1, "Sex": 1,
        "Age": 10, "Education": 4, "Income": 4
    }
    print("[Case 2/3] Diabetes Metabolic Risk Evaluation (Sedentary, BMI 34, HighBP):")
    res_diab = predict_disease("diabetes", diab_sample)
    display_prediction_results(res_diab)
    
    # 3. CKD Demo (Renal Impairment Case)
    ckd_sample = {
        "age": 58, "bp": 90, "sg": 1.010, "al": 3.0, "su": 1.0,
        "bgr": 180, "bu": 65, "sc": 3.4, "sod": 130, "pot": 5.2,
        "hemo": 9.4, "pcv": 28, "wbcc": 9800, "rbcc": 3.2,
        "rbc": "abnormal", "pc": "abnormal", "pcc": "present", "ba": "notpresent",
        "htn": "yes", "dm": "yes", "cad": "no", "appet": "poor", "pe": "yes", "ane": "yes"
    }
    print("[Case 3/3] Chronic Kidney Disease Evaluation (Elevated Creatinine 3.4, Anemia):")
    res_ckd = predict_disease("kidney", ckd_sample)
    display_prediction_results(res_ckd)

def show_benchmarks():
    """Displays benchmark results loaded from reports JSON."""
    print("\n" + "=" * 65)
    print("  PRODUCTION MACHINE LEARNING BENCHMARKS (Held-Out Test Sets)")
    print("=" * 65)
    
    files = [
        ("Heart Disease (UCI Cleveland)", HEART_METRICS_PATH),
        ("Diabetes (CDC BRFSS)", DIABETES_METRICS_PATH),
        ("Chronic Kidney Disease (UCI CKD)", KIDNEY_METRICS_PATH)
    ]
    
    for title, path in files:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            best_model = data.get("selected_model")
            benchmarks = data.get("benchmark_results", {})
            print(f"\n--- {title} | Selected Model: {best_model} ---")
            print(f"  {'Algorithm':<22} | {'Accuracy':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC'}")
            print("  " + "-" * 62)
            for m_name, m in benchmarks.items():
                acc = f"{m.get('accuracy', 0)*100:.1f}%"
                rec = f"{m.get('recall', 0)*100:.1f}%"
                f1 = f"{m.get('f1', 0):.4f}"
                auc = f"{m.get('roc_auc', 0):.4f}"
                print(f"  {m_name:<22} | {acc:<9} | {rec:<9} | {f1:<9} | {auc}")
        else:
            print(f"  Metrics file not found for {title}. Run train.py first.")
    print("=" * 65 + "\n")

def show_recent_logs():
    """Prints recent demonstration records from SQLite database."""
    print("\n" + "=" * 65)
    print("  RECENT DEMONSTRATION AUDIT LOGS (SQLite: healthcare.db)")
    print("=" * 65)
    logs = get_recent_predictions(limit=10)
    if not logs:
        print("  No demonstration logs found in database.")
    else:
        for log in logs:
            print(f"  [ID #{log['id']}] {log['timestamp']} | {log['disease_type']:<22} | Risk: {log['risk_category']:<12} | Prob: {log['probability']*100:.1f}%")
    print("=" * 65 + "\n")

def main():
    while True:
        print_banner()
        print("  Select an Option:")
        print("    [1] Heart Disease Risk Prediction (Interactive)")
        print("    [2] Diabetes Risk Prediction (Interactive)")
        print("    [3] Chronic Kidney Disease Prediction (Interactive)")
        print("    [4] Run Automated Quick Demo (All 3 Diseases)")
        print("    [5] View Model Performance Benchmark Table")
        print("    [6] View Recent SQLite Demonstration Logs")
        print("    [7] Exit")
        
        try:
            choice = input("\n  Enter choice [1-7]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break
            
        if choice == "1":
            inputs = prompt_features("heart", HEART_ALL_FEATURES, HEART_FEATURE_METADATA)
            res = predict_disease("heart", inputs)
            display_prediction_results(res)
        elif choice == "2":
            inputs = prompt_features("diabetes", DIABETES_ALL_FEATURES, DIABETES_FEATURE_METADATA)
            res = predict_disease("diabetes", inputs)
            display_prediction_results(res)
        elif choice == "3":
            inputs = prompt_features("kidney", KIDNEY_ALL_FEATURES, KIDNEY_FEATURE_METADATA)
            res = predict_disease("kidney", inputs)
            display_prediction_results(res)
        elif choice == "4":
            run_quick_demo()
        elif choice == "5":
            show_benchmarks()
        elif choice == "6":
            show_recent_logs()
        elif choice == "7":
            print("\nExiting Smart Healthcare System. Goodbye!\n")
            break
        else:
            print("\n[!] Invalid choice. Please enter a number between 1 and 7.\n")
            
        try:
            input("Press [Enter] to return to the main menu...")
        except (EOFError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    main()
