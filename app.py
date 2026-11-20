"""
Smart Healthcare Disease Prediction System - Streamlit Dashboard
Production-grade multi-page clinical decision support demonstration interface.
Author: Senior Data Scientist & ML Engineer
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import io
from datetime import datetime

# Local modules
from src.config import (
    HEART_FEATURE_METADATA,
    DIABETES_FEATURE_METADATA,
    KIDNEY_FEATURE_METADATA,
    HEART_METRICS_PATH,
    DIABETES_METRICS_PATH,
    KIDNEY_METRICS_PATH,
    REPORTS_DIR
)
from src.data_loader import load_heart_data, load_diabetes_data, load_kidney_data, get_dataset_summary
from src.predict import predict_disease
from database import (
    get_recent_predictions,
    get_model_metadata_history,
    clear_prediction_logs,
    log_feedback
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Healthcare Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS: PROFESSIONAL MEDICAL THEME (White & Navy Blue)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0284c7 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
    }
    
    .main-header h1 {
        color: white !important;
        font-weight: 700;
        font-size: 2.3rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem;
    }
    
    .main-header p {
        color: #e2e8f0 !important;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 0;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .metric-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    
    .metric-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    
    .metric-subtitle {
        color: #0284c7;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 0.45rem 1.1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        text-align: center;
        letter-spacing: 0.02em;
    }
    
    .risk-low {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    
    .risk-mod {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    .risk-high {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    .clinical-banner {
        background-color: #f8fafc;
        border-left: 5px solid #0284c7;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        font-size: 0.92rem;
        color: #334155;
    }
    
    .disclaimer-box {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-top: 1.5rem;
        font-size: 0.85rem;
        color: #64748b;
    }
    
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 100%);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.3);
        transition: all 0.2s ease;
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #172554 0%, #0369a1 100%);
        box-shadow: 0 6px 12px -2px rgba(2, 132, 199, 0.45);
        transform: translateY(-1px);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=64)
    st.title("Smart Healthcare")
    st.caption("AI Clinical Decision Support System")
    st.markdown("---")
    
    app_page = st.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "❤️ Heart Disease Prediction",
            "🩸 Diabetes Prediction",
            "🩺 Chronic Kidney Disease",
            "📊 Model Benchmarks",
            "📈 Exploratory Analytics",
            "📑 Demonstration Audit Logs",
            "ℹ️ About & Methodology"
        ],
        index=0
    )
    
    st.markdown("---")
    st.subheader("⚙️ System Settings")
    allow_logging = st.checkbox("Enable Privacy Demonstration Logging", value=True, help="Logs non-identifiable prediction queries to SQLite database.")
    
    st.markdown("---")
    st.caption("🛡️ **Educational Demonstration Only**")
    st.caption("Not intended for direct clinical diagnosis or medical management.")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def load_benchmark_json(file_path: str):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None

def render_risk_gauge(probability: float, risk_tier: str, risk_color: str):
    """Renders a visual horizontal probability risk meter."""
    pct = probability * 100
    st.markdown(f"""
        <div style="background-color: #e2e8f0; border-radius: 9999px; height: 18px; width: 100%; margin: 12px 0; overflow: hidden;">
            <div style="background-color: {risk_color}; height: 100%; width: {pct}%; border-radius: 9999px; transition: width 0.6s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748b; font-weight: 500;">
            <span>0% (Minimal Risk)</span>
            <span>35% (Moderate Threshold)</span>
            <span>65% (Elevated Threshold)</span>
            <span>100% (High Risk)</span>
        </div>
    """, unsafe_allow_html=True)

def render_shap_waterfall(top_factors: list):
    """Renders SHAP local contribution bar chart."""
    if not top_factors:
        st.info("No explainability factors available for this prediction.")
        return
        
    df_factors = pd.DataFrame(top_factors)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    
    # Reverse so top is at top of horizontal chart
    df_rev = df_factors.iloc[::-1]
    
    colors = ["#ef4444" if v > 0 else "#10b981" for v in df_rev["shap_value"]]
    bars = ax.barh(df_rev["feature"], df_rev["shap_value"], color=colors, height=0.55)
    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1.0)
    
    ax.set_xlabel("SHAP Impact on Disease Log-Odds (Risk Attribution)", fontsize=10, fontweight="bold")
    ax.set_title("Key Diagnostic Influencing Factors (Explainable AI)", fontsize=11, fontweight="bold", pad=10)
    
    for bar, val in zip(bars, df_rev["shap_value"]):
        offset = 0.02 if val >= 0 else -0.02
        ha = "left" if val >= 0 else "right"
        ax.annotate(f"{val:+.3f}", (val + offset, bar.get_y() + bar.get_height()/2),
                    ha=ha, va="center", fontsize=8.5, fontweight="bold")
                    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------------------------------------------
# PAGE 1: HOME DASHBOARD
# -----------------------------------------------------------------------------
if app_page == "🏠 Home Dashboard":
    st.markdown("""
        <div class="main-header">
            <h1>SMART HEALTHCARE DISEASE PREDICTION SYSTEM</h1>
            <p>Empowering Clinical Decision Support with Machine Learning & Explainable AI (SHAP)</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="clinical-banner">
            <strong>Platform Purpose:</strong> This interactive web application demonstrates modern end-to-end Machine Learning pipelines 
            for multi-disease risk stratification across cardiovascular, metabolic, and renal disease cohorts. 
            All models have been rigorously trained on legitimate public benchmark datasets from the <strong>UC Irvine Machine Learning Repository</strong> 
            and the <strong>CDC BRFSS Health Indicators</strong>.
        </div>
    """, unsafe_allow_html=True)
    
    # Overview Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Cardiovascular Health</div>
                <div class="metric-value">Heart Disease</div>
                <div class="metric-subtitle">UCI Cleveland Cohort (303 records)</div>
                <hr style="margin: 0.8rem 0; border: 0; border-top: 1px solid #f1f5f9;">
                <p style="font-size: 0.88rem; color: #475569; margin: 0;">
                    Predicts coronary artery disease presence using 13 clinical indicators (ECG, ST slope, fluoroscopy).
                </p>
                <div style="margin-top: 0.8rem; font-weight: 600; color: #1e3a8a; font-size: 0.85rem;">
                    Top Benchmark: 88.5% Accuracy | 0.967 ROC-AUC
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Metabolic Risk</div>
                <div class="metric-value">Diabetes</div>
                <div class="metric-subtitle">CDC BRFSS Indicators (70,692 balanced records)</div>
                <hr style="margin: 0.8rem 0; border: 0; border-top: 1px solid #f1f5f9;">
                <p style="font-size: 0.88rem; color: #475569; margin: 0;">
                    Screening tool evaluating BMI, physical activity, hypertension, and clinical history for diabetes risk.
                </p>
                <div style="margin-top: 0.8rem; font-weight: 600; color: #0284c7; font-size: 0.85rem;">
                    Top Benchmark: 74.5% Accuracy | 0.822 ROC-AUC
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Renal Function</div>
                <div class="metric-value">Kidney Disease (CKD)</div>
                <div class="metric-subtitle">UCI CKD Cohort (400 records)</div>
                <hr style="margin: 0.8rem 0; border: 0; border-top: 1px solid #f1f5f9;">
                <p style="font-size: 0.88rem; color: #475569; margin: 0;">
                    Multi-marker renal risk model utilizing serum creatinine, hemoglobin, albumin, and packed cell volume.
                </p>
                <div style="margin-top: 0.8rem; font-weight: 600; color: #0d9488; font-size: 0.85rem;">
                    Top Benchmark: 100.0% Accuracy | 1.000 ROC-AUC
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Machine Learning Architecture Flow
    st.subheader("🔬 Complete ML Lifecycle Architecture")
    flow_col1, flow_col2, flow_col3, flow_col4, flow_col5 = st.columns(5)
    
    with flow_col1:
        st.info("**1. Dataset Ingestion**\n\nAuthentic UCI/CDC public cohorts downloaded & validated.")
    with flow_col2:
        st.info("**2. Feature Engineering**\n\nLeakage-free Scikit-learn Pipelines, robust imputers & scalers.")
    with flow_col3:
        st.info("**3. Model Optimization**\n\n5-Fold Stratified CV tuning LogReg, Trees, RF & XGBoost.")
    with flow_col4:
        st.info("**4. Clinical Explainability**\n\nSHAP Tree & Linear feature attributions for transparent AI.")
    with flow_col5:
        st.info("**5. Secure Deployment**\n\nParameterized SQLite audit logs, zero-PII privacy guarantee.")

    # Medical Disclaimer Banner
    st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Medical Safety & Regulatory Disclaimer:</strong><br>
            This software is created exclusively for data science educational demonstration, academic research, and technical portfolio presentation. 
            It is <strong>NOT</strong> a certified medical diagnostic device, does not provide medical advice or establish clinical care relationships, 
            and must never substitute for professional clinical judgment, laboratory testing, or physician consultation.
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 2: HEART DISEASE PREDICTION
# -----------------------------------------------------------------------------
elif app_page == "❤️ Heart Disease Prediction":
    st.markdown("""
        <div class="main-header">
            <h1>❤️ Cardiovascular Disease Risk Screening</h1>
            <p>Real-time risk assessment using clinical biomarkers from the UCI Cleveland Dataset</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Preset Selector to Test Accuracy Based on Given Values
    preset_choice = st.selectbox(
        "🎯 Quick Test Presets (Load Patient Profiles with Known Medical Outcomes):",
        [
            "Custom Input (Enter your own values)",
            "Test Case A: Healthy Patient (Actual: Negative / Healthy)",
            "Test Case B: High-Risk Patient (Actual: Confirmed Coronary Artery Disease)"
        ]
    )
    
    # Set default values based on chosen preset
    if preset_choice == "Test Case A: Healthy Patient (Actual: Negative / Healthy)":
        p_age, p_sex, p_cp, p_bp, p_chol, p_fbs, p_ecg, p_maxhr, p_exang, p_oldpeak, p_slope, p_ca, p_thal = (
            34, 1, 1, 118, 182, 0, 0, 174, 0, 0.0, 1, 0, 3
        )
        st.info("Loaded Test Case A: Young active male, normal ECG, no fluoroscopy vessels colored. Expected Ground Truth: **Healthy (No Disease)**.")
    elif preset_choice == "Test Case B: High-Risk Patient (Actual: Confirmed Coronary Artery Disease)":
        p_age, p_sex, p_cp, p_bp, p_chol, p_fbs, p_ecg, p_maxhr, p_exang, p_oldpeak, p_slope, p_ca, p_thal = (
            67, 1, 4, 160, 286, 0, 2, 108, 1, 2.6, 2, 3, 7
        )
        st.warning("Loaded Test Case B: 67yo male, asymptomatic angina, resting BP 160, 3 colored vessels. Expected Ground Truth: **Diseased (High Risk)**.")
    else:
        p_age, p_sex, p_cp, p_bp, p_chol, p_fbs, p_ecg, p_maxhr, p_exang, p_oldpeak, p_slope, p_ca, p_thal = (
            54, 1, 1, 130, 240, 0, 0, 150, 0, 1.0, 1, 0, 3
        )

    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("📋 Patient Clinical Parameters")
        
        with st.form("heart_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age (years)", 20, 100, p_age)
                sex = st.selectbox("Biological Sex", options=[(1, "Male"), (0, "Female")], index=0 if p_sex==1 else 1, format_func=lambda x: x[1])[0]
                cp = st.selectbox("Chest Pain Type", options=[
                    (1, "1: Typical Angina"),
                    (2, "2: Atypical Angina"),
                    (3, "3: Non-anginal Pain"),
                    (4, "4: Asymptomatic")
                ], index=p_cp - 1, format_func=lambda x: x[1])[0]
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, p_bp)
                chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, p_chol)
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[(0, "False (<= 120 mg/dl)"), (1, "True (> 120 mg/dl)")], index=p_fbs, format_func=lambda x: x[1])[0]
                
            with c2:
                restecg = st.selectbox("Resting Electrocardiogram (ECG)", options=[
                    (0, "0: Normal"),
                    (1, "1: ST-T Wave Abnormality"),
                    (2, "2: Left Ventricular Hypertrophy")
                ], index=p_ecg, format_func=lambda x: x[1])[0]
                thalach = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, p_maxhr)
                exang = st.selectbox("Exercise Induced Angina", options=[(0, "No"), (1, "Yes")], index=p_exang, format_func=lambda x: x[1])[0]
                oldpeak = st.slider("ST Depression (Exercise vs Rest in mm)", 0.0, 6.5, float(p_oldpeak), step=0.1)
                slope = st.selectbox("Slope of Peak Exercise ST", options=[
                    (1, "1: Upsloping"),
                    (2, "2: Flat"),
                    (3, "3: Downsloping")
                ], index=p_slope - 1, format_func=lambda x: x[1])[0]
                ca = st.selectbox("Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3], index=p_ca)
                thal_idx = 0 if p_thal==3 else (1 if p_thal==6 else 2)
                thal = st.selectbox("Thalassemia Stress Result", options=[
                    (3, "3: Normal"),
                    (6, "6: Fixed Defect"),
                    (7, "7: Reversable Defect")
                ], index=thal_idx, format_func=lambda x: x[1])[0]
                
            submit_heart = st.form_submit_button("🔍 Compute Heart Disease Risk")

    with col_result:
        st.subheader("📊 Diagnostic Risk Assessment")
        
        if submit_heart:
            input_dict = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
                "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }
            
            with st.spinner("Executing pipeline inference & SHAP explainability..."):
                try:
                    result = predict_disease("heart", input_dict, allow_logging=allow_logging)
                    
                    # Risk Badge
                    badge_class = "risk-low" if result["risk_tier"] == "Low Risk" else ("risk-mod" if "Moderate" in result["risk_tier"] else "risk-high")
                    
                    st.markdown(f"""
                        <div style="background-color: white; border-radius: 12px; padding: 1.2rem; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="metric-title">Model Estimation</span>
                                <span class="risk-badge {badge_class}">{result['risk_tier']}</span>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-top: 0.5rem;">
                                {result['prediction_label']}
                            </div>
                            <div style="color: #64748b; font-size: 0.95rem; margin-top: 0.2rem;">
                                Estimated Probability: <strong>{result['probability_percent']}%</strong>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    render_risk_gauge(result["probability"], result["risk_tier"], result["risk_color"])
                    
                    # Explainable AI section
                    st.markdown("#### 🔍 Explainable AI (SHAP Analysis)")
                    render_shap_waterfall(result["top_factors"])
                    
                    # Interactive Clinician Feedback
                    st.markdown("##### 🩺 Demonstration Clinician Review")
                    fc1, fc2, fc3 = st.columns(3)
                    with fc1:
                        if st.button("✅ Accept", key="hb_acc"):
                            log_feedback(result["log_id"], "Heart Disease", "Accepted")
                            st.success("Decision logged: Accepted")
                    with fc2:
                        if st.button("✏️ Modify", key="hb_mod"):
                            log_feedback(result["log_id"], "Heart Disease", "Modified")
                            st.info("Decision logged: Modified")
                    with fc3:
                        if st.button("❌ Reject", key="hb_rej"):
                            log_feedback(result["log_id"], "Heart Disease", "Rejected")
                            st.warning("Decision logged: Rejected")
                            
                    # Download Report Button
                    report_text = f"""SMART HEALTHCARE DISEASE PREDICTION SYSTEM
CLINICAL DEMONSTRATION REPORT
==================================================
Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Disease Evaluated: Heart Disease (Cardiovascular Risk)
Model: Logistic Regression (UCI Cleveland Optimized)

DIAGNOSTIC PREDICTION:
----------------------
Assessment Result: {result['prediction_label']}
Risk Tier: {result['risk_tier']}
Estimated Probability: {result['probability_percent']}%

INPUT PARAMETERS:
-----------------
{json.dumps(input_dict, indent=2)}

KEY INFLUENCING FACTORS (SHAP):
-------------------------------
{json.dumps(result['top_factors'], indent=2)}

LEGAL DISCLAIMER:
-----------------
{result['disclaimer']}
==================================================
"""
                    st.download_button(
                        label="📥 Download Clinical Summary Report (.txt)",
                        data=report_text,
                        file_name=f"heart_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
        else:
            st.info("👈 Enter patient clinical vitals on the left and click **'Compute Heart Disease Risk'** to begin.")

# -----------------------------------------------------------------------------
# PAGE 3: DIABETES PREDICTION
# -----------------------------------------------------------------------------
elif app_page == "🩸 Diabetes Prediction":
    st.markdown("""
        <div class="main-header">
            <h1>🩸 Diabetes Metabolic Risk Screening</h1>
            <p>Population health risk stratification trained on 70,692 CDC BRFSS Health Indicators</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("📋 Demographic & Health Indicators")
        
        with st.form("diabetes_form"):
            c1, c2 = st.columns(2)
            with c1:
                high_bp = st.selectbox("High Blood Pressure Diagnosed?", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
                high_chol = st.selectbox("High Blood Cholesterol Diagnosed?", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
                chol_check = st.selectbox("Cholesterol Checked in Last 5 Years?", [(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
                bmi = st.slider("Body Mass Index (BMI)", 12, 60, 28)
                smoker = st.selectbox("Smoked >= 100 Cigarettes in Life?", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
                stroke = st.selectbox("History of Stroke?", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
                heart_attack = st.selectbox("History of Heart Attack / CHD?", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
                phys_activity = st.selectbox("Physical Activity in Past 30 Days?", [(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
                fruits = st.selectbox("Consumes Fruit >= 1 Time / Day?", [(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
                veggies = st.selectbox("Consumes Vegetables >= 1 Time / Day?", [(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
                heavy_alcohol = st.selectbox("Heavy Alcohol Consumption?", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
                
            with c2:
                healthcare = st.selectbox("Has Healthcare Coverage?", [(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
                no_doc_cost = st.selectbox("Could Not Afford Doctor in Past Year?", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
                gen_hlth = st.selectbox("General Health Assessment", [
                    (1, "1: Excellent"), (2, "2: Very Good"), (3, "3: Good"), (4, "4: Fair"), (5, "5: Poor")
                ], index=2, format_func=lambda x: x[1])[0]
                ment_hlth = st.slider("Days with Poor Mental Health (Past 30d)", 0, 30, 0)
                phys_hlth = st.slider("Days with Physical Injury/Illness (Past 30d)", 0, 30, 0)
                diff_walk = st.selectbox("Serious Difficulty Walking or Climbing Stairs?", [(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
                sex = st.selectbox("Biological Sex", [(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
                age_cat = st.selectbox("Age Bracket", [
                    (1, "18-24"), (2, "25-29"), (3, "30-34"), (4, "35-39"),
                    (5, "40-44"), (6, "45-49"), (7, "50-54"), (8, "55-59"),
                    (9, "60-64"), (10, "65-69"), (11, "70-74"), (12, "75-79"), (13, "80+")
                ], index=6, format_func=lambda x: x[1])[0]
                education = st.selectbox("Highest Education Level", [
                    (1, "Never attended / Kindergarten"),
                    (2, "Elementary (1-8)"),
                    (3, "Some High School (9-11)"),
                    (4, "High School Graduate / GED"),
                    (5, "Some College / Technical"),
                    (6, "College Graduate (4+ yrs)")
                ], index=4, format_func=lambda x: x[1])[0]
                income = st.selectbox("Annual Household Income Bracket", [
                    (1, "< $10k"), (2, "$10k-$15k"), (3, "$15k-$20k"), (4, "$20k-$25k"),
                    (5, "$25k-$35k"), (6, "$35k-$50k"), (7, "$50k-$75k"), (8, ">= $75k")
                ], index=5, format_func=lambda x: x[1])[0]
                
            submit_diabetes = st.form_submit_button("🔍 Compute Diabetes Risk")

    with col_result:
        st.subheader("📊 Diagnostic Risk Assessment")
        
        if submit_diabetes:
            input_dict = {
                "HighBP": high_bp, "HighChol": high_chol, "CholCheck": chol_check,
                "BMI": bmi, "Smoker": smoker, "Stroke": stroke,
                "HeartDiseaseorAttack": heart_attack, "PhysActivity": phys_activity,
                "Fruits": fruits, "Veggies": veggies, "HvyAlcoholConsump": heavy_alcohol,
                "AnyHealthcare": healthcare, "NoDocbcCost": no_doc_cost, "GenHlth": gen_hlth,
                "MentHlth": ment_hlth, "PhysHlth": phys_hlth, "DiffWalk": diff_walk,
                "Sex": sex, "Age": age_cat, "Education": education, "Income": income
            }
            
            with st.spinner("Evaluating XGBoost model & calculating SHAP attributions..."):
                try:
                    result = predict_disease("diabetes", input_dict, allow_logging=allow_logging)
                    badge_class = "risk-low" if result["risk_tier"] == "Low Risk" else ("risk-mod" if "Moderate" in result["risk_tier"] else "risk-high")
                    
                    st.markdown(f"""
                        <div style="background-color: white; border-radius: 12px; padding: 1.2rem; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="metric-title">Metabolic Estimation</span>
                                <span class="risk-badge {badge_class}">{result['risk_tier']}</span>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-top: 0.5rem;">
                                {result['prediction_label']}
                            </div>
                            <div style="color: #64748b; font-size: 0.95rem; margin-top: 0.2rem;">
                                Estimated Probability: <strong>{result['probability_percent']}%</strong>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    render_risk_gauge(result["probability"], result["risk_tier"], result["risk_color"])
                    st.markdown("#### 🔍 Explainable AI (SHAP Analysis)")
                    render_shap_waterfall(result["top_factors"])
                    
                except Exception as e:
                    st.error(f"Diabetes Prediction Error: {e}")
        else:
            st.info("👈 Enter health parameters on the left and click **'Compute Diabetes Risk'**.")

# -----------------------------------------------------------------------------
# PAGE 4: CHRONIC KIDNEY DISEASE PREDICTION
# -----------------------------------------------------------------------------
elif app_page == "🩺 Chronic Kidney Disease":
    st.markdown("""
        <div class="main-header">
            <h1>🩺 Chronic Kidney Disease (CKD) Screening</h1>
            <p>Renal function and metabolic laboratory panel evaluated by optimized XGBoost Pipeline</p>
        </div>
    """, unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("📋 Renal Biomarkers & Clinical History")
        
        with st.form("ckd_form"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Lab Biomarkers**")
                age = st.slider("Age (years)", 2, 100, 52)
                bp = st.slider("Blood Pressure (mm Hg)", 50, 180, 80)
                sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
                al = st.selectbox("Albumin in Urine", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=0)
                su = st.selectbox("Sugar in Urine", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=0)
                bgr = st.number_input("Random Blood Glucose (mg/dl)", 50, 500, 115)
                bu = st.number_input("Blood Urea (mg/dl)", 5, 400, 35)
                sc = st.number_input("Serum Creatinine (mg/dl)", 0.2, 16.0, 1.0, step=0.1)
                sod = st.number_input("Sodium (mEq/L)", 100, 165, 138)
                pot = st.number_input("Potassium (mEq/L)", 2.0, 10.0, 4.5, step=0.1)
                hemo = st.number_input("Hemoglobin (g/dl)", 3.0, 19.0, 14.2, step=0.1)
                pcv = st.number_input("Packed Cell Volume (Hematocrit %)", 12, 56, 42)
                
            with c2:
                st.markdown("**Microscopic & Clinical Findings**")
                wbcc = st.number_input("White Blood Cell Count (cells/cumm)", 2000, 27000, 7500)
                rbcc = st.number_input("Red Blood Cell Count (mil/cumm)", 2.0, 8.0, 5.0, step=0.1)
                rbc = st.selectbox("Red Blood Cells in Urine", ["normal", "abnormal"])
                pc = st.selectbox("Pus Cells in Urine", ["normal", "abnormal"])
                pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
                ba = st.selectbox("Bacteria in Urine", ["notpresent", "present"])
                htn = st.selectbox("Hypertension Diagnosed?", ["no", "yes"])
                dm = st.selectbox("Diabetes Mellitus History?", ["no", "yes"])
                cad = st.selectbox("Coronary Artery Disease?", ["no", "yes"])
                appet = st.selectbox("Patient Appetite", ["good", "poor"])
                pe = st.selectbox("Pedal Edema (Swelling)?", ["no", "yes"])
                ane = st.selectbox("Anemia Diagnosed?", ["no", "yes"])
                
            submit_ckd = st.form_submit_button("🔍 Compute Renal Risk")

    with col_result:
        st.subheader("📊 Diagnostic Risk Assessment")
        
        if submit_ckd:
            input_dict = {
                "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
                "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
                "hemo": hemo, "pcv": pcv, "wbcc": wbcc, "rbcc": rbcc,
                "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba, "htn": htn,
                "dm": dm, "cad": cad, "appet": appet, "pe": pe, "ane": ane
            }
            
            with st.spinner("Processing laboratory panel with XGBoost Pipeline..."):
                try:
                    result = predict_disease("kidney", input_dict, allow_logging=allow_logging)
                    badge_class = "risk-low" if result["risk_tier"] == "Low Risk" else ("risk-mod" if "Moderate" in result["risk_tier"] else "risk-high")
                    
                    st.markdown(f"""
                        <div style="background-color: white; border-radius: 12px; padding: 1.2rem; border: 1px solid #e2e8f0; margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="metric-title">Renal Risk Estimation</span>
                                <span class="risk-badge {badge_class}">{result['risk_tier']}</span>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-top: 0.5rem;">
                                {result['prediction_label']}
                            </div>
                            <div style="color: #64748b; font-size: 0.95rem; margin-top: 0.2rem;">
                                Estimated Probability: <strong>{result['probability_percent']}%</strong>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    render_risk_gauge(result["probability"], result["risk_tier"], result["risk_color"])
                    st.markdown("#### 🔍 Explainable AI (SHAP Analysis)")
                    render_shap_waterfall(result["top_factors"])
                    
                except Exception as e:
                    st.error(f"CKD Prediction Error: {e}")
        else:
            st.info("👈 Enter patient renal panel vitals and click **'Compute Renal Risk'**.")

# -----------------------------------------------------------------------------
# PAGE 5: MODEL PERFORMANCE BENCHMARKS
# -----------------------------------------------------------------------------
elif app_page == "📊 Model Benchmarks":
    st.markdown("""
        <div class="main-header">
            <h1>📊 Model Evaluation & Benchmarks</h1>
            <p>Rigorous comparative evaluation across 4 Machine Learning algorithms on untouched test sets</p>
        </div>
    """, unsafe_allow_html=True)
    
    disease_choice = st.selectbox(
        "Select Disease Cohort Evaluation:",
        ["Heart Disease (UCI Cleveland)", "Diabetes (CDC Health Indicators)", "Chronic Kidney Disease (UCI CKD)"]
    )
    
    if "Heart" in disease_choice:
        metrics_data = load_benchmark_json(HEART_METRICS_PATH)
        subfolder = "heart"
    elif "Diabetes" in disease_choice:
        metrics_data = load_benchmark_json(DIABETES_METRICS_PATH)
        subfolder = "diabetes"
    else:
        metrics_data = load_benchmark_json(KIDNEY_METRICS_PATH)
        subfolder = "kidney"
        
    if metrics_data:
        st.markdown(f"### Selected Production Model: **{metrics_data.get('selected_model')}**")
        
        benchmarks = metrics_data.get("benchmark_results", {})
        
        # Build comparison DataFrame
        table_rows = []
        for model_name, m in benchmarks.items():
            table_rows.append({
                "Algorithm": model_name,
                "Accuracy (%)": f"{m.get('accuracy', 0)*100:.2f}%",
                "Recall / Sensitivity (%)": f"{m.get('recall', 0)*100:.2f}%",
                "Specificity (%)": f"{m.get('specificity', 0)*100:.2f}%",
                "F1-Score": f"{m.get('f1', 0):.4f}",
                "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
                "CV ROC-AUC (Tuned)": f"{m.get('best_cv_roc_auc', 0):.4f}"
            })
            
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
        
        # Display Generated Visualizations
        st.markdown("### 📈 Diagnostic Performance Plots")
        pcol1, pcol2 = st.columns(2)
        
        comp_img = os.path.join(REPORTS_DIR, subfolder, "model_comparison.png")
        if os.path.exists(comp_img):
            with pcol1:
                st.image(comp_img, caption="Algorithm Performance Comparison Across Metrics", use_container_width=True)
                
        best_cm_img = os.path.join(REPORTS_DIR, subfolder, f"{metrics_data.get('selected_model').lower().replace(' ', '_')}_cm.png")
        if os.path.exists(best_cm_img):
            with pcol2:
                st.image(best_cm_img, caption=f"Confusion Matrix: {metrics_data.get('selected_model')}", use_container_width=True)
                
        pcol3, pcol4 = st.columns(2)
        best_roc_img = os.path.join(REPORTS_DIR, subfolder, f"{metrics_data.get('selected_model').lower().replace(' ', '_')}_roc.png")
        if os.path.exists(best_roc_img):
            with pcol3:
                st.image(best_roc_img, caption="Receiver Operating Characteristic (ROC) Curve", use_container_width=True)
                
        best_pr_img = os.path.join(REPORTS_DIR, subfolder, f"{metrics_data.get('selected_model').lower().replace(' ', '_')}_pr.png")
        if os.path.exists(best_pr_img):
            with pcol4:
                st.image(best_pr_img, caption="Precision-Recall Curve", use_container_width=True)
                
    else:
        st.warning("Benchmark metrics not found. Please run `python -m src.train` to generate benchmark files.")

# -----------------------------------------------------------------------------
# PAGE 6: EXPLORATORY DATA ANALYTICS
# -----------------------------------------------------------------------------
elif app_page == "📈 Exploratory Analytics":
    st.markdown("""
        <div class="main-header">
            <h1>📈 Exploratory Data Analytics (EDA)</h1>
            <p>Comprehensive statistical distributions, feature correlations, and cohort characteristics</p>
        </div>
    """, unsafe_allow_html=True)
    
    eda_choice = st.selectbox("Select Dataset to Explore:", ["Heart Disease (UCI)", "Diabetes (CDC)", "Kidney Disease (UCI CKD)"])
    
    if "Heart" in eda_choice:
        X, y = load_heart_data()
        df = pd.concat([X, y.rename("target")], axis=1)
        disease_title = "UCI Cleveland Heart Disease"
    elif "Diabetes" in eda_choice:
        X, y = load_diabetes_data(sample_size=5000)
        df = pd.concat([X, y.rename("target")], axis=1)
        disease_title = "CDC Diabetes Health Indicators (5,000 Sample)"
    else:
        X, y = load_kidney_data()
        df = pd.concat([X, y.rename("target")], axis=1)
        disease_title = "UCI Chronic Kidney Disease"
        
    st.write(f"**Dataset Summary:** {df.shape[0]} samples, {df.shape[1]} features.")
    
    tab1, tab2, tab3 = st.tabs(["📋 Data Overview", "📊 Target Distribution", "🔥 Correlation Heatmap"])
    
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("**Descriptive Statistics:**")
        st.dataframe(df.describe(), use_container_width=True)
        
    with tab2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        counts = df["target"].value_counts()
        labels = ["Negative (No Disease)", "Positive (Disease)"]
        ax.bar(labels, counts, color=["#0284c7", "#ef4444"], width=0.5)
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title(f"Class Balance: {disease_title}", fontweight="bold", pad=12)
        for i, v in enumerate(counts):
            ax.text(i, v / 2, f"{v} ({v/len(df):.1%})", ha="center", color="white", fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)
        
    with tab3:
        # Numeric correlation heatmap
        num_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(9, 6.5))
        sns.heatmap(corr, cmap="vlag", center=0, annot=False, ax=ax, linewidths=0.5)
        ax.set_title(f"Feature Correlation Matrix - {disease_title}", fontweight="bold", pad=12)
        st.pyplot(fig)
        plt.close(fig)

# -----------------------------------------------------------------------------
# PAGE 7: DEMONSTRATION AUDIT LOGS
# -----------------------------------------------------------------------------
elif app_page == "📑 Demonstration Audit Logs":
    st.markdown("""
        <div class="main-header">
            <h1>📑 Demonstration Prediction Audit Logs</h1>
            <p>SQLite parameter-governed audit records for educational tracking (Zero PII stored)</p>
        </div>
    """, unsafe_allow_html=True)
    
    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        if st.button("🗑️ Clear Logs"):
            clear_prediction_logs()
            st.success("Demonstration logs cleared.")
            st.rerun()
            
    recent_logs = get_recent_predictions(limit=50)
    
    if recent_logs:
        df_logs = pd.DataFrame(recent_logs)
        
        # Display formatted log table
        st.dataframe(
            df_logs[["id", "timestamp", "disease_type", "prediction_label", "probability", "risk_category"]],
            use_container_width=True
        )
        
        # CSV Export
        csv_data = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Logs as CSV",
            data=csv_data,
            file_name=f"demonstration_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No prediction logs recorded yet. Perform predictions on disease pages to generate audit entries.")

# -----------------------------------------------------------------------------
# PAGE 8: ABOUT & METHODOLOGY
# -----------------------------------------------------------------------------
elif app_page == "ℹ️ About & Methodology":
    st.markdown("""
        <div class="main-header">
            <h1>ℹ️ About the Project & Clinical Methodology</h1>
            <p>System Architecture, Data Provenance, Machine Learning Rigor, and Technical Interview Guide</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 1. Project Overview & Clinical Objective
    The **Smart Healthcare Disease Prediction System** is a professional-grade Data Science and Machine Learning project engineered 
    to demonstrate early risk screening and prognostic risk stratification across three major chronic disease categories:
    - **Cardiovascular Disease** (UCI Cleveland Heart Disease dataset)
    - **Type 2 Diabetes** (CDC Behavioral Risk Factor Surveillance System)
    - **Chronic Kidney Disease** (UCI CKD laboratory panel)

    ### 2. Machine Learning Methodology
    - **Leakage-Free Pipelines**: Strict separation between training and test sets (`train_test_split(stratify=y, test_size=0.20)`). Preprocessing transformers are fitted exclusively on training data.
    - **Imputation Strategy**: Median imputation for skewed numeric clinical features; mode imputation for categorical indicators.
    - **Algorithm Comparison**: For every disease cohort, four diverse algorithms were trained and tuned using **5-Fold Stratified Cross-Validation**:
      1. Logistic Regression (Linear baseline with L2 regularization)
      2. Decision Tree Classifier (Non-linear rule-based baseline)
      3. Random Forest Classifier (Bagging ensemble for variance reduction)
      4. XGBoost Classifier (Gradient boosted decision trees for state-of-the-art tabular accuracy)
    - **Metric Prioritization**: In medical diagnosis, optimizing solely for accuracy can be misleading due to class imbalance. We prioritize **Recall (Sensitivity)** to minimize fatal False Negatives, along with **ROC-AUC** and **F1-Score**.

    ### 3. Explainable AI (SHAP)
    - Integrated **SHAP (SHapley Additive exPlanations)** based on cooperative game theory to provide transparent, instance-level attributions for individual patient predictions.
    - **Crucial Ethical Note**: Model feature attributions reflect statistical correlation within training datasets and **do not** prove biological causation.

    ### 4. Technical Stack
    - **Language & Modeling**: Python 3.11, Scikit-learn, XGBoost, SHAP, Pandas, NumPy, Matplotlib, Seaborn.
    - **Database & Persistence**: SQLite, Joblib.
    - **User Interface**: Streamlit with custom responsive medical CSS.
    - **Testing**: Python `unittest` suite verifying preprocessing, model inference, and SQLite transactions.
    """)
