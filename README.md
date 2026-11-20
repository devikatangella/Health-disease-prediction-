# 🩺 SMART HEALTHCARE DISEASE PREDICTION SYSTEM
### *End-to-End Clinical Decision Support System with Machine Learning, Explainable AI (SHAP), and SQLite Audit Logging*

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![ML Library](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Gradient Boosting](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/SHAP-Explainable%20AI-red.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

> ### ⚠️ Regulatory & Medical Safety Disclaimer
> **FOR EDUCATIONAL, RESEARCH, AND PORTFOLIO DEMONSTRATION ONLY.**  
> This software is an engineering proof-of-concept and research demonstration. It does **NOT** provide medical advice, establish a clinical doctor-patient relationship, or produce certified clinical diagnoses. Output predictions must never substitute for laboratory testing, physical examination, or professional clinical judgment.

---

## 📌 1. Project Overview & Problem Statement

Early detection of chronic non-communicable diseases—specifically **Cardiovascular Disease**, **Type 2 Diabetes**, and **Chronic Kidney Disease (CKD)**—is vital to reducing worldwide mortality and avoiding costly critical-care interventions. However, clinical risk assessment frequently struggles with:
1. **Multi-marker Complexity**: Correlating dozens of non-linear laboratory vitals and lifestyle indicators simultaneously.
2. **The "Black Box" Problem**: Clinical practitioners are rightfully hesitant to adopt Machine Learning systems when individual prediction reasoning cannot be audited.
3. **Data Leakage & Sub-optimal Metrics**: Naive models in published literature often optimize for accuracy over **Recall (Sensitivity)**, leading to catastrophic **False Negatives (Type II Errors)**.
4. **Patient Privacy & PII Vulnerabilities**: Storing personally identifiable health data without governance violates HIPAA/GDPR standards.

### Solution Overview
The **Smart Healthcare Disease Prediction System** delivers a robust, production-grade diagnostic framework:
- **Three Dedicated Diagnostic Pipelines**: Separate models for Heart Disease (UCI Cleveland), Diabetes (CDC BRFSS Health Indicators), and Chronic Kidney Disease (UCI CKD).
- **Leakage-Free Preprocessing**: Scikit-learn `ColumnTransformer` pipelines strictly fitted on training splits with median/mode imputers and robust scaling.
- **Explainable AI via SHAP**: Instance-level local feature attribution (waterfall/bar plots) explaining which biomarkers elevated or mitigated individual patient risk.
- **Zero-PII SQLite Demonstration Logging**: Audit trail with parameterized SQL queries capturing model decisions and feedback while ensuring zero storage of names, phone numbers, or identifiable identifiers.
- **Modern Medical Dashboard**: Custom-styled Streamlit interface featuring a clean white-and-navy aesthetic, interactive probability risk meters, and downloadable diagnostic reports.

---

## 🏗️ 2. Project Architecture & Modular Folder Structure

```
healthcare_prediction_system/
├── datasets/                            # Authentic public benchmark datasets
│   ├── heart/heart_disease.csv          # UCI Cleveland Heart Disease (303 rows, 14 cols)
│   ├── diabetes/diabetes_cdc.csv        # CDC BRFSS Health Indicators (70,692 balanced rows)
│   └── kidney/kidney_disease.csv        # UCI Chronic Kidney Disease (400 rows, 25 cols)
├── models/                              # Persisted production Scikit-learn pipelines
│   ├── heart_disease_pipeline.joblib    # Tuned Logistic Regression Pipeline
│   ├── diabetes_pipeline.joblib         # Tuned XGBoost Pipeline
│   └── kidney_disease_pipeline.joblib   # Tuned XGBoost Pipeline
├── notebooks/                           # Research & exploratory notebooks
│   └── healthcare_eda_and_benchmarks.ipynb
├── reports/                             # High-resolution benchmark diagnostic plots & JSONs
│   ├── heart/                           # CM, ROC, PR, Model Comparison plots
│   ├── diabetes/                        # CM, ROC, PR, Model Comparison plots
│   ├── kidney/                          # CM, ROC, PR, Model Comparison plots
│   ├── heart_metrics.json
│   ├── diabetes_metrics.json
│   └── kidney_metrics.json
├── src/                                 # Reusable production source code
│   ├── __init__.py
│   ├── config.py                        # Centralized schemas, bounds, units & paths
│   ├── download_datasets.py             # Automated UCI repository downloader
│   ├── data_loader.py                   # Clean loaders with documented target encoding
│   ├── preprocessing.py                 # Leakage-free ColumnTransformer pipelines
│   ├── train.py                         # 5-Fold Stratified CV & GridSearchCV tuning
│   ├── evaluate.py                      # Healthcare metrics & visualization generators
│   ├── explainability.py                # Unified SHAP Tree/Linear explainers
│   └── predict.py                       # Real-time inference & risk stratification
├── tests/                               # Comprehensive unit & integration tests
│   ├── __init__.py
│   ├── test_heart_pipeline.py           # Heart disease pipeline test suite
│   └── test_all_pipelines.py            # Multi-disease & database integration tests
├── app.py                               # Production Streamlit clinical dashboard
├── database.py                          # SQLite database management layer (Zero PII)
├── healthcare.db                        # SQLite database file
├── requirements.txt                     # Pinned compatible dependencies
├── .gitignore                           # Git ignore rules
└── README.md                            # Comprehensive documentation & interview guide
```

---

## 📊 3. Machine Learning Benchmark Results (Held-Out Test Sets)

All models were evaluated on untouched **20% stratified test sets** after hyperparameter optimization via **5-Fold Stratified Cross-Validation**.

### A. Heart Disease (UCI Cleveland - 303 Samples, 13 Features)
| Algorithm | Test Accuracy | Test Recall (Sensitivity) | Test Specificity | Test F1-Score | Test ROC-AUC | Best Hyperparameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (Selected)** | **88.52%** | **92.86%** | **84.85%** | **0.8814** | **0.9665** | `C=1.0, solver='lbfgs'` |
| **XGBoost Classifier** | 86.89% | 89.29% | 84.85% | 0.8621 | 0.9361 | `lr=0.01, max_depth=3, n_est=100` |
| **Random Forest Classifier** | 83.61% | 82.14% | 84.85% | 0.8214 | 0.9437 | `max_depth=4, n_est=100` |
| **Decision Tree Classifier** | 80.33% | 82.14% | 78.79% | 0.7931 | 0.8339 | `criterion='entropy', max_depth=3` |

*Selection Rationale:* In cardiovascular risk screening, **Recall** is prioritized to prevent missing patients with underlying coronary artery disease. Logistic Regression demonstrated superior recall (92.86%) and ROC-AUC (0.9665) while maintaining well-calibrated odds ratios.

---

### B. Diabetes Risk (CDC BRFSS - 70,692 Balanced Samples, 21 Features)
| Algorithm | Test Accuracy | Test Recall (Sensitivity) | Test Specificity | Test F1-Score | Test ROC-AUC | Best Hyperparameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost Classifier (Selected)** | **74.52%** | **79.04%** | **70.00%** | **0.7562** | **0.8215** | `lr=0.1, max_depth=4, n_est=100` |
| **Random Forest Classifier** | 74.34% | 79.05% | 69.63% | 0.7549 | 0.8130 | `max_depth=12, min_samples_split=10` |
| **Logistic Regression** | 74.29% | 77.17% | 71.41% | 0.7501 | 0.8172 | `C=0.1, solver='lbfgs'` |
| **Decision Tree Classifier** | 72.78% | 77.82% | 67.74% | 0.7408 | 0.7949 | `max_depth=6, min_samples_split=20` |

*Selection Rationale:* XGBoost yielded the highest test ROC-AUC (0.8215) and balanced F1-score (0.7562) on this large-scale survey cohort.

---

### C. Chronic Kidney Disease (UCI CKD - 400 Samples, 24 Features)
| Algorithm | Test Accuracy | Test Recall (Sensitivity) | Test Specificity | Test F1-Score | Test ROC-AUC | Best Hyperparameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **XGBoost Classifier (Selected)** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **1.0000** | `lr=0.05, max_depth=3, n_est=100` |
| **Random Forest Classifier** | 98.75% | 100.00% | 96.67% | 0.9901 | 1.0000 | `max_depth=4, n_est=50` |
| **Decision Tree Classifier** | 98.75% | 100.00% | 96.67% | 0.9901 | 0.9993 | `criterion='entropy', max_depth=3` |
| **Logistic Regression** | 98.75% | 98.00% | 100.00% | 0.9899 | 1.0000 | `C=10.0, solver='lbfgs'` |

*Selection Rationale:* Multi-marker laboratory indicators (serum creatinine, hemoglobin, packed cell volume, and specific gravity) provide strong discriminatory boundaries for renal impairment.

---

## 🔍 4. Explainable AI (SHAP) Integration

Rather than delivering a black-box percentage, the system incorporates **SHAP (SHapley Additive exPlanations)** based on cooperative game theory:
- **Local Attribution**: Every prediction produces a breakdown of the top 5 biomarkers driving the decision.
- **Directional Impact**: Displays whether each measurement increased risk (red) or decreased risk (green).
- **Clinical Safety Guardrail**: Explicitly informs users that statistical model attribution does not prove clinical causation.

---

## 💻 5. Installation & Execution Guide

### Prerequisites
- Python 3.10 or 3.11 installed
- Git installed

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/smart-healthcare-disease-prediction.git
cd smart-healthcare-disease-prediction
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Training & Benchmark Pipeline (Optional - Pre-trained Models Included)
```bash
python -m src.train
```

### Step 5: Run Unit Tests
```bash
python -m unittest tests/test_all_pipelines.py
```

### Step 6: Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🚀 6. Streamlit Cloud Deployment Instructions

1. Push your repository to GitHub.
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"** and select your repository and `main` branch.
4. Set the Main File Path to: `app.py`.
5. Click **"Deploy"**. (The pinned `requirements.txt` handles all package installation automatically).

---

## 📝 7. Resume Project Bullet Points (Ready for Your CV)

### **Smart Healthcare Disease Prediction & Explainability System** | *Lead Data Scientist & ML Engineer*
- Architected an end-to-end multi-disease clinical decision support platform predicting risk for **Cardiovascular Disease**, **Type 2 Diabetes**, and **Chronic Kidney Disease** using Scikit-learn, XGBoost, and Streamlit.
- Designed leakage-free preprocessing pipelines featuring median/mode imputation, robust scaling, and one-hot encoding, achieving **88.5% Accuracy / 92.9% Recall** on Heart Disease (UCI) and **100% ROC-AUC** on Chronic Kidney Disease.
- Implemented **Explainable AI (SHAP)** with instance-level local feature attributions, demystifying black-box predictions for clinical practitioners.
- Engineered a zero-PII audit logging service with parameterized SQLite queries for privacy-compliant demonstration tracking.
- Developed an interactive, responsive medical dashboard with real-time probability meters, comparative benchmark charts, and automated clinical summary report generation.

---

## 🎤 8. Technical Interview Preparation & FAQs

### Q1: Why did you prioritize Recall over Accuracy in this project?
> **Answer:** In healthcare risk screening, the cost of a **False Negative (Type II Error)** is asymmetric and potentially fatal. If a patient with coronary artery disease is predicted to be healthy, they miss life-saving interventions. Conversely, a **False Positive (Type I Error)** merely leads to secondary diagnostic confirmation (e.g., an angiogram or stress test). Therefore, we tuned decision thresholds and selected models to maximize **Recall (Sensitivity)** while preserving high **ROC-AUC**.

### Q2: How did you prevent Data Leakage between training and testing?
> **Answer:** We utilized Scikit-learn `Pipeline` and `ColumnTransformer`. The train-test split (`test_size=0.20`, stratified by label) was performed before any transformation. Imputers and scalers were **fitted strictly on the training set** (`fit_transform`) and only used to transform the untouched test set (`transform`). This guarantees zero information from the test distribution leaks into training.

### Q3: How does SHAP work and why is it preferred over default feature importances?
> **Answer:** Default Gini importance in tree ensembles is biased toward high-cardinality features and reflects global impurity reduction without consistency. SHAP calculates Shapley values from cooperative game theory, satisfying mathematical axioms of local accuracy and consistency. It attributes exact positive or negative log-odds contributions to each individual patient's feature values.

### Q4: Why did Logistic Regression outperform XGBoost on the Heart Disease dataset?
> **Answer:** The UCI Cleveland dataset contains 303 samples. Gradient boosted decision trees (XGBoost) have higher variance and risk overfitting on small sample sizes with complex regularized splits. Regularized Logistic Regression (L2 penalty) has lower capacity, making it less susceptible to overfitting on small clinical tabular datasets. On the larger CDC Diabetes dataset (70,692 rows), XGBoost outperformed linear models.

---

## 📄 9. License & Citations
- **UCI Machine Learning Repository**: Cleveland Heart Disease (ID 45), Chronic Kidney Disease (ID 336).
- **CDC BRFSS**: Behavioral Risk Factor Surveillance System Diabetes Health Indicators (UCI ID 891).
- Distributed under the **MIT License**.
