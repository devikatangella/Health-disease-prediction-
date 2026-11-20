"""
Configuration and Metadata Definitions for Smart Healthcare Disease Prediction System
Defines schemas, physiological ranges, options, units, and paths for:
1. Heart Disease (UCI Cleveland)
2. Diabetes Health Indicators (CDC BRFSS)
3. Chronic Kidney Disease (UCI CKD)
"""

import os

# Project Root Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATABASE_PATH = os.path.join(BASE_DIR, "healthcare.db")

# Ensure required directories exist
for path in [DATASETS_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(path, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_STATE = 42

# -------------------------------------------------------------
# 1. HEART DISEASE CONFIGURATION (UCI Cleveland)
# -------------------------------------------------------------
HEART_DATA_PATH = os.path.join(DATASETS_DIR, "heart", "heart_disease.csv")
HEART_MODEL_PATH = os.path.join(MODELS_DIR, "heart_disease_pipeline.joblib")
HEART_METRICS_PATH = os.path.join(REPORTS_DIR, "heart_metrics.json")

HEART_NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
HEART_CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
HEART_ALL_FEATURES = HEART_NUMERIC_FEATURES + HEART_CATEGORICAL_FEATURES
HEART_TARGET = "target"

HEART_FEATURE_METADATA = {
    "age": {"label": "Age (years)", "min": 20, "max": 100, "default": 54, "unit": "years"},
    "sex": {"label": "Biological Sex", "options": {0: "Female", 1: "Male"}, "default": 1},
    "cp": {
        "label": "Chest Pain Type",
        "options": {
            1: "Typical Angina",
            2: "Atypical Angina",
            3: "Non-anginal Pain",
            4: "Asymptomatic"
        },
        "default": 1
    },
    "trestbps": {"label": "Resting Blood Pressure", "min": 80, "max": 220, "default": 130, "unit": "mm Hg"},
    "chol": {"label": "Serum Cholesterol", "min": 100, "max": 600, "default": 240, "unit": "mg/dl"},
    "fbs": {"label": "Fasting Blood Sugar > 120 mg/dl", "options": {0: "False (<= 120 mg/dl)", 1: "True (> 120 mg/dl)"}, "default": 0},
    "restecg": {
        "label": "Resting Electrocardiogram (ECG)",
        "options": {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        },
        "default": 0
    },
    "thalach": {"label": "Maximum Heart Rate Achieved", "min": 60, "max": 220, "default": 150, "unit": "bpm"},
    "exang": {"label": "Exercise Induced Angina", "options": {0: "No", 1: "Yes"}, "default": 0},
    "oldpeak": {"label": "ST Depression (Exercise vs Rest)", "min": 0.0, "max": 6.5, "default": 1.0, "step": 0.1, "unit": "mm"},
    "slope": {
        "label": "Slope of Peak Exercise ST Segment",
        "options": {
            1: "Upsloping",
            2: "Flat",
            3: "Downsloping"
        },
        "default": 1
    },
    "ca": {"label": "Major Vessels Colored by Fluoroscopy", "min": 0, "max": 3, "default": 0, "unit": "vessels"},
    "thal": {
        "label": "Thalassemia Stress Result",
        "options": {
            3: "Normal",
            6: "Fixed Defect",
            7: "Reversable Defect"
        },
        "default": 3
    }
}

# -------------------------------------------------------------
# 2. DIABETES CONFIGURATION (CDC BRFSS Indicators)
# -------------------------------------------------------------
DIABETES_DATA_PATH = os.path.join(DATASETS_DIR, "diabetes", "diabetes_cdc.csv")
DIABETES_MODEL_PATH = os.path.join(MODELS_DIR, "diabetes_pipeline.joblib")
DIABETES_METRICS_PATH = os.path.join(REPORTS_DIR, "diabetes_metrics.json")

DIABETES_NUMERIC_FEATURES = ["BMI", "MentHlth", "PhysHlth"]
DIABETES_CATEGORICAL_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", 
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", 
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", 
    "GenHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]
DIABETES_ALL_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]
DIABETES_TARGET = "target"

DIABETES_FEATURE_METADATA = {
    "HighBP": {"label": "High Blood Pressure Diagnosed", "options": {0: "No", 1: "Yes"}, "default": 0},
    "HighChol": {"label": "High Blood Cholesterol Diagnosed", "options": {0: "No", 1: "Yes"}, "default": 0},
    "CholCheck": {"label": "Cholesterol Checked Within 5 Years", "options": {0: "No", 1: "Yes"}, "default": 1},
    "BMI": {"label": "Body Mass Index (BMI)", "min": 12, "max": 70, "default": 27, "unit": "kg/m²"},
    "Smoker": {"label": "Smoked at Least 100 Cigarettes in Life", "options": {0: "No", 1: "Yes"}, "default": 0},
    "Stroke": {"label": "Ever Told Had a Stroke", "options": {0: "No", 1: "Yes"}, "default": 0},
    "HeartDiseaseorAttack": {"label": "History of Coronary Heart Disease / MI", "options": {0: "No", 1: "Yes"}, "default": 0},
    "PhysActivity": {"label": "Physical Activity in Past 30 Days", "options": {0: "No", 1: "Yes"}, "default": 1},
    "Fruits": {"label": "Consumes Fruit >= 1 Time per Day", "options": {0: "No", 1: "Yes"}, "default": 1},
    "Veggies": {"label": "Consumes Vegetables >= 1 Time per Day", "options": {0: "No", 1: "Yes"}, "default": 1},
    "HvyAlcoholConsump": {"label": "Heavy Alcohol Consumption", "options": {0: "No", 1: "Yes"}, "default": 0},
    "AnyHealthcare": {"label": "Has Any Healthcare Coverage", "options": {0: "No", 1: "Yes"}, "default": 1},
    "NoDocbcCost": {"label": "Could Not See Doctor Due to Cost", "options": {0: "No", 1: "Yes"}, "default": 0},
    "GenHlth": {
        "label": "General Health Assessment",
        "options": {
            1: "Excellent",
            2: "Very Good",
            3: "Good",
            4: "Fair",
            5: "Poor"
        },
        "default": 2
    },
    "MentHlth": {"label": "Days with Poor Mental Health (Past 30 Days)", "min": 0, "max": 30, "default": 0, "unit": "days"},
    "PhysHlth": {"label": "Days of Physical Illness / Injury (Past 30 Days)", "min": 0, "max": 30, "default": 0, "unit": "days"},
    "DiffWalk": {"label": "Difficulty Walking or Climbing Stairs", "options": {0: "No", 1: "Yes"}, "default": 0},
    "Sex": {"label": "Sex", "options": {0: "Female", 1: "Male"}, "default": 1},
    "Age": {
        "label": "Age Bracket",
        "options": {
            1: "18 - 24", 2: "25 - 29", 3: "30 - 34", 4: "35 - 39",
            5: "40 - 44", 6: "45 - 49", 7: "50 - 54", 8: "55 - 59",
            9: "60 - 64", 10: "65 - 69", 11: "70 - 74", 12: "75 - 79",
            13: "80 and older"
        },
        "default": 7
    },
    "Education": {
        "label": "Highest Education Attained",
        "options": {
            1: "Never Attended Kindergarten",
            2: "Elementary (Grades 1-8)",
            3: "Some High School (Grades 9-11)",
            4: "High School Graduate / GED",
            5: "Some College or Technical School",
            6: "College 4+ Years (Graduate)"
        },
        "default": 5
    },
    "Income": {
        "label": "Annual Household Income Level",
        "options": {
            1: "Less than $10,000",
            2: "$10,000 to <$15,000",
            3: "$15,000 to <$20,000",
            4: "$20,000 to <$25,000",
            5: "$25,000 to <$35,000",
            6: "$35,000 to <$50,000",
            7: "$50,000 to <$75,000",
            8: "$75,000 or greater"
        },
        "default": 6
    }
}

# -------------------------------------------------------------
# 3. CHRONIC KIDNEY DISEASE CONFIGURATION (UCI CKD)
# -------------------------------------------------------------
KIDNEY_DATA_PATH = os.path.join(DATASETS_DIR, "kidney", "kidney_disease.csv")
KIDNEY_MODEL_PATH = os.path.join(MODELS_DIR, "kidney_disease_pipeline.joblib")
KIDNEY_METRICS_PATH = os.path.join(REPORTS_DIR, "kidney_metrics.json")

KIDNEY_NUMERIC_FEATURES = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", 
    "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"
]
KIDNEY_CATEGORICAL_FEATURES = [
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"
]
KIDNEY_ALL_FEATURES = KIDNEY_NUMERIC_FEATURES + KIDNEY_CATEGORICAL_FEATURES
KIDNEY_TARGET = "target"

KIDNEY_FEATURE_METADATA = {
    "age": {"label": "Age", "min": 2, "max": 100, "default": 52, "unit": "years"},
    "bp": {"label": "Blood Pressure", "min": 50, "max": 180, "default": 80, "unit": "mm Hg"},
    "sg": {"label": "Specific Gravity", "options": {1.005: "1.005", 1.010: "1.010", 1.015: "1.015", 1.020: "1.020", 1.025: "1.025"}, "default": 1.020},
    "al": {"label": "Albumin in Urine", "options": {0.0: "0 (None)", 1.0: "1", 2.0: "2", 3.0: "3", 4.0: "4", 5.0: "5 (High)"}, "default": 0.0},
    "su": {"label": "Sugar in Urine", "options": {0.0: "0 (None)", 1.0: "1", 2.0: "2", 3.0: "3", 4.0: "4", 5.0: "5 (High)"}, "default": 0.0},
    "bgr": {"label": "Random Blood Glucose", "min": 50, "max": 500, "default": 115, "unit": "mg/dl"},
    "bu": {"label": "Blood Urea", "min": 5, "max": 400, "default": 35, "unit": "mg/dl"},
    "sc": {"label": "Serum Creatinine", "min": 0.2, "max": 16.0, "default": 1.0, "step": 0.1, "unit": "mg/dl"},
    "sod": {"label": "Sodium (Na+)", "min": 100, "max": 165, "default": 138, "unit": "mEq/L"},
    "pot": {"label": "Potassium (K+)", "min": 2.0, "max": 10.0, "default": 4.5, "step": 0.1, "unit": "mEq/L"},
    "hemo": {"label": "Hemoglobin", "min": 3.0, "max": 19.0, "default": 14.2, "step": 0.1, "unit": "g/dl"},
    "pcv": {"label": "Packed Cell Volume (Hematocrit)", "min": 12, "max": 56, "default": 42, "unit": "%"},
    "wbcc": {"label": "White Blood Cell Count", "min": 2000, "max": 27000, "default": 7500, "unit": "cells/cumm"},
    "rbcc": {"label": "Red Blood Cell Count", "min": 2.0, "max": 8.0, "default": 5.0, "step": 0.1, "unit": "mil/cumm"},
    "rbc": {"label": "Red Blood Cells in Urine", "options": {"normal": "Normal", "abnormal": "Abnormal"}, "default": "normal"},
    "pc": {"label": "Pus Cells in Urine", "options": {"normal": "Normal", "abnormal": "Abnormal"}, "default": "normal"},
    "pcc": {"label": "Pus Cell Clumps", "options": {"notpresent": "Not Present", "present": "Present"}, "default": "notpresent"},
    "ba": {"label": "Bacteria in Urine", "options": {"notpresent": "Not Present", "present": "Present"}, "default": "notpresent"},
    "htn": {"label": "Hypertension History", "options": {"no": "No", "yes": "Yes"}, "default": "no"},
    "dm": {"label": "Diabetes Mellitus History", "options": {"no": "No", "yes": "Yes"}, "default": "no"},
    "cad": {"label": "Coronary Artery Disease History", "options": {"no": "No", "yes": "Yes"}, "default": "no"},
    "appet": {"label": "Appetite", "options": {"good": "Good", "poor": "Poor"}, "default": "good"},
    "pe": {"label": "Pedal Edema (Ankle Swelling)", "options": {"no": "No", "yes": "Yes"}, "default": "no"},
    "ane": {"label": "Anemia History", "options": {"no": "No", "yes": "Yes"}, "default": "no"}
}
