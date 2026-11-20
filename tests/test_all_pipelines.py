"""
Unit and Integration Tests for All Disease Pipelines
Verifies Heart Disease, Diabetes, and Chronic Kidney Disease modules.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.config import (
    HEART_DATA_PATH, HEART_ALL_FEATURES, HEART_MODEL_PATH,
    DIABETES_DATA_PATH, DIABETES_ALL_FEATURES, DIABETES_MODEL_PATH,
    KIDNEY_DATA_PATH, KIDNEY_ALL_FEATURES, KIDNEY_MODEL_PATH
)
from src.data_loader import load_heart_data, load_diabetes_data, load_kidney_data
from src.predict import validate_disease_input, predict_disease
from database import init_database, get_recent_predictions, get_model_metadata_history

class TestAllDiseasePipelines(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        init_database()
        
    def test_01_heart_dataset_and_prediction(self):
        """Verifies heart disease pipeline end-to-end."""
        self.assertTrue(os.path.exists(HEART_MODEL_PATH), "Heart disease model must exist.")
        sample = {
            "age": 60, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260,
            "fbs": 0, "restecg": 0, "thalach": 140, "exang": 1,
            "oldpeak": 2.0, "slope": 2, "ca": 1, "thal": 7
        }
        is_valid, msg = validate_disease_input("heart", sample)
        self.assertTrue(is_valid, f"Heart input valid: {msg}")
        
        res = predict_disease("heart", sample)
        self.assertIn(res["risk_tier"], ["Low Risk", "Moderate / Borderline Risk", "High Risk"])
        self.assertGreaterEqual(res["probability"], 0.0)
        self.assertLessEqual(res["probability"], 1.0)
        self.assertGreater(len(res["top_factors"]), 0)

    def test_02_diabetes_dataset_and_prediction(self):
        """Verifies diabetes pipeline end-to-end."""
        self.assertTrue(os.path.exists(DIABETES_MODEL_PATH), "Diabetes model must exist.")
        sample = {
            "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 32, "Smoker": 0,
            "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0,
            "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
            "GenHlth": 4, "MentHlth": 5, "PhysHlth": 10, "DiffWalk": 1, "Sex": 1,
            "Age": 9, "Education": 4, "Income": 5
        }
        is_valid, msg = validate_disease_input("diabetes", sample)
        self.assertTrue(is_valid, f"Diabetes input valid: {msg}")
        
        res = predict_disease("diabetes", sample)
        self.assertIn(res["risk_tier"], ["Low Risk", "Moderate / Borderline Risk", "High Risk"])
        self.assertGreaterEqual(res["probability"], 0.0)
        self.assertLessEqual(res["probability"], 1.0)
        self.assertGreater(len(res["top_factors"]), 0)

    def test_03_kidney_dataset_and_prediction(self):
        """Verifies chronic kidney disease pipeline end-to-end."""
        self.assertTrue(os.path.exists(KIDNEY_MODEL_PATH), "Kidney model must exist.")
        sample = {
            "age": 55, "bp": 80, "sg": 1.015, "al": 2.0, "su": 0.0,
            "bgr": 140, "bu": 45, "sc": 2.2, "sod": 135, "pot": 4.8,
            "hemo": 10.5, "pcv": 32, "wbcc": 8500, "rbcc": 3.8,
            "rbc": "normal", "pc": "abnormal", "pcc": "present", "ba": "notpresent",
            "htn": "yes", "dm": "yes", "cad": "no", "appet": "poor", "pe": "yes", "ane": "yes"
        }
        is_valid, msg = validate_disease_input("kidney", sample)
        self.assertTrue(is_valid, f"Kidney input valid: {msg}")
        
        res = predict_disease("kidney", sample)
        self.assertIn(res["risk_tier"], ["Low Risk", "Moderate / Borderline Risk", "High Risk"])
        self.assertGreaterEqual(res["probability"], 0.0)
        self.assertLessEqual(res["probability"], 1.0)
        self.assertGreater(len(res["top_factors"]), 0)

    def test_04_database_model_registry(self):
        """Verifies all models are logged in SQLite metadata registry."""
        history = get_model_metadata_history()
        self.assertGreaterEqual(len(history), 12, "Should have registered runs for all algorithms across diseases.")

if __name__ == "__main__":
    unittest.main()
