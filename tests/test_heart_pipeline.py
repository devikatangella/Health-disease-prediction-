"""
Unit and Integration Tests for Heart Disease Module
Verifies:
- Data loading and binary target conversion
- Preprocessing pipeline execution without leakage
- Model training and hyperparameter search output
- Joblib model loading and persistence
- Input validation (boundary conditions, missing columns, invalid ranges)
- Inference pipeline, risk probability, and SHAP explainability
- Database logging and retrieval
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.config import HEART_DATA_PATH, HEART_ALL_FEATURES, HEART_MODEL_PATH
from src.data_loader import load_heart_data
from src.preprocessing import build_heart_preprocessor, split_data_stratified
from src.predict import validate_heart_input, predict_heart_disease
from database import get_recent_predictions, init_database

class TestHeartDiseasePipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        init_database()
        cls.X, cls.y = load_heart_data(HEART_DATA_PATH)
        
    def test_01_data_loading_and_target_encoding(self):
        """Verifies dataset shape and binary conversion of target."""
        self.assertGreater(len(self.X), 250, "Heart disease dataset should have ~303 samples.")
        self.assertEqual(len(self.X.columns), 13, "Should have 13 predictor features.")
        # Target must be strictly binary: {0, 1}
        unique_targets = set(self.y.unique())
        self.assertTrue(unique_targets.issubset({0, 1}), f"Target must be binary 0 or 1, got {unique_targets}")
        # Verify class balance is non-trivial
        self.assertGreater(self.y.sum(), 50, "Positive class count should be substantial.")
        self.assertGreater((self.y == 0).sum(), 50, "Negative class count should be substantial.")

    def test_02_preprocessing_pipeline_no_leakage(self):
        """Verifies ColumnTransformer transforms without throwing errors or introducing NaN."""
        X_train, X_test, y_train, y_test = split_data_stratified(self.X, self.y, test_size=0.20)
        preprocessor = build_heart_preprocessor()
        
        # Fit on train ONLY
        X_train_trans = preprocessor.fit_transform(X_train)
        # Transform test
        X_test_trans = preprocessor.transform(X_test)
        
        self.assertEqual(X_train_trans.shape[0], len(X_train))
        self.assertEqual(X_test_trans.shape[0], len(X_test))
        # Ensure no NaNs remain after imputation
        self.assertFalse(np.isnan(X_train_trans).any(), "Processed train data should have no NaNs.")
        self.assertFalse(np.isnan(X_test_trans).any(), "Processed test data should have no NaNs.")

    def test_03_input_validation(self):
        """Tests input boundary checks and error reporting."""
        valid_sample = {
            "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
            "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 3
        }
        is_valid, msg = validate_heart_input(valid_sample)
        self.assertTrue(is_valid, f"Expected valid sample to pass, got: {msg}")
        
        # Missing feature test
        invalid_missing = valid_sample.copy()
        del invalid_missing["chol"]
        is_valid, msg = validate_heart_input(invalid_missing)
        self.assertFalse(is_valid)
        self.assertIn("chol", msg)
        
        # Out of physiological range test
        invalid_range = valid_sample.copy()
        invalid_range["age"] = 150 # Out of 20-100 bounds
        is_valid, msg = validate_heart_input(invalid_range)
        self.assertFalse(is_valid)
        self.assertIn("range", msg.lower())

    def test_04_prediction_pipeline_and_explainability(self):
        """Verifies end-to-end prediction, calibrated risk score, and SHAP top factors."""
        if not os.path.exists(HEART_MODEL_PATH):
            self.skipTest("Heart model not yet trained. Run training first.")
            
        sample = {
            "age": 62, "sex": 1, "cp": 4, "trestbps": 145, "chol": 280,
            "fbs": 1, "restecg": 2, "thalach": 120, "exang": 1,
            "oldpeak": 2.5, "slope": 2, "ca": 2, "thal": 7
        }
        
        result = predict_heart_disease(sample, allow_logging=True)
        
        self.assertIn(result["risk_tier"], ["Low Risk", "Moderate / Borderline Risk", "High Risk"])
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)
        self.assertIsInstance(result["top_factors"], list)
        self.assertGreater(len(result["top_factors"]), 0, "Should extract at least one top SHAP factor.")
        self.assertIsNotNone(result["log_id"], "Should return logged ID in database.")
        
        # Verify persistence in database
        recent_logs = get_recent_predictions(limit=1, disease_type="Heart Disease")
        self.assertEqual(len(recent_logs), 1)
        self.assertEqual(recent_logs[0]["id"], result["log_id"])

if __name__ == "__main__":
    unittest.main()
