"""
Explainable AI (XAI) Module using SHAP
Provides local (instance-level) and global (population-level) feature attribution.
Includes standard medical safety disclaimers:
Feature importance reflects statistical model attribution and does NOT establish clinical causation.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple

class DiseaseModelExplainer:
    """
    Unified SHAP Explainer supporting Tree-based models (Random Forest, XGBoost, Decision Tree)
    and Linear models (Logistic Regression).
    """
    def __init__(self, pipeline, X_train: pd.DataFrame, feature_names: Optional[List[str]] = None):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps.get("preprocessor", None)
        self.classifier = pipeline.named_steps.get("classifier", None)
        
        # Transform training background data
        if self.preprocessor:
            self.X_train_transformed = self.preprocessor.transform(X_train)
            # Try to get transformed feature names
            try:
                self.feature_names = list(self.preprocessor.get_feature_names_out())
            except Exception:
                self.feature_names = [f"f_{i}" for i in range(self.X_train_transformed.shape[1])]
        else:
            self.X_train_transformed = X_train.values
            self.feature_names = feature_names or list(X_train.columns)
            
        self.explainer = None
        self._init_explainer()
        
    def _init_explainer(self):
        """Initializes the most appropriate SHAP explainer."""
        try:
            # Tree-based
            if hasattr(self.classifier, "feature_importances_"):
                self.explainer = shap.TreeExplainer(self.classifier)
            elif hasattr(self.classifier, "coef_"):
                # Linear models
                self.explainer = shap.LinearExplainer(self.classifier, self.X_train_transformed)
            else:
                # Fallback to Permutation / Explainer
                bg_sample = shap.sample(self.X_train_transformed, 50, random_state=42)
                self.explainer = shap.Explainer(self.classifier.predict_proba, bg_sample)
        except Exception as e:
            print(f"SHAP initialization note: {e}. Falling back to Kernel / Sample explainer.")
            bg_sample = shap.sample(self.X_train_transformed, min(50, len(self.X_train_transformed)), random_state=42)
            self.explainer = shap.Explainer(self.classifier.predict_proba, bg_sample)
            
    def explain_instance(self, sample_df: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Calculates local SHAP feature contributions for a single patient record.
        Returns top positive and negative contributing factors.
        """
        if self.preprocessor:
            sample_transformed = self.preprocessor.transform(sample_df)
        else:
            sample_transformed = sample_df.values
            
        try:
            shap_values = self.explainer(sample_transformed)
            
            # Handle different SHAP output formats (binary class output)
            if hasattr(shap_values, "values"):
                vals = shap_values.values[0]
                if len(vals.shape) == 2: # [features, classes]
                    vals = vals[:, 1]
            elif isinstance(shap_values, list) and len(shap_values) == 2:
                vals = shap_values[1][0]
            else:
                vals = np.array(shap_values)[0]
                
            contributions = []
            for name, val in zip(self.feature_names, vals):
                # Clean up feature name (e.g. 'num__age' -> 'Age', 'cat__sex_1' -> 'Sex (Male)')
                clean_name = name.replace("num__", "").replace("cat__", "")
                contributions.append({
                    "feature": clean_name,
                    "raw_feature": name,
                    "shap_value": float(val),
                    "impact": "Increases Risk" if val > 0 else "Decreases Risk",
                    "magnitude": abs(float(val))
                })
                
            # Sort by absolute impact
            contributions.sort(key=lambda x: x["magnitude"], reverse=True)
            return contributions[:top_k]
            
        except Exception as e:
            print(f"Error computing local SHAP explanation: {e}")
            # Fallback to model feature importances
            return self._fallback_instance_explanation(sample_df, top_k)
            
    def _fallback_instance_explanation(self, sample_df: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback feature importance calculation."""
        factors = []
        if hasattr(self.classifier, "feature_importances_"):
            importances = self.classifier.feature_importances_
            for name, imp in zip(self.feature_names, importances):
                clean_name = name.replace("num__", "").replace("cat__", "")
                factors.append({
                    "feature": clean_name,
                    "raw_feature": name,
                    "shap_value": float(imp),
                    "impact": "Model Weight",
                    "magnitude": float(imp)
                })
        elif hasattr(self.classifier, "coef_"):
            coefs = self.classifier.coef_[0]
            for name, coef in zip(self.feature_names, coefs):
                clean_name = name.replace("num__", "").replace("cat__", "")
                factors.append({
                    "feature": clean_name,
                    "raw_feature": name,
                    "shap_value": float(coef),
                    "impact": "Increases Risk" if coef > 0 else "Decreases Risk",
                    "magnitude": abs(float(coef))
                })
        factors.sort(key=lambda x: x["magnitude"], reverse=True)
        return factors[:top_k]

    def plot_local_explanation(self, contributions: List[Dict[str, Any]], title: str = "Key Influencing Factors (SHAP Local Attribution)") -> plt.Figure:
        """Generates a clean horizontal bar chart of local SHAP contributions."""
        fig, ax = plt.subplots(figsize=(7, 4))
        
        features = [c["feature"] for c in reversed(contributions)]
        values = [c["shap_value"] for c in reversed(contributions)]
        colors = ["#ef4444" if v > 0 else "#10b981" for v in values] # Red for risk increase, green for protective
        
        bars = ax.barh(features, values, color=colors, height=0.55)
        ax.axvline(0, color="#64748b", linestyle="--", linewidth=1.2)
        
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xlabel("SHAP Attribution (Log-Odds Impact)", fontweight="semibold")
        
        # Add labels on bars
        for bar, val in zip(bars, values):
            offset = 0.02 if val >= 0 else -0.02
            ha = "left" if val >= 0 else "right"
            ax.annotate(f"{val:+.3f}", (val + offset, bar.get_y() + bar.get_height()/2),
                        ha=ha, va="center", fontsize=9, fontweight="bold")
                        
        plt.tight_layout()
        return fig
