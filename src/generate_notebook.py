"""
Script to create the educational research notebook in notebooks/
"""
import json
import os

notebook_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks", "healthcare_eda_and_benchmarks.ipynb"))

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Smart Healthcare Disease Prediction System\n",
                "## End-to-End Exploratory Data Analysis & Machine Learning Benchmarks\n",
                "\n",
                "**Author:** Senior Data Scientist & ML Engineer  \n",
                "**Diseases Covered:**  \n",
                "1. Cardiovascular Disease (UCI Cleveland Heart Disease)  \n",
                "2. Type 2 Diabetes (CDC BRFSS Health Indicators)  \n",
                "3. Chronic Kidney Disease (UCI CKD Laboratory Panel)  \n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Imports\n",
                "import os\n",
                "import sys\n",
                "sys.path.append('..')\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from src.data_loader import load_heart_data, load_diabetes_data, load_kidney_data\n",
                "from src.predict import predict_disease\n",
                "sns.set_theme(style='whitegrid', palette='deep')\n",
                "print('Environment and modules loaded successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Heart Disease Exploration (UCI Cleveland)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_heart, y_heart = load_heart_data()\n",
                "df_heart = pd.concat([X_heart, y_heart.rename('target')], axis=1)\n",
                "print(f'Heart Dataset: {df_heart.shape[0]} rows, {df_heart.shape[1]} columns')\n",
                "print('Target Distribution:\\n', y_heart.value_counts(normalize=True))\n",
                "df_heart.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Diabetes Exploration (CDC BRFSS)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_diab, y_diab = load_diabetes_data(sample_size=1000)\n",
                "df_diab = pd.concat([X_diab, y_diab.rename('target')], axis=1)\n",
                "print(f'Diabetes Dataset: {df_diab.shape[0]} rows, {df_diab.shape[1]} columns')\n",
                "print('Target Distribution:\\n', y_diab.value_counts(normalize=True))\n",
                "df_diab.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Chronic Kidney Disease Exploration (UCI CKD)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_ckd, y_ckd = load_kidney_data()\n",
                "df_ckd = pd.concat([X_ckd, y_ckd.rename('target')], axis=1)\n",
                "print(f'CKD Dataset: {df_ckd.shape[0]} rows, {df_ckd.shape[1]} columns')\n",
                "print('Target Distribution:\\n', y_ckd.value_counts(normalize=True))\n",
                "df_ckd.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Interactive Prediction Demonstration via Modular Pipeline"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sample_patient = {\n",
                "    'age': 63, 'sex': 1, 'cp': 1, 'trestbps': 145, 'chol': 233,\n",
                "    'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,\n",
                "    'oldpeak': 2.3, 'slope': 3, 'ca': 0, 'thal': 6\n",
                "}\n",
                "res = predict_disease('heart', sample_patient)\n",
                "print('Risk Assessment:', res['prediction_label'])\n",
                "print('Probability:', res['probability_percent'], '%')\n",
                "print('Risk Category:', res['risk_tier'])\n",
                "print('Top Influencing Biomarkers (SHAP):', res['top_factors'])"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print("Saved notebook to", notebook_path)
