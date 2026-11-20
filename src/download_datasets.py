"""
Dataset Downloader & Local Cache for Smart Healthcare Disease Prediction System
Fetches authentic public datasets directly from UCI Machine Learning Repository via ucimlrepo.
1. Heart Disease (UCI Cleveland, id=45)
2. CDC Diabetes Health Indicators (UCI id=891)
3. Chronic Kidney Disease (UCI id=336)
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

def download_and_cache_datasets():
    print("=" * 60)
    print("Fetching authentic datasets from UCI Machine Learning Repository...")
    print("=" * 60)

    # 1. Heart Disease (UCI Cleveland id=45)
    print("\n[1/3] Fetching Heart Disease Dataset (UCI ID 45)...")
    heart_dir = os.path.join(DATASETS_DIR, "heart")
    os.makedirs(heart_dir, exist_ok=True)
    heart_csv = os.path.join(heart_dir, "heart_disease.csv")
    
    heart_repo = fetch_ucirepo(id=45)
    df_heart_X = heart_repo.data.features
    df_heart_y = heart_repo.data.targets
    df_heart = pd.concat([df_heart_X, df_heart_y], axis=1)
    df_heart.rename(columns={"num": "target"}, inplace=True)
    df_heart.to_csv(heart_csv, index=False)
    print(f"Saved Heart Disease dataset to {heart_csv} (Shape: {df_heart.shape})")
    print("Heart Disease Columns:", list(df_heart.columns))
    print("Heart Disease Target Distribution (raw 0-4):", df_heart["target"].value_counts().to_dict())

    # 2. Chronic Kidney Disease (UCI id=336)
    print("\n[2/3] Fetching Chronic Kidney Disease Dataset (UCI ID 336)...")
    kidney_dir = os.path.join(DATASETS_DIR, "kidney")
    os.makedirs(kidney_dir, exist_ok=True)
    kidney_csv = os.path.join(kidney_dir, "kidney_disease.csv")
    
    ckd_repo = fetch_ucirepo(id=336)
    df_ckd_X = ckd_repo.data.features
    df_ckd_y = ckd_repo.data.targets
    df_ckd = pd.concat([df_ckd_X, df_ckd_y], axis=1)
    # Target column is named 'class'
    df_ckd.rename(columns={"class": "target"}, inplace=True)
    df_ckd.to_csv(kidney_csv, index=False)
    print(f"Saved Kidney Disease dataset to {kidney_csv} (Shape: {df_ckd.shape})")
    print("Kidney Disease Columns:", list(df_ckd.columns))
    print("Kidney Disease Raw Target Distribution:", df_ckd["target"].value_counts().to_dict())

    # 3. CDC Diabetes Health Indicators (UCI id=891)
    print("\n[3/3] Fetching CDC Diabetes Health Indicators Dataset (UCI ID 891)...")
    diabetes_dir = os.path.join(DATASETS_DIR, "diabetes")
    os.makedirs(diabetes_dir, exist_ok=True)
    diabetes_csv = os.path.join(diabetes_dir, "diabetes_cdc.csv")
    
    cdc_repo = fetch_ucirepo(id=891)
    df_cdc_X = cdc_repo.data.features
    df_cdc_y = cdc_repo.data.targets
    df_cdc = pd.concat([df_cdc_X, df_cdc_y], axis=1)
    df_cdc.rename(columns={"Diabetes_binary": "target"}, inplace=True)
    
    # Save a high-quality balanced/stratified subset for optimal training speed & local responsiveness (50,000 samples)
    # alongside preserving full dataset option
    print(f"Full CDC Diabetes Shape: {df_cdc.shape}")
    print("Raw CDC Diabetes Target Distribution:", df_cdc["target"].value_counts().to_dict())
    
    # Save stratified sample of 50,000 samples for balanced training efficiency
    # (or 70,692 50/50 balanced which is standard for CDC BRFSS benchmark)
    n_pos = (df_cdc["target"] == 1).sum() # 35,346
    pos_samples = df_cdc[df_cdc["target"] == 1]
    neg_samples = df_cdc[df_cdc["target"] == 0].sample(n=n_pos, random_state=42)
    df_cdc_balanced = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_cdc_balanced.to_csv(diabetes_csv, index=False)
    print(f"Saved balanced CDC Diabetes dataset (70,692 rows) to {diabetes_csv}")
    
    print("\nAll 3 datasets successfully fetched and stored locally!")

if __name__ == "__main__":
    download_and_cache_datasets()
