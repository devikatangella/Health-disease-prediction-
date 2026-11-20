"""
Accuracy Verification Script: Compares Model Predictions vs Ground Truth on Real Test Cases
"""

import pandas as pd
from src.data_loader import load_heart_data, load_diabetes_data, load_kidney_data
from src.preprocessing import split_data_stratified
from src.predict import predict_disease

def test_disease_cases(disease_name, load_func, n_cases=6):
    print("\n" + "=" * 70)
    print(f"VERIFYING ACCURACY ON UNTOUCHED TEST SET: {disease_name.upper()}")
    print("=" * 70)
    
    X, y = load_func()
    _, X_test, _, y_test = split_data_stratified(X, y, test_size=0.20, random_state=42)
    
    correct_count = 0
    total = min(n_cases, len(X_test))
    
    for i in range(total):
        sample = X_test.iloc[i].to_dict()
        true_label = int(y_test.iloc[i])
        
        disease_key = "heart" if "Heart" in disease_name else ("diabetes" if "Diabetes" in disease_name else "kidney")
        res = predict_disease(disease_key, sample, allow_logging=False)
        pred_label = res["raw_prediction"]
        prob = res["probability_percent"]
        
        is_match = (pred_label == true_label)
        if is_match:
            correct_count += 1
            
        status = "CORRECT" if is_match else "MISCLASSIFIED"
        label_map = {0: "Negative (Healthy / No Disease)", 1: "Positive (Disease Present)"}
        
        print(f"\n[Test Case #{i+1}]")
        print(f"  • Ground Truth (Actual) : {label_map[true_label]}")
        print(f"  • Model Prediction      : {label_map[pred_label]}")
        print(f"  • Estimated Risk Prob   : {prob}% ({res['risk_tier']})")
        print(f"  • Verification Outcome  : {status}")
        
        # Print key values for the user
        if disease_key == "heart":
            print(f"  • Key Patient Vitals    : Age={sample['age']}, Sex={'Male' if sample['sex']==1 else 'Female'}, ChestPain={sample['cp']}, RestingBP={sample['trestbps']}, MaxHR={sample['thalach']}, Vessels(ca)={sample['ca']}")
        elif disease_key == "diabetes":
            print(f"  • Key Patient Vitals    : HighBP={sample['HighBP']}, HighChol={sample['HighChol']}, BMI={sample['BMI']}, GenHlth={sample['GenHlth']}, AgeBracket={sample['Age']}")
        elif disease_key == "kidney":
            print(f"  • Key Patient Vitals    : Creatinine={sample['sc']}, Hemoglobin={sample['hemo']}, Albumin={sample['al']}, BP={sample['bp']}, SpecificGravity={sample['sg']}")

    acc_pct = (correct_count / total) * 100
    print(f"\nSummary: {correct_count}/{total} cases correctly classified ({acc_pct:.1f}% accuracy on this test batch).")

if __name__ == "__main__":
    test_disease_cases("Heart Disease", load_heart_data, n_cases=6)
    test_disease_cases("Diabetes", lambda: load_diabetes_data(sample_size=2000), n_cases=6)
    test_disease_cases("Chronic Kidney Disease", load_kidney_data, n_cases=6)
