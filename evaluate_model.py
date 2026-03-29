import os
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from train_model import prepare_dataset
from src.ml_logic import LungClassifier

MODEL_PATH = "models/lung_model.joblib"

def evaluate():
    print("--- AeroLung Model Evaluation ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Run train_model.py first.")
        return

    # 1. Prepare Dataset
    X, y = prepare_dataset()
    
    if len(X) == 0:
        print("Error: No data found in 'data/' folder.")
        return

    # 2. Load Model
    classifier = LungClassifier()
    classifier.load_model(MODEL_PATH)

    # 3. Perform Prediction on full set (for diagnostic overview)
    y_pred = classifier.model.predict(X)
    
    # 4. Generate Report
    target_names = ['Healthy', 'Asthma', 'Pneumonia/COPD', 'COPD (Mixed)']
    
    # Filter target names based on classes actually present in y
    present_classes = np.unique(np.concatenate([y, y_pred]))
    filtered_target_names = [target_names[i] for i in present_classes]

    print("\n[Classification Report]")
    print(classification_report(y, y_pred, target_names=filtered_target_names))
    
    print("\n[Confusion Matrix]")
    print(confusion_matrix(y, y_pred))
    
    acc = accuracy_score(y, y_pred)
    print(f"\nFinal Training Accuracy: {acc*100:.2f}%")
    
    print("\n--- Recommendation ---")
    if len(X) < 40:
        print("WARNING: Dataset is too small (<10 samples per class).")
        print("The high accuracy likely indicates OVERFITTING.")
        print("Please provide 20+ more samples per category for clinical reliability.")
    else:
        print("Dataset size is improving. Continue monitoring F1-score.")

if __name__ == "__main__":
    evaluate()