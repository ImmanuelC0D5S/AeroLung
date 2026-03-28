import os
import numpy as np
import joblib
import librosa
from src.dsp_engine import apply_bandpass_filter, load_audio
from src.ml_logic import extract_mfcc_features, LungClassifier

# Define the ID to Label mapping as per user instructions
ID_MAPPING = {
    "101": 0, "102": 0, # Healthy
    "107": 1, "226": 1, # Asthma
    "130": 2, "154": 2, # Pneumonia/COPD
    "104": 3, "106": 3  # COPD Mixed
}

DATA_DIR = "data"
MODEL_PATH = "models/lung_model.joblib"

def prepare_dataset():
    X = []
    y = []
    
    # List all wav files in data directory
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
    
    print(f"Found {len(files)} potential audio files.")
    
    for filename in files:
        # Extract ID from filename (e.g., 101_1b1_Al_sc_Meditron.wav)
        file_id = filename.split("_")[0]
        
        if file_id in ID_MAPPING:
            label = ID_MAPPING[file_id]
            file_path = os.path.join(DATA_DIR, filename)
            
            try:
                # 1. Load Audio
                data, fs = load_audio(file_path)
                
                # 2. Preprocess: Bandpass Filter (200-2000Hz)
                # ISO/Technical standard for lung sound isolation
                filtered_data = apply_bandpass_filter(data, fs)
                
                # 3. Feature Extraction: 13 MFCCs
                features = extract_mfcc_features(filtered_data, fs)
                
                X.append(features)
                y.append(label)
                print(f"Processed {filename} -> Label {label}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    return np.array(X), np.array(y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def train():
    print("Preparing dataset...")
    X, y = prepare_dataset()
    
    if len(X) < 4:
        print("Not enough data to perform evaluation. Need at least 4 samples.")
        return
        
    print(f"Dataset summary: {len(X)} samples, {len(np.unique(y))} classes.")
    
    # Since we have very few samples (10), a split is risky but let's do 80/20
    # for a quick metric check.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples...")
    
    classifier = LungClassifier()
    classifier.model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n--- Model Metrics ---")
    print(f"Accuracy: {acc:.2f}")
    print(f"F1 Score (Weighted): {f1:.2f}")
    print(f"----------------------\n")
    
    # Retrain on full dataset for production
    print("Retraining on full dataset...")
    classifier.train(X, y)
    
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(classifier.model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
