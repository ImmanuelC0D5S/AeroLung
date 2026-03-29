import os
import numpy as np
import joblib
import librosa
from src.dsp_engine import apply_bandpass_filter, load_audio
from src.ml_logic import extract_mfcc_features, LungClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter

# Configuration
DATA_DIR = "data"
MODEL_PATH = "models/lung_model.joblib"

def parse_annotation(txt_path):
    """
    Parses ICBHI annotation file.
    Returns list of cycles: (start, end, crackles, wheezes)
    """
    cycles = []
    if not os.path.exists(txt_path):
        return cycles
        
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                # ICBHI annotations are tab-separated: Start, End, Crackles, Wheezes
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        crackles = int(parts[2])
                        wheezes = int(parts[3])
                        cycles.append((start, end, crackles, wheezes))
                    except ValueError:
                        # Skip header if present or malformed lines
                        continue
    except Exception as e:
        print(f"Error parsing {txt_path}: {e}")
        
    return cycles

def get_label(crackles, wheezes):
    """
    Map crackle/wheeze flags to logical diagnostic labels.
    - 0: Healthy
    - 1: Wheeze (Asthma-like)
    - 2: Crackle (Pneumonia/COPD-like)
    - 3: Both (Mixed/Chronic)
    """
    if crackles == 0 and wheezes == 0:
        return 0 # Healthy
    elif crackles == 0 and wheezes == 1:
        return 1 # Asthma/Wheeze
    elif crackles == 1 and wheezes == 0:
        return 2 # Pneumonia/Crackle
    else:
        return 3 # Both (Mixed)

def prepare_dataset():
    X = []
    y = []
    
    # Filter for wav files
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
    print(f"Found {len(files)} potential audio files in {DATA_DIR}.")
    
    for filename in files:
        wav_path = os.path.join(DATA_DIR, filename)
        txt_path = wav_path.replace(".wav", ".txt")
        
        # We need an annotation file to determine cycle labels
        if not os.path.exists(txt_path):
            continue
            
        cycles = parse_annotation(txt_path)
        if not cycles:
            # print(f"Skipping {filename}: No cycles found in annotation.")
            continue
            
        try:
            # Load and preprocess full audio once for efficiency
            data, fs = load_audio(wav_path)
            filtered_data = apply_bandpass_filter(data, fs)
            
            for start, end, c, w in cycles:
                # 1. Segment Audio based on timestamps
                start_idx = int(start * fs)
                end_idx = int(end * fs)
                
                # Validation: Skip extremely short segments (< 0.5 seconds)
                if (end_idx - start_idx) < (0.5 * fs):
                    continue
                    
                # Handle edge cases for end_idx
                if end_idx > len(filtered_data):
                    end_idx = len(filtered_data)
                    
                segment = filtered_data[start_idx:end_idx]
                
                if len(segment) == 0:
                    continue
                
                # 2. Extract 53 Enhanced Features (from ml_logic)
                features = extract_mfcc_features(segment, fs)
                
                # 3. Assign Label
                label = get_label(c, w)
                
                X.append(features)
                y.append(label)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    return np.array(X), np.array(y)

def oversample_data(X, y):
    """
    Performs random oversampling on minority classes to address ICBHI data skew.
    """
    counts = Counter(y)
    if not counts:
        return X, y
        
    max_count = max(counts.values())
    
    X_resampled = []
    y_resampled = []
    
    for label in sorted(counts.keys()):
        indices = np.where(y == label)[0]
        label_X = X[indices]
        label_y = y[indices]
        
        # Replicate samples to match the majority class count
        num_to_add = max_count - len(indices)
        if num_to_add > 0:
            extra_indices = np.random.choice(indices, size=num_to_add, replace=True)
            label_X = np.concatenate([label_X, X[extra_indices]])
            label_y = np.concatenate([label_y, y[extra_indices]])
            
        X_resampled.append(label_X)
        y_resampled.append(label_y)
        
    return np.concatenate(X_resampled), np.concatenate(y_resampled)

def train():
    print("--- AeroLung Advanced Training (ICBHI Mode) ---")
    print("Preparing dataset (extracting cycles per annotation)...")
    
    X, y = prepare_dataset()
    
    if len(X) == 0:
        print("Error: No valid cycles extracted. Ensure .wav and .txt pairs are present.")
        return
        
    print(f"Dataset summary: {len(X)} segments extracted.")
    print(f"Original Label distribution: {Counter(y)}")
    
    # 0: Healthy, 1: Asthma, 2: Pneumonia, 3: COPD Mixed
    
    # Perform stratified split (if possible)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback if some classes have only 1 sample
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"Training on {len(X_train)} segments, Testing on {len(X_test)} segments...")
    
    # Apply Oversampling to training set to handle imbalance
    X_train_res, y_train_res = oversample_data(X_train, y_train)
    print(f"Resampled training distribution: {Counter(y_train_res)}")
    
    classifier = LungClassifier()
    # Note: LungClassifier is already configured with class_weight='balanced'
    classifier.model.fit(X_train_res, y_train_res)
    
    # 4. Metric Report
    y_pred = classifier.model.predict(X_test)
    target_names = ['Healthy', 'Asthma (Wheeze)', 'Pueumonia (Crackle)', 'COPD (Both)']
    
    # Only include labels present in the test set
    labels_present = np.unique(np.concatenate([y_test, y_pred]))
    filtered_names = [target_names[i] for i in labels_present]

    print("\n--- Model Performance Report ---")
    print(classification_report(y_test, y_pred, target_names=filtered_names))
    
    # 5. Production Retrain
    print("Retraining on full dataset with oversampling for production...")
    X_full_res, y_full_res = oversample_data(X, y)
    classifier.train(X_full_res, y_full_res)
    
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(classifier.model, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")

if __name__ == "__main__":
    train()
