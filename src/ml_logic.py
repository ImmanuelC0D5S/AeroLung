import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
import joblib # To be used for saving/loading the model later

"""
AeroLung ML Logic
Feature extraction and classification skeleton for digital stethoscope.
Waiting for user-provided datasets before implementing training logic.
"""

def detect_crackles_rms(data, frame_length=2048, hop_length=512, threshold=0.1):
    """
    # [EXPERIMENT 3]
    Implement Anomaly Detection logic that calculates Time-Domain RMS energy 
    to find "Crackles".
    """
    rms = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)[0]
    # Simple thresholding to find high-energy peaks (crackles)
    anomalies = np.where(rms > (np.mean(rms) + threshold * np.std(rms)))[0]
    return rms, anomalies

def extract_mfcc_features(data, fs, n_mfcc=13):
    """
    # TECHNICAL REQUIREMENT 3/4
    Extract 13 MFCC features for the Random Forest model.
    """
    mfccs = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=n_mfcc)
    # Return the mean of each MFCC coefficient across time frames
    return np.mean(mfccs, axis=1)

class LungClassifier:
    """
    Skeleton for the Healthy vs Wheeze Classifier.
    Will be trained once real data is available.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.is_trained = False

    def train(self, X, y):
        """
        Train the model using provided features and labels.
        """
        # [WAITING FOR USER DATASETS]
        self.model.fit(X, y)
        self.is_trained = True
        print("Model trained successfully.")

    def load_model(self, model_path):
        """
        Load a trained model from a joblib file.
        """
        try:
            self.model = joblib.load(model_path)
            self.is_trained = True
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")

    def predict(self, features):
        """
        Predict specific lung condition based on extracted MFCC features.
        """
        if not self.is_trained:
            return "UNTRAINED (Awaiting Dataset)"
        
        prediction = self.model.predict(features.reshape(1, -1))
        
        # Mapping numerical predictions back to user-defined specific labels
        label_map = {
            0: 'Condition: Healthy (Normal)',
            1: 'Diagnosis: Asthma (Wheezing)',
            2: 'Diagnosis: Pneumonia/COPD (Crackles)',
            3: 'Diagnosis: COPD (Mixed Symptoms)'
        }
        return label_map.get(prediction[0], "Unknown Condition")

def check_lung_health(data, fs, mfcc_features, crackles_count):
    """
    Combined logic for the status card.
    If no model is trained, use heuristic for now.
    """
    # Heuristic: IF crackles are detected OR wheeze peaks are too high
    # This will be replaced by the Random Forest model later.
    if crackles_count > 5:
        return "ANOMALY DETECTED (High Crackle Count)"
    
    # Placeholder for model-based prediction
    return "HEALTHY (Heuristic)"
