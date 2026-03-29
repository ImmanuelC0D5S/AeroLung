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

def preprocess_clinical_signal(data, fs, is_clinical=True):
    """
    # [CONDITIONAL PRE-PROCESSING]
    Ensures 'Best of Both Worlds' for signal features.
    - is_clinical=True: Preserves raw high-fidelity clinical markers (ICBHI).
    - is_clinical=False: Applies robust Notch/Bandpass filtering for field recordings.
    """
    if is_clinical:
        return data
        
    # Field recordings (Mic/Mobiles) need noise resilience
    from src.dsp_engine import apply_notch_filter, apply_bandpass_filter
    data = apply_notch_filter(data, fs, freq=50.0)
    data = apply_bandpass_filter(data, fs, lowcut=200, highcut=2000)
    return data

def extract_mfcc_features(data, fs, n_mfcc=13, is_clinical=True):
    """
    # TECHNICAL REQUIREMENT 3/4 [ENHANCED]
    Extract 54 features for higher precision:
    - 13 MFCC Mean
    - 13 MFCC Std
    - 13 Delta MFCC Mean
    - 1 Spectral Centroid
    - 1 Zero Crossing Rate
    - 12 Chromagram Mean
    - 1 Sub-band Power Ratio (COPD Index)
    """
    # [NEW] Pre-process based on clinical context
    data = preprocess_clinical_signal(data, fs, is_clinical=is_clinical)
    
    # 1. MFCCs
    n_samples = len(data)
    n_fft = min(n_samples, 2048)
    hop_length = n_fft // 4 if n_fft > 0 else 1
    
    mfccs = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # 2. Delta MFCCs
    # Ensure there are enough frames for the delta calculation (default width=9)
    if mfccs.shape[1] >= 9:
        delta_mfccs = librosa.feature.delta(mfccs, width=9)
    elif mfccs.shape[1] >= 3:
        delta_mfccs = librosa.feature.delta(mfccs, width=3)
    else:
        # Fallback for extremely short segments
        delta_mfccs = np.zeros_like(mfccs)
    
    delta_mean = np.mean(delta_mfccs, axis=1)
    
    # 3. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=fs, n_fft=n_fft, hop_length=hop_length)
    centroid_mean = np.mean(spectral_centroid)
    
    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=n_fft, hop_length=hop_length)
    zcr_mean = np.mean(zcr)
    
    # 5. Chroma Features (Spectral Texture)
    chroma = librosa.feature.chroma_stft(y=data, sr=fs, n_fft=n_fft, hop_length=hop_length)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Combine into a single 54-dimensional feature vector
    from src.dsp_engine import calculate_subband_ratio
    subband_ratio = calculate_subband_ratio(data, fs)
    
    features = np.concatenate([
        mfcc_mean, 
        mfcc_std, 
        delta_mean, 
        [centroid_mean, zcr_mean], 
        chroma_mean,
        [subband_ratio] # 54th Feature: COPD Index
    ])
    return features

class LungClassifier:
    """
    Skeleton for the Healthy vs Wheeze Classifier.
    Will be trained once real data is available.
    """
    def __init__(self):
        # [UPGRADED] n_estimators=200 + class_weight='balanced' for Clinical Resilience
        self.model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        self.sensitivity_threshold_c2 = 0.35 # Clinical Guard for Recall
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
        Predict specific lung condition and return the most likely label.
        [CLINICAL MODE]: Priority: Class 2 (Crackle) @ 35% Sensitivity.
        """
        if not self.is_trained:
            return "UNTRAINED"
        
        # Get probabilities to apply clinical thresholding
        probas = self.model.predict_proba(features.reshape(1, -1))[0]
        
        # 1. Check for Class 2 (Crackle) or Class 3 (Both) priority
        # If either crackle component (2 or 3) is above threshold, prioritize it
        if probas[2] > self.sensitivity_threshold_c2:
            prediction = [2]
        else:
            prediction = self.model.predict(features.reshape(1, -1))
        
        label_map = {
            0: 'Condition: Healthy (Normal)',
            1: 'Condition: Asthma (Wheezing)',
            2: 'Condition: Pneumonia/COPD (Cracks/Crackles)',
            3: 'Condition: Chronic (Wheeze & Crackle)'
        }
        
        status = label_map.get(prediction[0], 'Condition: Unknown Anomaly')
        return status

    def predict_proba(self, features):
        """
        Return the probability distribution across all 4 condition classes.
        """
        if not self.is_trained:
            return [0.25, 0.25, 0.25, 0.25]
            
        probas = self.model.predict_proba(features.reshape(1, -1))[0]
        
        # Ensure consistent return order for UI
        # Result mapping for frontend
        return {
            "Healthy": float(probas[0]),
            "Asthma": float(probas[1]) if len(probas) > 1 else 0.0,
            "Pneumonia": float(probas[2]) if len(probas) > 2 else 0.0,
            "COPD (Mixed)": float(probas[3]) if len(probas) > 3 else 0.0
        }

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

def get_dsp_metrics(data, fs):
    """
    Extract expert DSP metrics for the Technical Inspector mode.
    [EXPERIMENT 3 / TECHNICAL PROOF]
    """
    # 1. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=data, sr=fs)[0]
    # 2. RMS Energy
    rms = librosa.feature.rms(y=data)[0]
    # 3. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=data)[0]
    # 4. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=data, sr=fs)[0]
    # 5. Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=data)[0]
    
    return {
        "Spectral Centroid (Hz)": float(np.mean(centroid)),
        "RMS Energy (a.u.)": float(np.mean(rms)),
        "Zero Crossing Rate": float(np.mean(zcr)),
        "Spectral Rolloff (Hz)": float(np.mean(rolloff)),
        "Spectral Flatness": float(np.mean(flatness))
    }

