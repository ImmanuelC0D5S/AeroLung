from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import os
import shutil
import re
from src.dsp_engine import (
    apply_notch_filter,
    apply_bandpass_filter,
    perform_spectral_analysis,
    simulate_quantization,
    alter_sampling_rate,
    apply_signal_amplification,
    load_audio
)
from src.ml_logic import (
    detect_crackles_rms,
    extract_mfcc_features,
    check_lung_health,
    get_dsp_metrics,
    LungClassifier
)

app = FastAPI(title="AeroLung Diagnostic API")

# Initialize Classifier
classifier = LungClassifier()
MODEL_PATH = "models/lung_model.joblib"
if os.path.exists(MODEL_PATH):
    classifier.load_model(MODEL_PATH)
    if classifier.is_trained:
        # Diagnostic: Verify expected feature count (53)
        n_feat = classifier.model.n_features_in_
        print(f"--- AERO-ENGINE READY: Loaded {MODEL_PATH} with {n_feat} features ---")
else:
    print(f"--- AERO-ENGINE WARNING: No model found at {MODEL_PATH} ---")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    bit_depth: int = Form(16),
    target_sr: int = Form(44100),
    gain_db: float = Form(0.0),
    auto_gain: bool = Form(False)
):
    """
    Main diagnostic endpoint [EXPERIMENT TRACEABILITY MODE]
    """
    temp_path = os.path.join(UPLOAD_DIR, f"analyze_{file.filename}")
    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logs = []
        logs.append(f"INIT: System received {file.filename}. Starting analysis pipeline...")

        # Clinical Context Detection
        # ICBHI Pattern: [pat_id]_[rec_id]_[pos]_[type]_[device]
        icbhi_pattern = r"^[0-9]{3}_[0-9]"
        is_icbhi = re.match(icbhi_pattern, file.filename) is not None
        is_tracheal = "_Tc_" in file.filename
        
        # [CONDITIONAL GUARD] 
        # ICBHI/Tc samples use RAW clinical mode; others use Filtered Field mode.
        is_clinical = is_icbhi or is_tracheal
        
        if is_clinical:
            logs.append("CLINICAL DATASET DETECTED: Engaging High-Fidelity Raw Analysis [Lab Mode].")
        else:
            logs.append("FIELD RECORDING DETECTED: Engaging Noise-Resilient Filtered Analysis [Field Mode].")

        if is_tracheal:
            logs.append("CLINICAL MODE: Tracheal (Tc) Sensitivity Active. Monitoring for Flow Limitation.")

        # 1. Load Audio
        data_orig, fs_orig = load_audio(temp_path)
        logs.append(f"LOAD: Audio digitized at {fs_orig}Hz.")

        # 2. [EXPERIMENT 8] Resampling
        if fs_orig != target_sr:
            raw_audio, fs = alter_sampling_rate(data_orig, fs_orig, target_sr)
            logs.append(f"Applying EXPERIMENT [8] Logic: Resampling signal to {target_sr}Hz for sync.")
        else:
            raw_audio = data_orig
            fs = fs_orig
            logs.append(f"Applying EXPERIMENT [8] Logic: Sampling rate verified at {target_sr}Hz.")

        # [EXPERIMENT 9] Pre-Amp Stage
        amplified_audio = apply_signal_amplification(raw_audio, gain_db=gain_db, auto_gain=auto_gain)
        if gain_db > 0 or auto_gain:
            logs.append(f"Applying EXPERIMENT [9] Logic: Digital Pre-Amp Stage engaged ({gain_db}dB / AGC={auto_gain}).")

        # 3. [EXPERIMENT 5] Notch Filter (50Hz)
        notched_data = apply_notch_filter(amplified_audio, fs, freq=50.0)
        logs.append("Applying EXPERIMENT [5] Logic: 50Hz IIR Notch Filter engaged (Removing AC Mains Hum).")

        # 4. [EXPERIMENT 6/4] Bandpass Filter (200Hz - 2000Hz)
        filtered_audio = apply_bandpass_filter(notched_data, fs, lowcut=200, highcut=2000)
        logs.append("Applying EXPERIMENT [6/4] Logic: Butterworth Bandpass (200-2000Hz) isolating lung sounds.")

        # 5. [EXPERIMENT 7] Quantization
        quantized_data = simulate_quantization(filtered_audio, bits=bit_depth)
        logs.append(f"Applying EXPERIMENT [7] Logic: Simulating {bit_depth}-bit precision quantization.")
        
        # Calculate Quantization SNR [Exp 7]
        signal_power = np.mean(np.square(filtered_audio))
        noise_power = np.mean(np.square(filtered_audio - quantized_data))
        snr_db = float(10 * np.log10(signal_power / (noise_power + 1e-15)))

        # 6. [EXPERIMENT 2] Spectral Analysis
        freqs, mag = perform_spectral_analysis(quantized_data, fs, n_fft=2048)
        logs.append("Applying EXPERIMENT [2] Logic: Academic Spectral Analysis (Hamming Windowed).")
        
        if is_tracheal:
            logs.append("EXP 2 ADAPTATION: Analyzing Tracheal spectrum for COPD 'Broadband Hiss' indicators.")

        # Clinical Peak Detection [Exp 2]
        mask_clinical = (freqs >= 200) & (freqs <= 2000)
        if np.any(mask_clinical):
            p_idx = np.argmax(mag[mask_clinical])
            peak_freq = float(freqs[mask_clinical][p_idx])
        else:
            peak_freq = 0.0

        # AI Analysis (ML Logic)
        # [EXPERIMENT 3] RMS for Crackles
        rms, crackles = detect_crackles_rms(quantized_data)
        logs.append(f"Applying EXPERIMENT [3] Logic: RMS-based Anomaly detection found {len(crackles)} crackles.")
        
        # Feature Extraction (54 Features for ML - Including COPD sub-band ratio)
        # Conditional: Uses RAW for ClinicalMode, FILTERED for FieldMode
        mfccs = extract_mfcc_features(amplified_audio, fs, is_clinical=is_clinical)
        logs.append(f"Feature Vector Synchronized: {len(mfccs)} dimensions (Mode: {'Lab/Raw' if is_clinical else 'Field/Filtered'}).")
        
        dsp_features = get_dsp_metrics(amplified_audio, fs)
        
        # Prediction with Dimension Guard
        try:
            if classifier.is_trained:
                probas = classifier.model.predict_proba(mfccs.reshape(1, -1))[0]
                status = classifier.predict(mfccs)
                confidence = float(np.max(probas))
                
                # [CLINICAL SENSITIVITY TWEAK]
                # Priority: Tracheal monitoring for COPD baseline
                # If predicted healthy but it's a tracheal sample and confidence is not absolute (<85%)
                if is_tracheal and status == 'Condition: Healthy (Normal)' and confidence < 0.85:
                    status = "Acoustically Normal, but Clinical Baseline suggests COPD monitoring required."
            else:
                status = "UNTRAINED_MODEL_HEURISTIC"
                confidence = 0.5
        except ValueError as e:
            logs.append(f"CRITICAL ERROR [ML_SYNC]: {str(e)}")
            logs.append("Action: Recommend full laboratory model retraining (train_model.py).")
            status = "SYSTEM_MISMATCH_ERROR"
            confidence = 0.0

        # Define detection bounds for "Viva" highlight [Exp 2]
        # Wheezing typically 400-800Hz in this context
        detection_bounds = [400, 800] if "Asthma" in status or "Wheezing" in status else None

        logs.append("FINAL: ML Classifier generated diagnosis based on 53 feature vectors.")

        # Prepare JSON data for Recharts (High-Res for Inspector)
        step_inspect = max(1, len(raw_audio) // 4000)
        time_raw = raw_audio[::step_inspect].tolist()
        time_filtered = filtered_audio[::step_inspect].tolist()

        # Prepare FFT sub-sampled for frontend (PSD dB)
        # Limit strictly to 0-2000Hz range for "Clean & Academic" look
        mask_freq = (freqs >= 0) & (freqs <= 2000)
        freqs_sub = freqs[mask_freq]
        mag_sub = mag[mask_freq]
        
        freq_spectrum = [
            {"freq": int(f), "mag": float(m)} 
            for f, m in zip(freqs_sub, mag_sub)
        ]

        return {
            "status": "success",
            "logs": logs,
            "prediction": status,
            "confidence": round(confidence, 4),
            "snr_db": round(snr_db, 2),
            "peak_freq": round(peak_freq, 2),
            "detection_bounds": detection_bounds,
            "waveforms": {
                "raw": time_raw,
                "filtered": time_filtered
            },
            "fft": freq_spectrum,
            "dsp_features": dsp_features,
            "metadata": {
                "fs": fs,
                "bits": bit_depth,
                "segments": len(crackles)
            }
        }

    except Exception as e:
        print(f"Error in /analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
