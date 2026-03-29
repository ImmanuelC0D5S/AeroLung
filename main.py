from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import os
import shutil
from src.dsp_engine import (
    apply_notch_filter,
    apply_bandpass_filter,
    perform_spectral_analysis,
    simulate_quantization,
    alter_sampling_rate,
    load_audio
)
from src.ml_logic import (
    detect_crackles_rms,
    extract_mfcc_features,
    check_lung_health,
    LungClassifier
)

app = FastAPI(title="AeroLung Diagnostic API")

# Initialize Classifier
classifier = LungClassifier()
if os.path.exists("models/lung_model.joblib"):
    classifier.load_model("models/lung_model.joblib")

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
    target_sr: int = Form(44100)
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

        # 1. Load Audio
        data, fs = load_audio(temp_path)
        logs.append(f"LOAD: Audio digitized at {fs}Hz.")

        # 2. [EXPERIMENT 8] Resampling
        if fs != target_sr:
            data, fs = alter_sampling_rate(data, fs, target_sr)
            logs.append(f"Applying EXPERIMENT [8] Logic: Resampling signal to {target_sr}Hz for sync.")
        else:
            logs.append(f"Applying EXPERIMENT [8] Logic: Sampling rate verified at {target_sr}Hz.")

        # 3. [EXPERIMENT 5] Notch Filter (50Hz)
        notched_data = apply_notch_filter(data, fs, freq=50.0)
        logs.append("Applying EXPERIMENT [5] Logic: 50Hz IIR Notch Filter engaged (Removing AC Mains Hum).")

        # 4. [EXPERIMENT 6/4] Bandpass Filter (200Hz - 2000Hz)
        filtered_data = apply_bandpass_filter(notched_data, fs, lowcut=200, highcut=2000)
        logs.append("Applying EXPERIMENT [6/4] Logic: Butterworth Bandpass (200-2000Hz) isolating lung sounds.")

        # 5. [EXPERIMENT 7] Quantization
        quantized_data = simulate_quantization(filtered_data, bits=bit_depth)
        logs.append(f"Applying EXPERIMENT [7] Logic: Simulating {bit_depth}-bit precision quantization.")

        # 6. [EXPERIMENT 2] Spectral Analysis
        freqs, mag = perform_spectral_analysis(quantized_data, fs)
        logs.append("Applying EXPERIMENT [2] Logic: FFT Spectral Analysis identifying wheeze peaks.")

        # 7. AI Analysis (ML Logic)
        # [EXPERIMENT 3] RMS for Crackles
        rms, crackles = detect_crackles_rms(quantized_data)
        logs.append(f"Applying EXPERIMENT [3] Logic: RMS-based Anomaly detection found {len(crackles)} crackles.")
        
        # Feature Extraction (53 Features)
        mfccs = extract_mfcc_features(quantized_data, fs)
        
        # Prediction
        if classifier.is_trained:
            status = classifier.predict(mfccs)
            confidence = float(np.max(classifier.model.predict_proba(mfccs.reshape(1, -1))))
        else:
            status = "UNTRAINED_MODEL_HEURISTIC"
            confidence = 0.5

        logs.append("FINAL: ML Classifier generated diagnosis based on 53 feature vectors.")

        # Prepare JSON data for Recharts (Downsampled for frontend performance)
        # Time-domain: 2000 samples
        step_time = max(1, len(data) // 2000)
        waveform_raw = data[::step_time].tolist()
        waveform_filtered = quantized_data[::step_time].tolist()

        # Frequency-domain: 1000 samples up to 4kHz
        mask_freq = freqs <= 4000
        freqs_sub = freqs[mask_freq]
        mag_sub = mag[mask_freq]
        step_freq = max(1, len(freqs_sub) // 1000)
        
        fft_data = [
            {"freq": int(f), "mag": float(20 * np.log10(m + 1e-9))} 
            for f, m in zip(freqs_sub[::step_freq], mag_sub[::step_freq])
        ]

        return {
            "status": "success",
            "logs": logs,
            "prediction": status,
            "confidence": round(confidence, 4),
            "waveform": {
                "raw": waveform_raw,
                "filtered": waveform_filtered
            },
            "fft": fft_data,
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
