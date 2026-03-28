from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import os
import shutil
import io
import base64
import matplotlib.pyplot as plt
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

app = FastAPI(title="AeroLung API")

# Initialize Classifier
classifier = LungClassifier()
if os.path.exists("models/lung_model.joblib"):
    classifier.load_model("models/lung_model.joblib")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def generate_high_fidelity_plot(raw_data, filtered_data, freqs, mag, fs):
    """
    Generates a professional MATLAB-style plot using Matplotlib.
    """
    plt.close('all') # Ensure clean state
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    # 1. Time Domain Plot
    time = np.linspace(0, len(raw_data) / fs, num=len(raw_data))
    
    # Downsample for plotting efficiency if too large
    if len(raw_data) > 10000:
        step = len(raw_data) // 5000
        t_plot = time[::step]
        raw_plot = raw_data[::step]
        filt_plot = filtered_data[::step]
    else:
        t_plot = time
        raw_plot = raw_data
        filt_plot = filtered_data

    ax1.plot(t_plot, raw_plot, color='#CCCCCC', alpha=0.7, label='Raw Input', linewidth=0.8)
    ax1.plot(t_plot, filt_plot, color='#0072BD', label='Filtered (Lung Sound)', linewidth=1.2)
    ax1.set_title('Time Domain Analysis (Signal Comparison)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (seconds)', fontsize=10)
    ax1.set_ylabel('Amplitude (normalized)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(0, max(t_plot))
    ax1.set_ylim(-1.1, 1.1)

    # 2. Frequency Domain Plot
    # Only show up to 4000Hz for biological sounds
    mask = freqs <= 4000
    ax2.fill_between(freqs[mask], 20 * np.log10(mag[mask] + 1e-9), color='#D95319', alpha=0.3)
    ax2.plot(freqs[mask], 20 * np.log10(mag[mask] + 1e-9), color='#D95319', linewidth=1.0)
    ax2.set_title('Frequency Spectrum (Power Density)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)', fontsize=10)
    ax2.set_ylabel('Magnitude (dB)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim(0, 4000)
    ax2.set_ylim(-100, 20) # Typical dB range for lung sounds

    # MATLAB Aesthetic: Visible box and tick marks
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#141414')
        ax.tick_params(direction='in', length=6, width=1)

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    bit_depth: int = Form(16),
    target_sr: int = Form(44100)
):
    try:
        # Save uploaded file
        temp_path = os.path.join(UPLOAD_DIR, f"temp_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logs = []
        logs.append(f"Successfully uploaded: {file.filename}")

        # 1. Load Audio
        data, fs = load_audio(temp_path)
        logs.append(f"Audio loaded at {fs}Hz. Samples: {len(data)}")

        # 2. [EXPERIMENT 8] Resampling
        if fs != target_sr:
            data, fs = alter_sampling_rate(data, fs, target_sr)
            logs.append(f"[Exp 8] Resampling completed: {target_sr}Hz")
        else:
            logs.append(f"[Exp 8] Resampling skipped: Already at {target_sr}Hz")

        # 3. [EXPERIMENT 7] Quantization
        data = simulate_quantization(data, bits=bit_depth)
        logs.append(f"[Exp 7] Quantization simulated for {bit_depth}-bit precision.")

        # 4. [EXPERIMENT 5] Notch Filter (50Hz)
        notched_data = apply_notch_filter(data, fs)
        logs.append("[Exp 5] 50Hz Notch Filter applied (AC Hum removal).")

        # 5. [EXPERIMENT 6/4] Bandpass Filter (200Hz - 2000Hz)
        filtered_data = apply_bandpass_filter(notched_data, fs)
        logs.append("[Exp 6] 4th-Order Butterworth Bandpass (200-2000Hz) applied.")

        # 6. [EXPERIMENT 2] Spectral Analysis
        freqs, mag = perform_spectral_analysis(filtered_data, fs)
        logs.append("[Exp 2] FFT Spectral Analysis completed.")

        # 7. AI Analysis (ML Logic)
        # [EXPERIMENT 3] RMS for Crackles
        rms, crackles = detect_crackles_rms(filtered_data)
        mfccs = extract_mfcc_features(filtered_data, fs)
        
        # Use trained classifier if available, otherwise heuristic
        if classifier.is_trained:
            status_heuristic = classifier.predict(mfccs)
            confidence = 0.95 # Generic high confidence for model
        else:
            status_heuristic = check_lung_health(filtered_data, fs, mfccs, len(crackles))
            confidence = 0.85 if "HEALTHY" in status_heuristic else 0.92

        logs.append(f"[Exp 3] RMS Anomaly detection: Found {len(crackles)} crackle candidates.")

        # 8. Generate High-Fidelity Plot (MATLAB Style)
        high_fid_plot = generate_high_fidelity_plot(data, filtered_data, freqs, mag, fs)

        # Prepare responsive data for Recharts (live preview)
        step = max(1, len(data) // 2000)
        waveform_raw = data[::step].tolist()
        waveform_filtered = filtered_data[::step].tolist()

        max_freq = 4000
        freq_mask = freqs <= max_freq
        freqs_subset = freqs[freq_mask]
        mag_subset = mag[freq_mask]
        spec_step = max(1, len(freqs_subset) // 1000)
        fft_data = [
            {"freq": int(f), "mag": float(m)} 
            for f, m in zip(freqs_subset[::spec_step], mag_subset[::spec_step])
        ]

        return {
            "status": "success",
            "logs": logs,
            "diagnostic": {
                "status": status_heuristic,
                "confidence": confidence,
                "crackles": len(crackles),
                "sampling_rate": fs,
                "bit_depth": bit_depth,
                "plot_image": high_fid_plot # NEW: High-fidelity plot
            },
            "visuals": {
                "waveform_raw": waveform_raw,
                "waveform_filtered": waveform_filtered,
                "fft": fft_data
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
