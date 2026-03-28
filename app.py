import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
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
    check_lung_health
)

# Page configuration for Premium Look
st.set_page_config(page_title="AeroLung: AI-Assisted Digital Stethoscope", layout="wide")

st.title("🫁 AeroLung: AI-Assisted Digital Stethoscope")
st.markdown("Automated detection of Asthma/Pneumonia in noisy environments.")

# --- SIDEBAR: Configuration & Experiment Controls ---
st.sidebar.header("🔬 Experiment Controls")

# [EXPERIMENT 7]: Bit Depth Simulation
bit_depth = st.sidebar.slider("Simulated Bit Depth (Exp 7)", 8, 16, 16)

# [EXPERIMENT 8]: Sampling Rate Control
# Standard SR values: 44100, 22050, 16000, 8000
target_sr = st.sidebar.selectbox("Resampling Rate (Exp 8)", [44100, 22050, 16000, 8000], index=0)

# Dashboard Styling
st.markdown("""
<style>
    .status-card {
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .healthy { background-color: #28a745; }
    .anomaly { background-color: #dc3545; }
    .waiting { background-color: #6c757d; }
</style>
""", unsafe_allow_html=True)

# --- MAIN: File Uploader ---
uploaded_file = st.file_uploader("Upload Lung Sound (.wav)", type=["wav"])

if uploaded_file is not None:
    # Save the file temporarily
    temp_path = os.path.join("data", "uploaded_temp.wav")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 1. Load Data
    data, fs = load_audio(temp_path)
    
    # 2. Apply Experiment 8 (Resampling)
    if fs != target_sr:
        data, fs = alter_sampling_rate(data, fs, target_sr)

    # 3. Apply Experiment 7 (Quantization)
    data = simulate_quantization(data, bits=bit_depth)

    # 4. DSP Processing
    # [EXPERIMENT 5]: Notch Filter (50Hz)
    notched_data = apply_notch_filter(data, fs)

    # [EXPERIMENT 6/4]: Bandpass (200-2000Hz)
    filtered_data = apply_bandpass_filter(notched_data, fs)

    # [EXPERIMENT 2]: Spectral Analysis
    freqs, mag = perform_spectral_analysis(filtered_data, fs)

    # 5. ML / Anomaly Analysis (Heuristic based on User feedback for now)
    # [EXPERIMENT 3]: RMS for Crackles
    rms, crackles = detect_crackles_rms(filtered_data)
    mfccs = extract_mfcc_features(filtered_data, fs)
    
    status = check_lung_health(filtered_data, fs, mfccs, len(crackles))

    # --- UI: Status Display ---
    st.subheader("Diagnostic Status")
    if "HEALTHY" in status:
        st.markdown(f'<div class="status-card healthy">STATUS: {status}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-card anomaly">STATUS: {status}</div>', unsafe_allow_html=True)

    # --- UI: Visualization ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Signal Visualization (Raw vs Filtered)")
        fig, ax = plt.subplots(figsize=(10, 4))
        # Take a segment if it's too long
        segment = 5000 if len(data) > 5000 else len(data)
        ax.plot(data[:segment], label="Raw (Post SR/Bits)", alpha=0.5, color='gray')
        ax.plot(filtered_data[:segment], label="Filtered (Lung Sound)", color='blue')
        ax.set_title("Time Domain Analysis")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("FFT Spectrum (Exp 2)")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        # Log scale for better visualization of sounds
        ax2.plot(freqs, 20 * np.log10(mag + 1e-9), color='red')
        ax2.set_title("Frequency Domain Analysis")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude (dB)")
        ax2.set_xlim(0, 4000) # Zoom into lung sound range
        st.pyplot(fig2)

    # Additional Metrics
    st.subheader("Feature Metrics")
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Sampling Rate", f"{fs} Hz")
    m_col2.metric("Bit Depth", f"{bit_depth} Bits")
    m_col3.metric("Crackles Detected", len(crackles))

else:
    st.info("Please upload a .wav file to begin analysis.")
    st.markdown('<div class="status-card waiting">AWAITING INPUT...</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("AeroLung AI-Assisted Digital Stethoscope - Version 0.1 MVP")
