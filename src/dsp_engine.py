import numpy as np
from scipy.signal import iirnotch, filtfilt, butter, resample
import librosa

"""
AeroLung DSP Engine
Core signal processing functions for digital stethoscope analysis.
All functions are tagged with [EXPERIMENT #] as per lab manual requirements.
"""

def apply_notch_filter(data, fs, freq=50.0, Q=30.0):
    """
    # [EXPERIMENT 5]
    Implement a 50Hz IIR Notch Filter to remove AC Mains Hum.
    """
    b, a = iirnotch(freq, Q, fs)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def apply_bandpass_filter(data, fs, lowcut=200.0, highcut=2000.0, order=4):
    """
    # [EXPERIMENT 6/4]
    Implement a 4th-order Butterworth Bandpass filter (200Hz - 2000Hz) 
    to isolate lung sounds.
    """
    nyq = 0.5 * fs
    # Handle cases where highcut is above Nyquist (e.g. low sampling rate audio)
    if highcut >= nyq:
        highcut = nyq - 10 # Safety margin
    
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure low < high
    if low >= high:
        low = high * 0.5
        
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def perform_spectral_analysis(data, fs):
    """
    # [EXPERIMENT 2]
    Implement a function for Spectral Analysis using np.fft.rfft 
    to find "Wheeze" peaks.
    """
    n = len(data)
    rfft_result = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    magnitude = np.abs(rfft_result)
    
    # Simple peak finding logic for wheeze detection (high frequency components)
    # In a real scenario, this would be more complex.
    return freqs, magnitude

def simulate_quantization(data, bits=16):
    """
    # [EXPERIMENT 7]
    Implement a Quantization function that simulates 8-bit vs 16-bit audio 
    to test hardware efficiency.
    """
    if bits == 16:
        # Assuming input is already normalized between -1 and 1
        return np.round(data * 32767) / 32767
    elif bits == 8:
        return np.round(data * 127) / 127
    else:
        # General case
        levels = 2**(bits - 1) - 1
        return np.round(data * levels) / levels

def alter_sampling_rate(data, fs, target_fs):
    """
    # [EXPERIMENT 8]
    Implement a Sampling Rate Alteration function (Decimation/Interpolation) 
    to test low-power sync.
    """
    num_samples = int(len(data) * float(target_fs) / fs)
    resampled_data = resample(data, num_samples)
    return resampled_data, target_fs

def load_audio(file_path):
    """
    Utility function to load audio using librosa.
    """
    data, fs = librosa.load(file_path, sr=None)
    return data, fs
