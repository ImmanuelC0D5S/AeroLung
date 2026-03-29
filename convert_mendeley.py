import os
import shutil
import pandas as pd
import numpy as np
import librosa

# Paths
BASE_DIR = r"C:\Users\imman\Kokyu AI\AeroLung_Project"
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_SRC = os.path.join(DATA_DIR, "Audio files")
EXCEL_PATH = os.path.join(DATA_DIR, "Data annotation.xlsx")
OUT_DIR = os.path.join(DATA_DIR, "mendeley_converted")

def convert():
    print("--- Mendeley to ICBHI Digital Converter ---")
    
    # 1. Ensure Output Directory
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        print(f"Created output directory: {OUT_DIR}")

    # 2. Read Annotation Excel
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name='Sheet1')
        print(f"Loaded {len(df)} annotation rows from {os.path.basename(EXCEL_PATH)} [Sheet: Sheet1]")
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    # Verify columns - updating based on the new schema mapped
    required_cols = ['Sound type'] # Using the column we identified
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns {required_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        return

    converted_count = 0
    
    # 3. Process each row
    for index, row in df.iterrows():
        base_id = str(index + 100).strip() # Approximate mappings since patient ID isn't directly clean
        sound_label = str(row['Sound type']).strip().upper()
        
        # Determine flags (C=Crackle, W=Wheeze, N=Normal)
        crackle_flag = 1 if ('C' in sound_label or 'CREP' in sound_label) else 0
        wheeze_flag = 1 if 'W' in sound_label else 0
        
        # Filter for Diaphragm (D) and Extended (E) stethoscope modes
        for prefix in ['D', 'E']:
            wav_filename = f"{prefix}_{base_id}.wav"
            src_path = os.path.join(AUDIO_SRC, wav_filename)
            
            if os.path.exists(src_path):
                try:
                    # Get Audio Duration for segmentation
                    duration = librosa.get_duration(path=src_path)
                    
                    # Implement 3s Window Segmentation (User Recommendation)
                    segments = []
                    window_size = 3.0
                    
                    # Tile the duration into window_size segments
                    # e.g., 0.0-3.0, 3.0-6.0...
                    for start in np.arange(0, duration, window_size):
                        end = start + window_size
                        # Ensure we don't go past the actual file duration
                        if end > duration:
                            end = duration
                        
                        # Only keep segments that are at least 1.0s long to avoid tiny artifacts
                        if (end - start) >= 1.0:
                            segments.append(f"{start:.3f}\t{end:.3f}\t{crackle_flag}\t{wheeze_flag}")
                    
                    if not segments:
                        continue
                        
                    # Write ICBHI-style .txt file
                    txt_filename = wav_filename.replace(".wav", ".txt")
                    out_txt_path = os.path.join(OUT_DIR, txt_filename)
                    with open(out_txt_path, 'w') as f:
                        f.write("\n".join(segments))
                    
                    # Copy .wav to consolidated folder
                    out_wav_path = os.path.join(OUT_DIR, wav_filename)
                    shutil.copy(src_path, out_wav_path)
                    
                    converted_count += 1
                    if converted_count % 10 == 0:
                        print(f"Converted {converted_count} files...")
                        
                except Exception as e:
                    print(f"Error processing {wav_filename}: {e}")
            
    print(f"\nSUCCESS: Converted {converted_count} Diaphragm/Extended recordings into ICBHI format.")
    print(f"Location: {OUT_DIR}")

if __name__ == "__main__":
    convert()