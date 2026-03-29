import pandas as pd
import os

# Check Excel
df = pd.read_excel(r"C:\Users\imman\Kokyu AI\AeroLung_Project\data\Data Annotation.xlsx")
print("Columns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# Check wav files
files = os.listdir(r"C:\Users\imman\Kokyu AI\AeroLung_Project\data\mendeley_converted")
wavs = [f for f in files if f.endswith(".wav")]
print(f"\nWav files found: {len(wavs)}")
print("Sample:", wavs[:5])