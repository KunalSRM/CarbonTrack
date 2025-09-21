# src/data_preprocessing.py
import pandas as pd
import os

RAW_DATA_PATH = os.path.join("..", "data", "raw", "AirQuality.csv")
PROCESSED_DATA_PATH = os.path.join("..", "data", "processed", "AirQuality_processed.csv")

def preprocess():
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    # Read CSV with correct separator
    df = pd.read_csv(RAW_DATA_PATH, sep=';')

    # Drop empty columns
    df = df.loc[:, df.columns.notna()]

    # Replace comma decimal with dot and convert to numeric
    for col in df.columns[2:]:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    # Create datetime column
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')

    # Rename columns for Python-friendly names
    df.rename(columns={
        'CO(GT)': 'CO',
        'PT08.S1(CO)': 'PT08_S1_CO',
        'NMHC(GT)': 'NMHC',
        'C6H6(GT)': 'C6H6',
        'PT08.S2(NMHC)': 'PT08_S2_NMHC',
        'NOx(GT)': 'NOx',
        'PT08.S3(NOx)': 'PT08_S3_NOx',
        'NO2(GT)': 'NO2'
    }, inplace=True)

    # Save processed CSV
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Preprocessing complete! Saved at: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess()
