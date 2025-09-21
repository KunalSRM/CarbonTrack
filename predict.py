# src/predict.py
import pandas as pd
import joblib
import os

PROCESSED_PATH = os.path.join("..", "data", "processed", "AirQuality_processed.csv")
MODEL_PATH = os.path.join("..", "models", "carbon_model.pkl")

def predict():
    df = pd.read_csv(PROCESSED_PATH)
    features = ['CO', 'PT08_S1_CO', 'NMHC', 'C6H6', 'PT08_S2_NMHC', 'NOx', 'PT08_S3_NOx', 'NO2']
    X = df[features]

    model = joblib.load(MODEL_PATH)
    df['anomaly'] = model.predict(X)  # -1 for anomaly, 1 for normal

    anomalies = df[df['anomaly'] == -1]
    print(f"Found {len(anomalies)} anomalous records:")
    print(anomalies.head())

if __name__ == "__main__":
    predict()
