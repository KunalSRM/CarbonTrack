# src/train.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

PROCESSED_PATH = os.path.join("..", "data", "processed", "AirQuality_processed.csv")
MODEL_PATH = os.path.join("..", "models", "carbon_model.pkl")

def train_model():
    df = pd.read_csv(PROCESSED_PATH)

    features = ['CO', 'PT08_S1_CO', 'NMHC', 'C6H6', 'PT08_S2_NMHC', 'NOx', 'PT08_S3_NOx', 'NO2']
    X = df[features]

    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
