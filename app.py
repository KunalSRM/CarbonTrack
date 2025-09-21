# # # # app.py
# # # import streamlit as st
# # # import pandas as pd
# # # import os
# # # import joblib
# # # from sklearn.ensemble import IsolationForest
# # # import plotly.express as px

# # # st.set_page_config(page_title="CarbonTrack", layout="wide")
# # # st.title("🌿 CarbonTrack – Air Quality Anomaly Detection")

# # # # Paths
# # # RAW_PATH = "data/raw/AirQuality.csv"
# # # PROCESSED_PATH = "data/processed/AirQuality_processed.csv"
# # # MODEL_PATH = "models/carbon_model.pkl"

# # # # 1️⃣ Upload CSV
# # # uploaded_file = st.file_uploader("Upload Air Quality CSV", type="csv")

# # # if uploaded_file:
# # #     df = pd.read_csv(uploaded_file, sep=';')
# # # else:
# # #     df = pd.read_csv(RAW_PATH, sep=';')
# # # st.success("✅ Dataset loaded!")

# # # # 2️⃣ Preprocess
# # # def preprocess(df):
# # #     df = df.copy()
# # #     df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
# # #     df.replace(',', '.', regex=True, inplace=True)
# # #     numeric_cols = df.columns[2:10]
# # #     df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
# # #     os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
# # #     df.to_csv(PROCESSED_PATH, index=False)
# # #     return df

# # # if st.button("Preprocess Data"):
# # #     df = preprocess(df)
# # #     st.success("✅ Data preprocessed!")

# # # # 3️⃣ Train Model
# # # def train_model(df):
# # #     features = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
# # #                 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)']
# # #     X = df[features]
# # #     model = IsolationForest(contamination=0.01, random_state=42)
# # #     model.fit(X)
# # #     os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
# # #     joblib.dump(model, MODEL_PATH)
# # #     return model

# # # if st.button("Train Model"):
# # #     model = train_model(df)
# # #     st.success("✅ Model trained!")

# # # # 4️⃣ Predict anomalies
# # # if os.path.exists(MODEL_PATH):
# # #     model = joblib.load(MODEL_PATH)
# # #     features = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
# # #                 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)']
# # #     df['anomaly'] = model.predict(df[features])
# # #     st.success(f"✅ Prediction done! Found {sum(df['anomaly']==-1)} anomalous records")
    
# # #     st.subheader("Anomalies")
# # #     st.dataframe(df[df['anomaly']==-1])

# # #     # 5️⃣ Plot
# # #     pollutant = st.selectbox("Select pollutant to visualize", features)
# # #     fig = px.line(df, x='datetime', y=pollutant, title=f"{pollutant} over time")
# # #     fig.add_scatter(x=df[df['anomaly']==-1]['datetime'],
# # #                     y=df[df['anomaly']==-1][pollutant],
# # #                     mode='markers', marker_color='red', name='Anomaly')
# # #     st.plotly_chart(fig, use_container_width=True)

# # # app.py

# # import streamlit as st
# # import pandas as pd
# # import joblib

# # # Load trained model
# # MODEL_PATH = "models/carbon_model.pkl"
# # model = joblib.load(MODEL_PATH)

# # st.title("Carbon Emission Anomaly Detection")
# # st.markdown("Enter the sensor readings manually to check for anomalies.")

# # # ---- Manual input fields ----
# # co = st.number_input("CO(GT)", min_value=0.0, step=0.1)
# # pt08_s1 = st.number_input("PT08.S1(CO)", min_value=0)
# # nmhc = st.number_input("NMHC(GT)", min_value=0)
# # c6h6 = st.number_input("C6H6(GT)", min_value=0.0)
# # pt08_s2 = st.number_input("PT08.S2(NMHC)", min_value=0)
# # nox = st.number_input("NOx(GT)", min_value=0)
# # pt08_s3 = st.number_input("PT08.S3(NOx)", min_value=0)
# # no2 = st.number_input("NO2(GT)", min_value=0)

# # # ---- Predict button ----
# # if st.button("Predict Anomaly"):
# #     # Convert inputs into DataFrame
# #     input_df = pd.DataFrame({
# #         "CO": [co],
# #         "PT08_S1_CO": [pt08_s1],
# #         "NMHC": [nmhc],
# #         "C6H6": [c6h6],
# #         "PT08_S2_NMHC": [pt08_s2],
# #         "NOx": [nox],
# #         "PT08_S3_NOx": [pt08_s3],
# #         "NO2": [no2]
# #     })

# #     # Predict
# #     prediction = model.predict(input_df)

# #     if prediction[0] == -1:
# #         st.error("⚠️ Anomaly detected!")
# #     else:
# #         st.success("✅ Reading is normal")


# # app.py

# import streamlit as st
# import pandas as pd
# import joblib
# import os

# # Paths
# MODEL_PATH = os.path.join("models", "carbon_model.pkl")

# # Load trained model
# model = joblib.load(MODEL_PATH)

# st.title("CarbonTrack - Air Quality Anomaly Detection")
# st.write("Enter sensor readings below to check for anomalies:")

# # Manual input fields
# co = st.number_input("CO (ppm)", min_value=0.0, step=0.1)
# pt08_s1_co = st.number_input("PT08_S1(CO)", min_value=0.0, step=1.0)
# nmhc = st.number_input("NMHC (ppm)", min_value=0.0, step=0.1)
# c6h6 = st.number_input("C6H6 (ppm)", min_value=0.0, step=0.1)
# pt08_s2_nmhc = st.number_input("PT08_S2(NMHC)", min_value=0.0, step=1.0)
# nox = st.number_input("NOx (ppb)", min_value=0.0, step=1.0)
# pt08_s3_nox = st.number_input("PT08_S3(NOx)", min_value=0.0, step=1.0)
# no2 = st.number_input("NO2 (ppb)", min_value=0.0, step=1.0)

# # Button to run prediction
# if st.button("Check Anomaly"):
#     # Create DataFrame in the correct format
#     input_df = pd.DataFrame({
#         'CO': [co],
#         'PT08_S1_CO': [pt08_s1_co],
#         'NMHC': [nmhc],
#         'C6H6': [c6h6],
#         'PT08_S2_NMHC': [pt08_s2_nmhc],
#         'NOx': [nox],
#         'PT08_S3_NOx': [pt08_s3_nox],
#         'NO2': [no2]
#     })
    
#     # Predict
#     anomaly = model.predict(input_df)[0]
    
#     if anomaly == -1:
#         st.error("⚠️ Anomalous reading detected!")
#     else:
#         st.success("✅ Reading is normal.")

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# Paths & Constants
# -------------------------------
MODEL_PATH = "models/carbon_model.pkl"

FEATURES = [
    'CO', 'PT08_S1_CO', 'NMHC', 'C6H6',
    'PT08_S2_NMHC', 'NOx', 'PT08_S3_NOx', 'NO2'
]

# Safe limits for reference (example values)
SAFE_LIMITS = {
    'CO': 9,           # ppm
    'C6H6': 5,         # µg/m³
    'NMHC': 0.2,       # ppm
    'NOx': 200,        # µg/m³
    'NO2': 200,        # µg/m³
    'PT08_S1_CO': None,  # sensor readings
    'PT08_S2_NMHC': None,
    'PT08_S3_NOx': None
}

# Harm messages for pollutants
HARM_MSGS = {
    'CO': "Causes headaches, dizziness, nausea, and can be fatal at high exposure.",
    'C6H6': "Carcinogenic! Affects blood and nervous system.",
    'NMHC': "Respiratory irritation, can trigger asthma.",
    'NOx': "Lung irritation, environmental acid rain.",
    'NO2': "Reduces lung immunity, triggers asthma.",
}

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌿 CarbonTrack – Air Quality Anomaly Detection")
st.write("Enter sensor readings manually to check for anomalies and get a detailed diagnosis.")

# Manual input fields
co = st.number_input("CO (ppm)", min_value=0.0, step=0.1)
pt08_s1_co = st.number_input("PT08_S1(CO)", min_value=0.0, step=1.0)
nmhc = st.number_input("NMHC (ppm)", min_value=0.0, step=0.1)
c6h6 = st.number_input("C6H6 (ppm)", min_value=0.0, step=0.1)
pt08_s2_nmhc = st.number_input("PT08_S2(NMHC)", min_value=0.0, step=1.0)
nox = st.number_input("NOx (ppb)", min_value=0.0, step=1.0)
pt08_s3_nox = st.number_input("PT08_S3(NOx)", min_value=0.0, step=1.0)
no2 = st.number_input("NO2 (ppb)", min_value=0.0, step=1.0)

# Button to run prediction
if st.button("Check Anomaly"):
    # Prepare input dataframe
    input_df = pd.DataFrame([{
        'CO': co,
        'PT08_S1_CO': pt08_s1_co,
        'NMHC': nmhc,
        'C6H6': c6h6,
        'PT08_S2_NMHC': pt08_s2_nmhc,
        'NOx': nox,
        'PT08_S3_NOx': pt08_s3_nox,
        'NO2': no2
    }])
    
    # Anomaly prediction
    anomaly = model.predict(input_df)[0]
    is_anomaly = anomaly == -1

    # -------------------------------
    # Diagnosis
    # -------------------------------
    diagnosis = []
    recommendations = []
    for feature in FEATURES:
        val = input_df[feature][0]
        safe = SAFE_LIMITS.get(feature)
        
        if safe is not None:
            if val > safe:
                diagnosis.append(f"{feature} level is {val} – HIGH! {HARM_MSGS.get(feature, '')}")
                recommendations.append(f"Reduce exposure to {feature}. Ventilate area if indoors.")
            else:
                diagnosis.append(f"{feature} level is {val} – within safe limits.")
        else:
            # Sensor readings abnormal check
            if val < 100 or val > 2000:
                diagnosis.append(f"{feature} reading is abnormal: {val} – check sensor calibration.")
                recommendations.append(f"Check sensor {feature} calibration.")

    # Display results
    if is_anomaly:
        st.error("⚠️ Anomalous reading detected!")
    else:
        st.success("✅ Reading is normal.")

    st.subheader("Detailed Diagnosis:")
    for d in diagnosis:
        st.write("-", d)

    # Visualization
    st.subheader("Readings vs Safe Limits")
    viz_df = pd.DataFrame({
        'Pollutant': FEATURES,
        'Value': [input_df[f][0] for f in FEATURES],
        'Safe Limit': [SAFE_LIMITS[f] if SAFE_LIMITS[f] else 0 for f in FEATURES]
    })
    st.bar_chart(viz_df.set_index('Pollutant'))

    # Recommendations
    st.subheader("Recommendations:")
    if recommendations:
        for r in recommendations:
            st.write("-", r)
    else:
        st.write("All readings within safe limits. No immediate action needed.")
