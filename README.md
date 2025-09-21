🌿 CarbonTrack – Air Quality Anomaly Detection

CarbonTrack is a Python-based Streamlit web application that allows users to monitor air quality by detecting anomalous readings from air pollution sensors. The app provides real-time anomaly detection and helps understand potential environmental and health impacts.

🚀 Features

Manual Sensor Input: Enter values for key air quality parameters like CO, NMHC, NOx, etc.

Anomaly Detection: Uses an Isolation Forest machine learning model to identify unusual sensor readings.

Interactive Visualization: Visualize sensor readings and anomalies over time using Plotly charts.

Preprocessing Pipeline: Handles raw CSV data, converts formats, and prepares it for model training.

📂 Project Structure
CarbonTrack/
├── data/
│   ├── raw/                # Original AirQuality.csv
│   └── processed/          # Preprocessed data
├── models/
│   └── carbon_model.pkl    # Trained Isolation Forest model
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── eda_visualization.py
├── app.py                  # Streamlit web app
├── requirements.txt
└── README.md

🛠 Installation

Clone the repository

git clone https://github.com/<your-username>/CarbonTrack.git
cd CarbonTrack


Create and activate a virtual environment

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate


Install dependencies

pip install -r requirements.txt

🧩 Usage
1️⃣ Preprocess Data
python src/data_preprocessing.py

2️⃣ Train Model
python src/train.py

3️⃣ Predict Anomalies
python src/predict.py

4️⃣ Run Streamlit Web App
streamlit run app.py


Open the URL provided in your terminal to use the app.

⚙️ Manual Input Fields in App

CO (ppm)

PT08_S1(CO)

NMHC (ppm)

C6H6 (ppm)

PT08_S2(NMHC)

NOx (ppb)

PT08_S3(NOx)

NO2 (ppb)

The app will detect anomalies based on these inputs and provide feedback.

📊 Data Source

The raw data is based on the UCI Air Quality Dataset, which contains hourly averaged responses from gas sensors in an indoor environment.

📈 Machine Learning Model

Algorithm: Isolation Forest

Purpose: Identify anomalous sensor readings

Features Used: CO, PT08.S1(CO), NMHC, C6H6, PT08.S2(NMHC), NOx, PT08.S3(NOx), NO2

⚠️ Notes

Ensure the models/carbon_model.pkl exists before running the Streamlit app.

Large CSV uploads may slow down preprocessing.

The app currently supports manual input; future versions may include CSV upload and batch predictions.

📄 License

This project is licensed under the MIT License – see the LICENSE
 file for details.
