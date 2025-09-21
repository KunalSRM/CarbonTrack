# src/eda_visualization.py
import pandas as pd
import plotly.express as px
import os

PROCESSED_PATH = os.path.join("..", "data", "processed", "AirQuality_processed.csv")

# Load processed data
df = pd.read_csv(PROCESSED_PATH, parse_dates=['datetime'])

# List of pollutants to visualize
pollutants = ['CO', 'PT08_S1_CO', 'NMHC', 'C6H6', 'PT08_S2_NMHC', 'NOx', 'PT08_S3_NOx', 'NO2']

# Plot line charts for each pollutant
for pollutant in pollutants:
    fig = px.line(df, x='datetime', y=pollutant, title=f'{pollutant} levels over time')
    fig.show()
