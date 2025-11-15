🌦️ Come Rain, Come Shine: Weather Prediction with Machine Learning

Comparative Study of ML Models for Daily Weather Forecasting
Advanced Data Analytics — HEC Lausanne

Author: Ruben Mimouni
Advisor: Dr. Maria Pia Lombardo
Date: November 2025

📘 Overview

This project conducts a comparative evaluation of machine learning models for daily weather prediction in Suisse Romande, using 25 years of high-quality meteorological records from the Geneva-Cointrin (GVE) and Pully (PUY) stations.

🎯 Research Question

Which ML approach—linear models, ensemble methods, or deep learning—achieves the best predictive performance with engineered meteorological features?

🔧 Prediction Tasks

Regression: Next-day mean temperature
Metrics: RMSE, MAE, R²

Classification: Probability of rainfall
Metrics: F1-Score, ROC-AUC

📁 Project Structure
weather_forecast_project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
│
├── notebooks/
├── models/
├── results/
├── docs/
└── tests/

🌡️ Data Description

Source: MeteoSwiss Open Data Platform
Stations: Geneva-Cointrin (GVE), Pully (PUY)
Period: 2000–2024
Observations: 9,132 days per station
Completeness: 100% after light interpolation

Variables Collected (12 per station)

Temperature: mean / max / min

Humidity (%)

Pressure (hPa)

Precipitation (mm)

Global radiation (W/m²)

Sunshine duration (h)

Wind: speed, gust, direction

Evaporation (FAO, mm/day)

🤖 Models Compared
Model	Type	Notes	Training Time
Ridge Regression	Linear	Baseline	~1 min
Random Forest	Ensemble	Robust, non-linear	~5 min
XGBoost	Gradient Boosting	High accuracy	~10 min
LightGBM	Gradient Boosting	Best expected	~10 min
LSTM (optional)	Deep Learning	Temporal modeling	~30 min
⚙️ Setup Instructions
1. Clone the Repository
git clone https://github.com/yourusername/weather-forecast-ml.git
cd weather-forecast-ml

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Provide the Data

Place the following files in data/raw/:

ogd-smn_gve_d_historical.csv

ogd-smn_puy_d_historical.csv

5. Run the Data & Modeling Pipeline
python src/data/preprocess.py
python src/features/build_features.py
python src/models/train_models.py
python src/evaluation/evaluate_models.py

🚀 Quick Start
Option A — Run Full Pipeline
python main.py

Option B — Explore via Jupyter
jupyter notebook notebooks/01_data_exploration.ipynb

Option C — Import as Python Module
from src.data import load_data, clean_data
from src.features import build_features
from src.models import train_model

df = load_data.load_master_dataset()
df_clean = clean_data.handle_missing_values(df)
df_features = build_features.engineer_all_features(df_clean)

model = train_model.train_lightgbm(df_features)

🔍 Validation Strategy

A strict temporal cross-validation design avoids data leakage.

Training:   2000–2019  (80%)
Validation: 2020–2022  (12%)
Test:       2023–2024  (8%)


❗ No shuffling — time moves forward only.

📊 Key Results (To Be Updated)
Temperature Prediction (RMSE °C)
Model	Train	Val	Test
Ridge	—	—	—
Random Forest	—	—	—
XGBoost	—	—	—
LightGBM	—	—	—
Rain Prediction (F1-Score)
Model	Train	Val	Test
Ridge	—	—	—
Random Forest	—	—	—
XGBoost	—	—	—
LightGBM	—	—	—
📦 Deliverables

SIAM-format report (8–10 pages)

Reproducible GitHub repository

15-min video presentation

📅 Progress Tracker

 Week 1–2 — Data acquisition & cleaning

 Week 3–4 — Feature engineering

 Week 5–6 — Classical ML models

 Week 7–8 — Deep learning (optional)

 Week 9–10 — Evaluation & comparison

 Week 11 — Report writing

 Week 12 — Video & final submission

📬 Contact

Ruben Mimouni
Advanced Data Analytics — HEC Lausanne

📄 License

This project is for academic and research purposes only.

🙏 Acknowledgments

MeteoSwiss for open data

Dr. Maria Pia Lombardo for guidance

Anthropic Claude for development assistance