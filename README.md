# Come Rain or Come Shine: Weather Prediction with Machine Learning

**Comparative Study of ML Models for Daily Weather Forecasting**  
*Advanced Data Analytics - HEC Lausanne*

**Author:** Ruben Mimouni  
**Advisor:** Dr. Maria Pia Lombardo  
**Date:** November 2025

---

## Project Overview

This project compares the effectiveness of different machine learning models for predicting daily weather in Suisse Romande using 25 years of high-quality meteorological data from Geneva-Cointrin and Pully weather stations.

**Research Question:** Which machine learning approach (linear models, ensemble methods, or deep learning) provides the best performance for daily weather forecasting with engineered features?

**Prediction Tasks:**
1. **Regression:** Next-day mean temperature (RMSE, MAE, R²)
2. **Classification:** Rain probability (F1-score, ROC-AUC)

---

## Project Structure

```
ADAF2025_RubenM/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file
│
├── data/                          # Data directory
│   ├── raw/                       # Original downloaded data
│   ├── processed/                 # Cleaned and processed data
│   └── features/                  # Feature-engineered datasets
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── data/                      # Data processing modules
│   │   ├── __init__.py
│   │   ├── load_data.py          # Data loading functions
│   │   ├── clean_data.py         # Data cleaning functions
│   │   └── preprocess.py         # Preprocessing pipeline
│   │
│   ├── features/                  # Feature engineering modules
│   │   ├── __init__.py
│   │   ├── temporal_features.py  # Time-based features
│   │   ├── lag_features.py       # Lagged variables
│   │   ├── rolling_features.py   # Rolling statistics
│   │   ├── derived_features.py   # Meteorological derivations
│   │   └── cross_station.py      # Cross-station features
│   │
│   ├── models/                    # Model implementations
│   │   ├── __init__.py
│   │   ├── base_model.py         # Base model class
│   │   ├── linear_models.py      # Ridge regression
│   │   ├── random_forest.py      # Random Forest
│   │   ├── xgboost_model.py      # XGBoost
│   │   ├── lightgbm_model.py     # LightGBM
│   │   └── lstm_model.py         # LSTM (optional)
│   │
│   ├── evaluation/                # Model evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py            # Evaluation metrics
│   │   ├── cross_validation.py   # Temporal CV
│   │   └── visualization.py      # Result visualizations
│   │
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── config.py             # Configuration settings
│       └── helpers.py            # Helper functions
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_final_analysis.ipynb
│
├── models/                        # Saved trained models
│   ├── ridge_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
|   ├── lgbm_model.pkl
│   └── tft_model.h5
│
├── results/                       # Results and figures
│   ├── figures/                  # All plots
│   ├── tables/                   # Result tables
│   └── model_comparison.csv      # Final comparison
│
├── docs/                          # Documentation
│   ├── report/                   # SIAM format report
│   │   ├── main.tex
│   │   ├── references.bib
│   │   └── figures/
│   └── presentation/             # Video presentation materials
│
└── tests/                         # Unit tests
    ├── __init__.py
    ├── test_data.py
    ├── test_features.py
    └── test_models.py
```

---

## Data Description

**Source:** MeteoSwiss Open Data Platform  
**Stations:** Geneva-Cointrin (GVE), Pully (PUY)  
**Period:** January 1, 2000 - December 31, 2024 (25 years)  
**Records:** 9,132 daily observations per station  
**Completeness:** 100% (after interpolation)

**Variables (12 per station = 24 total):**
- Temperature: mean, max, min (°C)
- Humidity: relative (%)
- Pressure: QFF (hPa)
- Precipitation: daily total (mm)
- Radiation: global (W/m²)
- Sunshine: duration (hours)
- Wind: speed (m/s), direction (°), gust (m/s)
- Evaporation: FAO reference (mm/day)

---

## Models Compared

| Model | Type | Expected Performance | Training Time |
|-------|------|---------------------|---------------|
| Ridge Regression | Linear | Baseline | ~1 min |
| Random Forest | Ensemble | Good | ~5 min |
| XGBoost | Gradient Boosting | Very Good | ~10 min |
| LightGBM | Gradient Boosting | **Best** | ~10 min |
| LSTM (optional) | Deep Learning | Good | ~30 min |

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/flashlasagna/ADAF2025_RubenM.git
cd weather-forecast-ml
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Data
Place MeteoSwiss CSV files in `data/raw/`:
- `ogd-smn_gve_d_historical.csv`
- `ogd-smn_puy_d_historical.csv`

### 5. Run Data Pipeline
```bash
# Process data
python src/data/preprocess.py

# Engineer features
python src/features/build_features.py

# Train models
python src/models/train_models.py

# Evaluate
python src/evaluation/evaluate_models.py
```

---

## Quick Start

### Option 1: Run Complete Pipeline
```bash
python main.py
```

### Option 2: Use Notebooks
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Option 3: Import as Module
```python
from src.data import load_data, clean_data
from src.features import build_features
from src.models import train_model

# Load and process
df = load_data.load_master_dataset()
df_clean = clean_data.handle_missing_values(df)
df_features = build_features.engineer_all_features(df_clean)

# Train model
model = train_model.train_lightgbm(df_features)
```

---

## Validation Strategy

**Critical:** We use **temporal cross-validation** to avoid data leakage.

```
Training:   2000-01-01 to 2019-12-31 (7,305 days = 80%)
Validation: 2020-01-01 to 2022-12-31 (1,096 days = 12%)
Test:       2023-01-01 to 2024-12-31 (731 days = 8%)
```

**Never shuffle!** Time flows forward only.

---

## Key Results (To Be Updated)

### Temperature Prediction (RMSE in °C)
| Model | Train | Validation | Test |
|-------|-------|------------|------|
| Ridge | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD |

### Rain Prediction (F1-Score)
| Model | Train | Validation | Test |
|-------|-------|------------|------|
| Ridge | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD |

---

## Deliverables

1. **SIAM Report** (8-10 pages)
   - Introduction & motivation
   - Data & methodology
   - Model descriptions
   - Results & comparison
   - Discussion & conclusions

2. **GitHub Repository**
   - Clean, documented code
   - Reproducible pipeline
   - README with instructions

3. **Video Presentation** (max duration)
   - Project overview
   - Key findings
   - Model comparison
   - Practical implications

---

## Progress Tracker

- [x] Data Foundation
  - [x] Data loading & cleaning
  - [x] Missing value treatment
  - [x] Initial exploration
- [x] Feature Engineering
  - [x] Temporal features
  - [x] Lag features
  - [x] Rolling statistics
  - [x] Derived variables
  - [x] Cross-station features
- [x] Classical Models
  - [x] Ridge Regression
  - [x] Random Forest
  - [x] XGBoost
  - [x] LightGBM
- [ ] Deep Learning (optional)
  - [ ] TFT implementation
  - [ ] Hyperparameter tuning
- [x] Model Comparison
  - [x] Temporal cross-validation
  - [x] Statistical testing
  - [x] Error analysis
- [ ] Report Writing
  - [ ] SIAM format document
  - [ ] Figures & tables
  - [ ] References
- [ ] Video & Submission
  - [ ] Record presentation
  - [ ] Final code review
  - [ ] Submit

---

## Contact

**Ruben Mimouni**\
ruben.mimouni@unil.ch\
Advanced Data Analytics\
HEC Lausanne

---

## License

This project is for academic purposes only.

---

## Acknowledgments

- MeteoSwiss for open weather data
- Dr. Maria Pia Lombardo for guidance
- Anthropic Claude for development assistance
