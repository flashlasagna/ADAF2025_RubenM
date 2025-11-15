"""
Configuration file for weather forecasting project.
Contains all paths, parameters, and settings.
"""

from pathlib import Path
import os

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"

# Results subdirectories
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR,
                  MODELS_DIR, FIGURES_DIR, TABLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================

# Raw data files
GENEVA_RAW_FILE = RAW_DATA_DIR / "ogd-smn_gve_d_historical.csv"
PULLY_RAW_FILE = RAW_DATA_DIR / "ogd-smn_puy_d_historical.csv"
METADATA_FILE = RAW_DATA_DIR / "ogd-smn_meta_parameters.csv"

# Processed data files
GENEVA_CLEAN_FILE = PROCESSED_DATA_DIR / "gve_2000_2024_cleaned.csv"
PULLY_CLEAN_FILE = PROCESSED_DATA_DIR / "puy_2000_2024_cleaned.csv"
MASTER_DATASET_FILE = PROCESSED_DATA_DIR / "weather_master_2000_2024.csv"
DATA_DICTIONARY_FILE = PROCESSED_DATA_DIR / "data_dictionary.csv"

# Feature-engineered files
FEATURES_DATASET_FILE = FEATURES_DATA_DIR / "weather_features_full.csv"
TRAIN_FILE = FEATURES_DATA_DIR / "train_data.csv"
VAL_FILE = FEATURES_DATA_DIR / "val_data.csv"
TEST_FILE = FEATURES_DATA_DIR / "test_data.csv"

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Time period
START_DATE = "2000-01-01"
END_DATE = "2024-12-31"

# Core meteorological variables (MeteoSwiss codes)
CORE_VARIABLES = {
    'tre200d0': 'temp_mean',
    'tre200dx': 'temp_max',
    'tre200dn': 'temp_min',
    'ure200d0': 'humidity',
    'pp0qffd0': 'pressure',
    'rre150d0': 'precipitation',
    'gre000d0': 'radiation',
    'sre000d0': 'sunshine',
    'fkl010d0': 'wind_speed',
    'dkl010d0': 'wind_direction',
    'erefaod0': 'evaporation',
    'fkl010d1': 'wind_gust'
}

# Station information
STATIONS = {
    'GVE': {
        'name': 'Geneva-Cointrin',
        'lat': 46.25,
        'lon': 6.13,
        'elevation': 412,  # meters
        'suffix': '_gve'
    },
    'PUY': {
        'name': 'Pully',
        'lat': 46.51,
        'lon': 6.66,
        'elevation': 456,  # meters
        'suffix': '_puy'
    }
}

# Distance between stations (km)
STATION_DISTANCE = 60.0

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# Lag features
LAG_PERIODS = [1, 2, 7]  # days

# Rolling window features
ROLLING_WINDOWS = [3, 7, 30]  # days

# Variables for rolling statistics
ROLLING_VARS = ['temp_mean', 'pressure', 'humidity']

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Train/val/test splits (temporal)
TRAIN_END_DATE = "2019-12-31"
VAL_END_DATE = "2022-12-31"
# Test starts after VAL_END_DATE until END_DATE

# Random seed for reproducibility
RANDOM_SEED = 42

# Target variables
REGRESSION_TARGET = 'temp_mean_gve'  # predict next day
CLASSIFICATION_TARGET = 'rain_binary'  # 1 if precip > threshold

# Rain threshold (mm)
RAIN_THRESHOLD = 0.1

# ============================================================================
# HYPERPARAMETER SEARCH SPACES
# ============================================================================

# Ridge Regression
RIDGE_PARAMS = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# Random Forest
RF_PARAMS = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# XGBoost
XGB_PARAMS = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# LightGBM
LGBM_PARAMS = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7, -1],
    'num_leaves': [31, 63, 127],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# LSTM
LSTM_PARAMS = {
    'sequence_length': [7, 14, 30],
    'hidden_units': [64, 128, 256],
    'num_layers': [1, 2, 3],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.01],
    'batch_size': [32, 64, 128]
}

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

# Metrics for regression
REGRESSION_METRICS = ['rmse', 'mae', 'r2', 'mape']

# Metrics for classification
CLASSIFICATION_METRICS = ['f1', 'precision', 'recall', 'roc_auc', 'accuracy']

# Cross-validation folds (temporal)
N_CV_FOLDS = 5

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Plot style
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
FIGURE_DPI = 300

# Color palette
COLOR_PALETTE = {
    'geneva': '#2E86AB',
    'pully': '#A23B72',
    'ridge': '#95E1D3',
    'rf': '#F38181',
    'xgboost': '#38A3A5',
    'lightgbm': '#22A39F',
    'lstm': '#C7A5C9'
}

# ============================================================================
# REPORT PARAMETERS
# ============================================================================

# SIAM format specifications
SIAM_PAGE_LIMIT = 10  # pages (excluding references)
SIAM_SECTIONS = [
    'Introduction',
    'Data and Methodology',
    'Models',
    'Results',
    'Discussion',
    'Conclusion'
]

# ============================================================================
# COMPUTATIONAL PARAMETERS
# ============================================================================

# Number of parallel jobs (-1 = use all cores)
N_JOBS = -1

# Verbosity level
VERBOSE = 1

# Memory usage
MAX_MEMORY_GB = 8

# ============================================================================
# LOGGING
# ============================================================================

import logging

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO
LOG_FILE = PROJECT_ROOT / 'project.log'

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_station_suffix(station_code):
    """Get suffix for station code."""
    return STATIONS[station_code]['suffix']

def get_variable_name(var_code, station_code):
    """Get full variable name with station suffix."""
    base_name = CORE_VARIABLES[var_code]
    suffix = get_station_suffix(station_code)
    return f"{base_name}{suffix}"

def print_config():
    """Print current configuration."""
    print("=" * 80)
    print("PROJECT CONFIGURATION")
    print("=" * 80)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Period: {START_DATE} to {END_DATE}")
    print(f"Stations: {', '.join(STATIONS.keys())}")
    print(f"Variables: {len(CORE_VARIABLES)}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"\nTrain/Val/Test Split:")
    print(f"  Train: {START_DATE} to {TRAIN_END_DATE}")
    print(f"  Val:   {TRAIN_END_DATE} to {VAL_END_DATE}")
    print(f"  Test:  {VAL_END_DATE} to {END_DATE}")
    print("=" * 80)

if __name__ == "__main__":
    print_config()