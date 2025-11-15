"""
Model training orchestrator.
Trains all models and saves results.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from typing import Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.data_split import prepare_regression_data, prepare_classification_data
from src.models.linear_models import RidgeModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.utils.config import MODELS_DIR

logger = logging.getLogger(__name__)


def train_all_regression_models() -> Dict:
    """
    Train all models for temperature regression task.

    Returns
    -------
    dict
        Dictionary of trained models
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING ALL REGRESSION MODELS")
    logger.info("=" * 80)

    # Load data
    data = prepare_regression_data()

    # Initialize models
    models = {
        'ridge': RidgeModel(task='regression', alpha=1.0),
        'random_forest': RandomForestModel(task='regression', n_estimators=200, max_depth=20),
        'xgboost': XGBoostModel(task='regression', n_estimators=500, learning_rate=0.05),
        'lightgbm': LightGBMModel(task='regression', n_estimators=500, learning_rate=0.05)
    }

    # Train each model
    trained_models = {}
    for name, model in models.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training {name.upper()}")
        logger.info(f"{'=' * 60}")

        # Train
        model.fit(data['X_train'], data['y_train'],
                  data['X_val'], data['y_val'])

        # Save
        model_path = MODELS_DIR / f'{name}_regression.pkl'
        model.save(model_path)

        trained_models[name] = model

    logger.info("\n" + "=" * 80)
    logger.info("ALL REGRESSION MODELS TRAINED!")
    logger.info("=" * 80)

    return trained_models


def train_all_classification_models() -> Dict:
    """
    Train all models for rain classification task.

    Returns
    -------
    dict
        Dictionary of trained models
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING ALL CLASSIFICATION MODELS")
    logger.info("=" * 80)

    # Load data
    data = prepare_classification_data()

    # Initialize models
    models = {
        'ridge': RidgeModel(task='classification', alpha=1.0),
        'random_forest': RandomForestModel(task='classification', n_estimators=200, max_depth=20),
        'xgboost': XGBoostModel(task='classification', n_estimators=500, learning_rate=0.05),
        'lightgbm': LightGBMModel(task='classification', n_estimators=500, learning_rate=0.05)
    }

    # Train each model
    trained_models = {}
    for name, model in models.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training {name.upper()}")
        logger.info(f"{'=' * 60}")

        # Train
        model.fit(data['X_train'], data['y_train'],
                  data['X_val'], data['y_val'])

        # Save
        model_path = MODELS_DIR / f'{name}_classification.pkl'
        model.save(model_path)

        trained_models[name] = model

    logger.info("\n" + "=" * 80)
    logger.info("ALL CLASSIFICATION MODELS TRAINED!")
    logger.info("=" * 80)

    return trained_models


def train_all_models():
    """
    Train all models for both tasks.

    Returns
    -------
    dict
        Nested dictionary: {'regression': {...}, 'classification': {...}}
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING ALL MODELS - REGRESSION & CLASSIFICATION")
    logger.info("=" * 80)

    # Train regression models
    regression_models = train_all_regression_models()

    # Train classification models
    classification_models = train_all_classification_models()

    logger.info("\n" + "=" * 80)
    logger.info("ALL MODELS TRAINED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\nModels saved to: {MODELS_DIR}")

    return {
        'regression': regression_models,
        'classification': classification_models
    }


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(PROJECT_ROOT / 'model_training.log'),
            logging.StreamHandler()
        ]
    )

    # Train all models
    all_models = train_all_models()

    print("\n✓ Model training complete!")
    print(f"  Regression models: {list(all_models['regression'].keys())}")
    print(f"  Classification models: {list(all_models['classification'].keys())}")