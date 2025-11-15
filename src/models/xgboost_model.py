"""
XGBoost model for weather forecasting.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.base_model import BaseWeatherModel
from src.utils.config import RANDOM_SEED, N_JOBS

logger = logging.getLogger(__name__)


class XGBoostModel(BaseWeatherModel):
    """
    XGBoost for weather forecasting.

    Gradient boosting - powerful, often wins competitions.
    """

    def __init__(self, task='regression', n_estimators=500, learning_rate=0.05,
                 max_depth=5, min_child_weight=3, subsample=0.8, colsample_bytree=0.8):
        """
        Initialize XGBoost.

        Parameters
        ----------
        task : str
            'regression' or 'classification'
        n_estimators : int
            Number of boosting rounds
        learning_rate : float
            Learning rate (smaller = more robust but slower)
        max_depth : int
            Maximum tree depth
        min_child_weight : int
            Minimum sum of instance weight in child
        subsample : float
            Fraction of samples per tree
        colsample_bytree : float
            Fraction of features per tree
        """
        super().__init__(model_name='XGBoost', task=task)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.model = self.build_model()

    def build_model(self, **kwargs):
        """Build XGBoost model."""
        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': RANDOM_SEED,
            'n_jobs': N_JOBS,
            'verbosity': 0
        }

        if self.task == 'regression':
            params['objective'] = 'reg:squarederror'
            return xgb.XGBRegressor(**params)
        else:
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
            return xgb.XGBClassifier(**params)

    def _fit_model(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train XGBoost with early stopping."""
        if X_val is not None and y_val is not None:
            # Use early stopping (newer XGBoost API)
            self.model.set_params(early_stopping_rounds=50)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            if hasattr(self.model, 'best_iteration'):
                logger.info(f"  Best iteration: {self.model.best_iteration}")
        else:
            # No early stopping
            self.model.fit(X_train, y_train)

        logger.info(f"  Estimators: {self.n_estimators}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Max depth: {self.max_depth}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Test XGBoost
    from src.utils.data_split import prepare_regression_data

    logger.info("Testing XGBoost...")

    # Load data
    data = prepare_regression_data()

    # Train model
    model = XGBoostModel(task='regression', n_estimators=200)
    model.fit(data['X_train'], data['y_train'],
              data['X_val'], data['y_val'])

    # Make predictions
    y_pred = model.predict(data['X_val'])

    # Calculate RMSE
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(data['y_val'], y_pred))
    r2 = r2_score(data['y_val'], y_pred)

    logger.info(f"\nValidation Results:")
    logger.info(f"  RMSE: {rmse:.3f}°C")
    logger.info(f"  R²: {r2:.4f}")

    # Show top features
    importance = model.get_feature_importance()
    logger.info(f"\nTop 10 features:")
    print(importance.head(10))

    logger.info("\n✓ XGBoost test complete!")