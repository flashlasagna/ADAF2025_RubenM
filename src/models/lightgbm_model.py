"""
LightGBM model for weather forecasting.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.base_model import BaseWeatherModel
from src.utils.config import RANDOM_SEED, N_JOBS

logger = logging.getLogger(__name__)


class LightGBMModel(BaseWeatherModel):
    """
    LightGBM for weather forecasting.

    Fast gradient boosting - often best performer for tabular data.
    """

    def __init__(self, task='regression', n_estimators=500, learning_rate=0.05,
                 max_depth=7, num_leaves=63, min_child_samples=20,
                 subsample=0.8, colsample_bytree=0.8):
        """
        Initialize LightGBM.

        Parameters
        ----------
        task : str
            'regression' or 'classification'
        n_estimators : int
            Number of boosting rounds
        learning_rate : float
            Learning rate
        max_depth : int
            Maximum tree depth (-1 = no limit)
        num_leaves : int
            Maximum number of leaves
        min_child_samples : int
            Minimum data in leaf
        subsample : float
            Fraction of samples per tree
        colsample_bytree : float
            Fraction of features per tree
        """
        super().__init__(model_name='LightGBM', task=task)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.model = self.build_model()

    def build_model(self, **kwargs):
        """Build LightGBM model."""
        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': RANDOM_SEED,
            'n_jobs': N_JOBS,
            'verbosity': -1
        }

        if self.task == 'regression':
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
            return lgb.LGBMRegressor(**params)
        else:
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'
            return lgb.LGBMClassifier(**params)

    def _fit_model(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train LightGBM with early stopping."""
        callbacks = [lgb.log_evaluation(period=0)]  # Suppress verbose logging

        if X_val is not None and y_val is not None:
            # Use early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            logger.info(f"  Best iteration: {self.model.best_iteration_}")
        else:
            # No early stopping
            self.model.fit(X_train, y_train, callbacks=callbacks)

        logger.info(f"  Estimators: {self.n_estimators}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Num leaves: {self.num_leaves}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Test LightGBM
    from src.utils.data_split import prepare_regression_data

    logger.info("Testing LightGBM...")

    # Load data
    data = prepare_regression_data()

    # Train model
    model = LightGBMModel(task='regression', n_estimators=200)
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

    logger.info("\n✓ LightGBM test complete!")