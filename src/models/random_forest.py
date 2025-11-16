"""
Random Forest model for weather forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.base_model import BaseWeatherModel
from src.utils.config import RANDOM_SEED, N_JOBS

logger = logging.getLogger(__name__)


class RandomForestModel(BaseWeatherModel):
    """
    Random Forest for weather forecasting.

    Ensemble of decision trees - robust, handles non-linearity well.
    """

    def __init__(self, task='regression', n_estimators=200, max_depth=20,
                 min_samples_split=5, min_samples_leaf=2, max_features='sqrt'):
        """
        Initialize Random Forest.

        Parameters
        ----------
        task : str
            'regression' or 'classification'
        n_estimators : int
            Number of trees
        max_depth : int or None
            Maximum tree depth
        min_samples_split : int
            Minimum samples to split node
        min_samples_leaf : int
            Minimum samples in leaf
        max_features : str or int
            Features to consider for splits
        """
        super().__init__(model_name='RandomForest', task=task)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model = self.build_model()

    def build_model(self, **kwargs):
        """Build Random Forest model."""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': RANDOM_SEED,
            'n_jobs': N_JOBS,
            'verbose': 0
        }

        if self.task == 'regression':
            return RandomForestRegressor(**params)
        else:
            return RandomForestClassifier(**params)

    def _fit_model(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train Random Forest."""
        # Train model (no scaling needed for tree-based models!)
        self.model.fit(X_train, y_train)

        logger.info(f"  Trees: {self.n_estimators}")
        logger.info(f"  Max depth: {self.max_depth}")
        logger.info(f"  Features: {X_train.shape[1]}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Test Random Forest
    from src.utils.data_split import prepare_regression_data

    logger.info("Testing Random Forest...")

    # Load data
    data = prepare_regression_data()

    # Train model
    model = RandomForestModel(task='regression', n_estimators=100)
    model.fit(data['X_train'], data['y_train'])

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

    logger.info("\n--OK-- Random Forest test complete!")