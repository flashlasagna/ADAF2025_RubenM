"""
Linear models for weather forecasting.
Includes Ridge Regression and Lasso.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, RidgeClassifier
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.base_model import BaseWeatherModel
from src.utils.config import RANDOM_SEED

logger = logging.getLogger(__name__)


class RidgeModel(BaseWeatherModel):
    """
    Ridge Regression for weather forecasting.

    Linear model with L2 regularization.
    Good baseline - fast, interpretable, robust.
    """

    def __init__(self, task='regression', alpha=1.0):
        """
        Initialize Ridge model.

        Parameters
        ----------
        task : str
            'regression' or 'classification'
        alpha : float
            Regularization strength (higher = more regularization)
        """
        super().__init__(model_name='Ridge', task=task)
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = self.build_model()

    def build_model(self, **kwargs):
        """Build Ridge model."""
        if self.task == 'regression':
            return Ridge(alpha=self.alpha, random_state=RANDOM_SEED)
        else:
            return RidgeClassifier(alpha=self.alpha, random_state=RANDOM_SEED)

    def _fit_model(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train Ridge model with feature scaling."""
        # Scale features (important for Ridge!)
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        logger.info(f"  Alpha: {self.alpha}")
        logger.info(f"  Features: {X_train.shape[1]}")

    def predict(self, X):
        """Make predictions with scaling."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Predict probabilities (classification only)."""
        if self.task != 'classification':
            raise ValueError("predict_proba only for classification")

        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        X_scaled = self.scaler.transform(X)
        # Ridge classifier doesn't have predict_proba, use decision_function
        decision = self.model.decision_function(X_scaled)

        # Convert to probabilities using sigmoid
        proba = 1 / (1 + np.exp(-decision))
        return np.vstack([1 - proba, proba]).T

    def get_coefficients(self):
        """
        Get model coefficients.

        Returns
        -------
        pd.DataFrame
            Coefficients sorted by absolute value
        """
        if not self.is_fitted:
            return None

        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        })
        coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
        coef_df = coef_df.sort_values('abs_coef', ascending=False)

        return coef_df.drop('abs_coef', axis=1)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Test Ridge model
    from src.utils.data_split import prepare_regression_data

    logger.info("Testing Ridge Regression...")

    # Load data
    data = prepare_regression_data()

    # Train model
    model = RidgeModel(task='regression', alpha=1.0)
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

    # Show top coefficients
    coefs = model.get_coefficients()
    logger.info(f"\nTop 10 features:")
    print(coefs.head(10))

    logger.info("\n--OK-- Ridge model test complete!")