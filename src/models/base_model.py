"""
Base model class for weather forecasting.
All models inherit from this class for consistent interface.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
import joblib
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class BaseWeatherModel(ABC):
    """
    Abstract base class for all weather forecasting models.

    Provides:
    - Consistent interface (fit, predict, evaluate)
    - Model saving/loading
    - Performance tracking
    - Logging
    """

    def __init__(self, model_name: str, task: str = 'regression'):
        """
        Initialize base model.

        Parameters
        ----------
        model_name : str
            Name of the model (e.g., 'Ridge', 'RandomForest')
        task : str
            Task type: 'regression' or 'classification'
        """
        self.model_name = model_name
        self.task = task
        self.model = None
        self.feature_names = None
        self.feature_importance_ = None
        self.training_time = None
        self.is_fitted = False

        logger.info(f"Initialized {model_name} for {task}")

    @abstractmethod
    def build_model(self, **kwargs) -> Any:
        """
        Build the model with specified hyperparameters.

        Must be implemented by subclasses.
        """
        pass

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs) -> 'BaseWeatherModel':
        """
        Train the model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame, optional
            Validation features (for early stopping)
        y_val : pd.Series, optional
            Validation target
        **kwargs
            Additional training parameters

        Returns
        -------
        self
        """
        logger.info(f"\nTraining {self.model_name}...")
        logger.info(f"  Training samples: {len(X_train):,}")
        if X_val is not None:
            logger.info(f"  Validation samples: {len(X_val):,}")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Track training time
        start_time = time.time()

        # Train model
        self._fit_model(X_train, y_train, X_val, y_val, **kwargs)

        self.training_time = time.time() - start_time
        self.is_fitted = True

        logger.info(f"✓ Training complete in {self.training_time:.2f}s")

        return self

    @abstractmethod
    def _fit_model(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Actual model training logic.

        Must be implemented by subclasses.
        """
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Features

        Returns
        -------
        np.ndarray
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Parameters
        ----------
        X : pd.DataFrame
            Features

        Returns
        -------
        np.ndarray
            Class probabilities
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification tasks")

        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (if available).

        Returns
        -------
        pd.DataFrame or None
            Feature importance sorted by value
        """
        if not self.is_fitted:
            logger.warning("Model not fitted yet")
            return None

        if not hasattr(self.model, 'feature_importances_'):
            logger.warning(f"{self.model_name} does not support feature importance")
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def save(self, filepath: Path):
        """
        Save model to disk.

        Parameters
        ----------
        filepath : Path
            Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        # Save model and metadata
        save_dict = {
            'model': self.model,
            'model_name': self.model_name,
            'task': self.task,
            'feature_names': self.feature_names,
            'training_time': self.training_time,
            'is_fitted': self.is_fitted
        }

        joblib.dump(save_dict, filepath)
        logger.info(f"✓ Saved {self.model_name} to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'BaseWeatherModel':
        """
        Load model from disk.

        Parameters
        ----------
        filepath : Path
            Path to saved model

        Returns
        -------
        BaseWeatherModel
            Loaded model instance
        """
        save_dict = joblib.load(filepath)

        # Create instance
        instance = cls(model_name=save_dict['model_name'],
                       task=save_dict['task'])

        # Restore attributes
        instance.model = save_dict['model']
        instance.feature_names = save_dict['feature_names']
        instance.training_time = save_dict['training_time']
        instance.is_fitted = save_dict['is_fitted']

        logger.info(f"✓ Loaded {instance.model_name} from {filepath}")

        return instance

    def __repr__(self):
        status = "fitted" if self.is_fitted else "unfitted"
        return f"{self.model_name}({self.task}, {status})"