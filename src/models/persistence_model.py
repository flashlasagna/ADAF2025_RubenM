import pandas as pd
import numpy as np
import logging
from .base_model import BaseWeatherModel

logger = logging.getLogger(__name__)


class PersistencePredictor:
    """
    A helper class that acts as the internal 'model' object.
    It mimics the interface of sklearn/xgboost models (has a .predict method).
    """

    def __init__(self, feature_name):
        self.feature_name = feature_name

    def predict(self, X):
        """Returns the column value as the prediction."""
        if self.feature_name not in X.columns:
            raise ValueError(f"Feature '{self.feature_name}' missing from input data")
        return X[self.feature_name].values

    def predict_proba(self, X):
        """
        For classification: returns probabilities.
        Assumes prediction is 0 or 1, so proba is [1.0, 0.0] or [0.0, 1.0].
        """
        preds = self.predict(X)
        # return [prob_class_0, prob_class_1]
        return np.column_stack((1 - preds, preds))


class PersistenceModel(BaseWeatherModel):
    """
    Wrapper for the Persistence baseline.
    """

    def __init__(self, task='regression', persistence_feature=None):
        super().__init__(model_name='Persistence', task=task)
        self.persistence_feature = persistence_feature

    def build_model(self, **kwargs):
        # Create our helper object to serve as the internal model
        return PersistencePredictor(self.persistence_feature)

    def _fit_model(self, X_train, y_train, X_val, y_val, **kwargs):
        # 1. Check if feature exists
        if self.persistence_feature not in X_train.columns:
            raise ValueError(f"Persistence feature '{self.persistence_feature}' not found in dataset!")

        # 2. Set the internal model to our Predictor class
        self.model = self.build_model()

        logger.info(f"Persistence model fitted using feature: {self.persistence_feature}")
        return self