"""
Evaluation metrics for weather forecasting models.
Calculates regression and classification metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_regression_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 model_name: str = 'Model') -> Dict[str, float]:
    """
    Calculate regression metrics.

    Metrics:
    - RMSE: Root Mean Squared Error (°C)
    - MAE: Mean Absolute Error (°C)
    - R²: Coefficient of determination
    - MAPE: Mean Absolute Percentage Error (%)

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    model_name : str
        Name of model (for logging)

    Returns
    -------
    dict
        Dictionary of metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE - handle division by zero
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = np.nan

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

    logger.info(f"\n{model_name} Regression Metrics:")
    logger.info(f"  RMSE: {rmse:.3f}°C")
    logger.info(f"  MAE:  {mae:.3f}°C")
    logger.info(f"  R²:   {r2:.4f}")
    logger.info(f"  MAPE: {mape:.2f}%")

    return metrics


def calculate_classification_metrics(y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     y_pred_proba: np.ndarray = None,
                                     model_name: str = 'Model') -> Dict[str, float]:
    """
    Calculate classification metrics.

    Metrics:
    - F1-Score: Harmonic mean of precision and recall
    - Precision: Positive predictive value
    - Recall: Sensitivity / True positive rate
    - Accuracy: Overall correctness
    - ROC-AUC: Area under ROC curve (if probabilities provided)

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_pred_proba : np.ndarray, optional
        Predicted probabilities (for ROC-AUC)
    model_name : str
        Name of model (for logging)

    Returns
    -------
    dict
        Dictionary of metrics
    """
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }

    # ROC-AUC if probabilities provided
    if y_pred_proba is not None:
        # Handle different probability formats
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            # Binary classification with 2 columns [P(0), P(1)]
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            # Single column probability
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        metrics['roc_auc'] = roc_auc
    else:
        metrics['roc_auc'] = np.nan

    logger.info(f"\n{model_name} Classification Metrics:")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    if 'roc_auc' in metrics and not np.isnan(metrics['roc_auc']):
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    return metrics


def get_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Get confusion matrix and derived metrics.

    Returns
    -------
    tuple
        (confusion_matrix, metrics_dict)
    """
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0
    }

    return cm, metrics


def print_classification_report(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                model_name: str = 'Model'):
    """
    Print detailed classification report.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    model_name : str
        Name of model
    """
    logger.info(f"\n{model_name} Classification Report:")
    logger.info("\n" + classification_report(y_true, y_pred,
                                             target_names=['No Rain', 'Rain']))


def calculate_residuals(y_true: np.ndarray,
                        y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate residuals and residual statistics.

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    dict
        Dictionary with residuals and statistics
    """
    residuals = y_true - y_pred

    stats = {
        'residuals': residuals,
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'median': np.median(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'q25': np.percentile(residuals, 25),
        'q75': np.percentile(residuals, 75)
    }

    return stats


def mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean bias error (systematic over/under prediction).

    MBE = mean(predictions - actuals)

    Positive = tendency to overpredict
    Negative = tendency to underpredict

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    float
        Mean bias error
    """
    return np.mean(y_pred - y_true)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Test metrics with dummy data
    logger.info("Testing metrics calculation...")

    # Regression test
    y_true_reg = np.array([10.0, 15.0, 12.0, 18.0, 14.0])
    y_pred_reg = np.array([10.5, 14.5, 12.2, 17.8, 14.3])

    reg_metrics = calculate_regression_metrics(y_true_reg, y_pred_reg, "Test Model")

    # Classification test
    y_true_clf = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred_clf = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    y_proba_clf = np.array([[0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.9, 0.1],
                            [0.6, 0.4], [0.4, 0.6], [0.1, 0.9], [0.2, 0.8],
                            [0.7, 0.3], [0.8, 0.2]])

    clf_metrics = calculate_classification_metrics(y_true_clf, y_pred_clf,
                                                   y_proba_clf, "Test Model")

    cm, cm_metrics = get_confusion_matrix(y_true_clf, y_pred_clf)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"CM Metrics: {cm_metrics}")

    print_classification_report(y_true_clf, y_pred_clf, "Test Model")

    logger.info("\n✓ Metrics module test complete!")