"""
Model evaluation orchestrator.
Loads trained models and evaluates performance.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from typing import Dict, List
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.data_split import prepare_regression_data, prepare_classification_data
from src.evaluation.metrics import (
    calculate_regression_metrics,
    calculate_classification_metrics,
    get_confusion_matrix,
    print_classification_report
)
from src.evaluation.statistical_tests import (
    compare_all_models_regression,
    compare_all_models_classification
)
from src.utils.config import MODELS_DIR, RESULTS_DIR, TABLES_DIR

logger = logging.getLogger(__name__)


def load_model(model_path: Path):
    """
    Load a trained model from disk.

    Parameters
    ----------
    model_path : Path
        Path to saved model

    Returns
    -------
    model
        Loaded model object
    """
    logger.info(f"Loading model from {model_path.name}...")
    model_dict = joblib.load(model_path)
    return model_dict['model']


def evaluate_regression_models() -> pd.DataFrame:
    """
    Evaluate all regression models on train/val/test sets.

    Returns
    -------
    pd.DataFrame
        Results dataframe with all metrics
    """
    logger.info("\n" + "="*80)
    logger.info("EVALUATING REGRESSION MODELS")
    logger.info("="*80)

    # Load data
    data = prepare_regression_data()
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    # Model names
    model_names = ['ridge', 'random_forest', 'xgboost', 'lightgbm', 'persistence']

    # Results storage
    results = []

    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {model_name.upper()} Regression")
        logger.info(f"{'='*60}")

        # Load model
        model_path = MODELS_DIR / f'{model_name}_regression.pkl'

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue

        model = load_model(model_path)

        # Evaluate on each set
        for set_name, X, y in [('train', X_train, y_train),
                                ('val', X_val, y_val),
                                ('test', X_test, y_test)]:

            # Make predictions
            y_pred = model.predict(X)

            # Calculate metrics
            metrics = calculate_regression_metrics(y, y_pred,
                                                   f"{model_name} ({set_name})")

            # Store results
            result = {
                'model': model_name,
                'dataset': set_name,
                **metrics
            }
            results.append(result)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Save results
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results_path = TABLES_DIR / 'regression_results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n--OK-- Saved regression results to {results_path}")

    return results_df


def evaluate_classification_models() -> pd.DataFrame:
    """
    Evaluate all classification models on train/val/test sets.

    Returns
    -------
    pd.DataFrame
        Results dataframe with all metrics
    """
    logger.info("\n" + "="*80)
    logger.info("EVALUATING CLASSIFICATION MODELS")
    logger.info("="*80)

    # Load data
    data = prepare_classification_data()
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    # Model names
    model_names = ['ridge', 'random_forest', 'xgboost', 'lightgbm']

    # Results storage
    results = []

    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {model_name.upper()} Classification")
        logger.info(f"{'='*60}")

        # Load model
        model_path = MODELS_DIR / f'{model_name}_classification.pkl'

        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            continue

        model = load_model(model_path)

        # Evaluate on each set
        for set_name, X, y in [('train', X_train, y_train),
                                ('val', X_val, y_val),
                                ('test', X_test, y_test)]:

            # Make predictions
            y_pred = model.predict(X)

            # Get probabilities (if available)
            try:
                y_pred_proba = model.predict_proba(X)
            except:
                y_pred_proba = None

            # Calculate metrics
            metrics = calculate_classification_metrics(y, y_pred, y_pred_proba,
                                                      f"{model_name} ({set_name})")

            # Confusion matrix for test set
            if set_name == 'test':
                cm, cm_metrics = get_confusion_matrix(y, y_pred)
                logger.info(f"\nConfusion Matrix (Test Set):\n{cm}")
                print_classification_report(y, y_pred, model_name)

            # Store results
            result = {
                'model': model_name,
                'dataset': set_name,
                **metrics
            }
            results.append(result)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Save results
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results_path = TABLES_DIR / 'classification_results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n--OK-- Saved classification results to {results_path}")

    return results_df


def create_comparison_table(results_df: pd.DataFrame,
                           task: str = 'regression') -> pd.DataFrame:
    """
    Create formatted comparison table for report.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from evaluate_*_models
    task : str
        'regression' or 'classification'

    Returns
    -------
    pd.DataFrame
        Pivot table for easy comparison
    """
    if task == 'regression':
        # Focus on key metrics
        metrics = ['rmse', 'mae', 'r2']
    else:
        # Classification metrics
        metrics = ['f1', 'precision', 'recall', 'roc_auc']

    # Create pivot tables for each metric
    comparison = {}
    for metric in metrics:
        pivot = results_df.pivot(index='model', columns='dataset', values=metric)
        comparison[metric] = pivot

    return comparison


def print_summary_tables(reg_results: pd.DataFrame,
                        clf_results: pd.DataFrame):
    """
    Print summary tables to console and log.

    Parameters
    ----------
    reg_results : pd.DataFrame
        Regression results
    clf_results : pd.DataFrame
        Classification results
    """
    logger.info("\n" + "="*80)
    logger.info("SUMMARY: REGRESSION MODELS (Test Set)")
    logger.info("="*80)

    test_reg = reg_results[reg_results['dataset'] == 'test'].copy()
    test_reg = test_reg.sort_values('rmse')

    logger.info(f"\n{'Model':<15} {'RMSE (°C)':<12} {'MAE (°C)':<12} {'R²':<8}")
    logger.info("-" * 50)
    for _, row in test_reg.iterrows():
        logger.info(f"{row['model']:<15} {row['rmse']:<12.3f} {row['mae']:<12.3f} {row['r2']:<8.4f}")

    logger.info("\n" + "="*80)
    logger.info("SUMMARY: CLASSIFICATION MODELS (Test Set)")
    logger.info("="*80)

    test_clf = clf_results[clf_results['dataset'] == 'test'].copy()
    test_clf = test_clf.sort_values('f1', ascending=False)

    logger.info(f"\n{'Model':<15} {'F1':<8} {'Precision':<12} {'Recall':<8} {'ROC-AUC':<8}")
    logger.info("-" * 55)
    for _, row in test_clf.iterrows():
        logger.info(f"{row['model']:<15} {row['f1']:<8.4f} {row['precision']:<12.4f} "
                   f"{row['recall']:<8.4f} {row['roc_auc']:<8.4f}")


def evaluate_all_models():
    """
    Evaluate all models and generate comparison tables.

    Returns
    -------
    dict
        Dictionary with regression and classification results
    """
    logger.info("\n" + "="*80)
    logger.info("COMPLETE MODEL EVALUATION")
    logger.info("="*80)

    # Evaluate regression models
    reg_results = evaluate_regression_models()

    # Evaluate classification models
    clf_results = evaluate_classification_models()

    # Statistical testing
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL SIGNIFICANCE TESTING")
    logger.info("="*80)

    # Get test set predictions for statistical tests
    data_reg = prepare_regression_data()
    data_clf = prepare_classification_data()

    X_test_reg = data_reg['X_test']
    y_test_reg = data_reg['y_test']

    X_test_clf = data_clf['X_test']
    y_test_clf = data_clf['y_test']

    # Load models and get test predictions
    model_names = ['ridge', 'random_forest', 'xgboost', 'lightgbm']

    reg_predictions = {}
    clf_predictions = {}

    for model_name in model_names:
        # Regression
        try:
            model_path = MODELS_DIR / f'{model_name}_regression.pkl'
            if model_path.exists():
                model = load_model(model_path)
                reg_predictions[model_name] = model.predict(X_test_reg)
        except Exception as e:
            logger.warning(f"Could not load {model_name} regression: {e}")

        # Classification
        try:
            model_path = MODELS_DIR / f'{model_name}_classification.pkl'
            if model_path.exists():
                model = load_model(model_path)
                clf_predictions[model_name] = model.predict(X_test_clf)
        except Exception as e:
            logger.warning(f"Could not load {model_name} classification: {e}")

    # Statistical comparisons
    if len(reg_predictions) >= 2:
        p_matrix_reg = compare_all_models_regression(y_test_reg, reg_predictions)
        p_matrix_reg.to_csv(TABLES_DIR / 'regression_significance_tests.csv')

    if len(clf_predictions) >= 2:
        p_matrix_clf = compare_all_models_classification(y_test_clf, clf_predictions)
        p_matrix_clf.to_csv(TABLES_DIR / 'classification_significance_tests.csv')

    # Print summary
    print_summary_tables(reg_results, clf_results)

    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {TABLES_DIR}")

    return {
        'regression': reg_results,
        'classification': clf_results
    }


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(PROJECT_ROOT / 'evaluation.log'),
            logging.StreamHandler()
        ]
    )

    # Evaluate all models
    results = evaluate_all_models()

    print("\n--OK-- Model evaluation complete!")
    print(f"  Regression results: {len(results['regression'])} rows")
    print(f"  Classification results: {len(results['classification'])} rows")