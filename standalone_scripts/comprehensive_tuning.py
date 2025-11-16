"""
Comprehensive Hyperparameter Tuning - Full Search
No time constraints - thorough optimization for best results.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any
import joblib
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, f1_score
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.data_split import prepare_regression_data, prepare_classification_data
from src.models.linear_models import RidgeModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.utils.config import MODELS_DIR, TABLES_DIR

logger = logging.getLogger(__name__)

# ============================================================================
# COMPREHENSIVE PARAMETER GRIDS - Full Search
# ============================================================================

RIDGE_PARAM_GRID = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # 8 combinations
}

RANDOM_FOREST_PARAM_GRID = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 25, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
    # 4 * 6 * 3 * 3 * 2 = 432 combinations
}

XGBOOST_PARAM_GRID = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    # 4 * 4 * 4 * 3 * 4 * 4 = 3072 combinations
}

LIGHTGBM_PARAM_GRID = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [3, 5, 7, -1],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [10, 20, 50, 100],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    # 4 * 4 * 4 * 4 * 4 * 4 * 4 = 16384 combinations (too many!)
}

# More reasonable LightGBM grid
LIGHTGBM_PARAM_GRID_REASONABLE = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 500, 1000],
    'max_depth': [5, 7, -1],
    'num_leaves': [31, 63, 127],
    'min_child_samples': [20, 50, 100]
    # 3 * 3 * 3 * 3 * 3 = 243 combinations
}


# ============================================================================
# GRID SEARCH WITH EARLY STOPPING (for large grids)
# ============================================================================

def smart_grid_search(model_class, param_grid: Dict,
                      X_train, y_train, X_val, y_val,
                      task: str = 'regression',
                      model_name: str = 'Model',
                      max_combinations: int = None,
                      patience: int = 50) -> Dict:
    """
    Grid search with optional early stopping for very large grids.

    Parameters
    ----------
    model_class : class
        Model class to tune
    param_grid : dict
        Parameter grid
    X_train, y_train : training data
    X_val, y_val : validation data
    task : str
        'regression' or 'classification'
    model_name : str
        Name for logging
    max_combinations : int, optional
        Maximum combinations to test (None = test all)
    patience : int
        Stop if no improvement for this many combinations

    Returns
    -------
    dict
        Results including best params and all scores
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"COMPREHENSIVE GRID SEARCH: {model_name} ({task})")
    logger.info(f"{'=' * 80}")

    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    n_combinations = len(param_combinations)

    logger.info(f"Total parameter combinations: {n_combinations:,}")
    logger.info(f"Parameter grid:")
    for param, values in param_grid.items():
        logger.info(f"  {param}: {values}")

    # Limit combinations if specified
    if max_combinations and n_combinations > max_combinations:
        logger.info(f"Limiting to {max_combinations} random combinations")
        import random
        random.seed(42)
        param_combinations = random.sample(param_combinations, max_combinations)
        n_combinations = max_combinations

    # Track results
    results = []
    best_score = float('inf') if task == 'regression' else -float('inf')
    best_params = None
    no_improvement_count = 0

    # Progress tracking
    start_time = time.time()

    logger.info(f"\nStarting search...")
    logger.info(f"Early stopping patience: {patience} combinations\n")

    # Evaluate each combination
    for i, params in enumerate(param_combinations, 1):
        try:
            # Train model
            model = model_class(task=task, **params)
            model.fit(X_train, y_train, X_val, y_val)

            # Evaluate
            y_pred = model.predict(X_val)

            if task == 'regression':
                score = np.sqrt(mean_squared_error(y_val, y_pred))
                is_better = score < best_score
            else:
                score = f1_score(y_val, y_pred)
                is_better = score > best_score

            results.append({
                'params': params,
                'score': score
            })

            # Update best
            if is_better:
                improvement = best_score - score if task == 'regression' else score - best_score
                best_score = score
                best_params = params
                no_improvement_count = 0
                logger.info(f"  !! NEW BEST [{i}/{n_combinations}] Score: {score:.4f} (↑{improvement:.4f})")
                logger.info(f"     Params: {params}")
            else:
                no_improvement_count += 1

            # Progress every 25 combinations or milestones
            if i % 25 == 0 or i in [10, 50, 100, 200, 500]:
                elapsed = time.time() - start_time
                eta = (elapsed / i) * (n_combinations - i)
                metric_name = 'RMSE' if task == 'regression' else 'F1'
                logger.info(f"  [{i}/{n_combinations}] Best {metric_name}: {best_score:.4f} | "
                            f"No improvement: {no_improvement_count} | ETA: {eta / 60:.1f}min")

            # Early stopping check
            if patience and no_improvement_count >= patience:
                logger.info(f"\n⏹️  Early stopping after {i} combinations ({no_improvement_count} without improvement)")
                break

        except Exception as e:
            logger.error(f"Error with params {params}: {e}")
            results.append({
                'params': params,
                'score': float('inf') if task == 'regression' else 0.0
            })

    total_time = time.time() - start_time

    # Final results
    logger.info(f"\n{'=' * 80}")
    logger.info(f"SEARCH COMPLETE - {model_name}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Combinations tested: {len(results):,}")
    logger.info(f"Total time: {total_time / 60:.2f} minutes ({total_time / 3600:.2f} hours)")
    logger.info(f"Average time per combination: {total_time / len(results):.2f} seconds")
    logger.info(f"\nBest score: {best_score:.4f}")
    logger.info(f"Best parameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")

    return {
        'model_name': model_name,
        'task': task,
        'best_score': best_score,
        'best_params': best_params,
        'all_results': results,
        'n_combinations_tested': len(results),
        'n_combinations_total': n_combinations,
        'total_time': total_time
    }


# ============================================================================
# TUNING ORCHESTRATION
# ============================================================================

def tune_all_models_comprehensive(task: str = 'regression',
                                  use_early_stopping: bool = True) -> Dict:
    """
    Comprehensive tuning for all models.

    Parameters
    ----------
    task : str
        'regression' or 'classification'
    use_early_stopping : bool
        Whether to use early stopping for large grids

    Returns
    -------
    dict
        Tuning results for all models
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"COMPREHENSIVE HYPERPARAMETER TUNING - {task.upper()}")
    logger.info("=" * 80)

    # Load data once
    if task == 'regression':
        data = prepare_regression_data()
    else:
        data = prepare_classification_data()

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    all_results = {}

    # Ridge - small grid, no early stopping needed
    logger.info("\n" + "=" * 80)
    logger.info("MODEL 1/4: RIDGE REGRESSION")
    logger.info("=" * 80)
    all_results['ridge'] = smart_grid_search(
        RidgeModel, RIDGE_PARAM_GRID,
        X_train, y_train, X_val, y_val,
        task=task, model_name='Ridge',
        patience=None  # Small grid, test all
    )

    # Random Forest - medium grid
    logger.info("\n" + "=" * 80)
    logger.info("MODEL 2/4: RANDOM FOREST")
    logger.info("=" * 80)
    all_results['random_forest'] = smart_grid_search(
        RandomForestModel, RANDOM_FOREST_PARAM_GRID,
        X_train, y_train, X_val, y_val,
        task=task, model_name='RandomForest',
        max_combinations=200 if use_early_stopping else None,
        patience=50 if use_early_stopping else None
    )

    # XGBoost - large grid
    logger.info("\n" + "=" * 80)
    logger.info("MODEL 3/4: XGBOOST")
    logger.info("=" * 80)
    all_results['xgboost'] = smart_grid_search(
        XGBoostModel, XGBOOST_PARAM_GRID,
        X_train, y_train, X_val, y_val,
        task=task, model_name='XGBoost',
        max_combinations=300 if use_early_stopping else None,
        patience=75 if use_early_stopping else None
    )

    # LightGBM - large grid
    logger.info("\n" + "=" * 80)
    logger.info("MODEL 4/4: LIGHTGBM")
    logger.info("=" * 80)
    all_results['lightgbm'] = smart_grid_search(
        LightGBMModel, LIGHTGBM_PARAM_GRID_REASONABLE,
        X_train, y_train, X_val, y_val,
        task=task, model_name='LightGBM',
        patience=75 if use_early_stopping else None
    )

    return all_results


def save_comprehensive_results(tuning_results: Dict, task: str):
    """
    Save comprehensive tuning results.

    Parameters
    ----------
    tuning_results : dict
        Results from tune_all_models_comprehensive
    task : str
        'regression' or 'classification'
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"SAVING COMPREHENSIVE RESULTS - {task.upper()}")
    logger.info(f"{'=' * 80}")

    # Create tables directory
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Summary table
    summary_rows = []
    for model_name, result in tuning_results.items():
        summary_rows.append({
            'Model': model_name,
            'Best Score': result['best_score'],
            'Combinations Tested': result['n_combinations_tested'],
            'Total Combinations': result['n_combinations_total'],
            'Time (hours)': result['total_time'] / 3600,
            'Best Params': str(result['best_params'])
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLES_DIR / f'hyperparameter_tuning_{task}_comprehensive.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"--OK-- Saved summary: {summary_path}")

    # Detailed results for each model
    for model_name, result in tuning_results.items():
        all_results = pd.DataFrame(result['all_results'])

        # Expand params dict into columns
        params_df = pd.json_normalize(all_results['params'])
        all_results = pd.concat([params_df, all_results['score']], axis=1)
        all_results = all_results.sort_values('score',
                                              ascending=(task == 'regression'))

        detail_path = TABLES_DIR / f'tuning_{task}_{model_name}_comprehensive.csv'
        all_results.to_csv(detail_path, index=False)
        logger.info(f"--OK-- Saved {model_name} details: {detail_path}")

    # Save best parameters as JSON
    import json
    best_params_dict = {
        model_name: result['best_params']
        for model_name, result in tuning_results.items()
    }

    params_path = TABLES_DIR / f'best_params_{task}_comprehensive.json'
    with open(params_path, 'w') as f:
        json.dump(best_params_dict, f, indent=2)
    logger.info(f"--OK-- Saved best params: {params_path}")

    # Print summary
    logger.info(f"\n{'=' * 80}")
    logger.info("COMPREHENSIVE TUNING SUMMARY")
    logger.info(f"{'=' * 80}")
    print("\n" + summary_df.to_string(index=False))


def train_with_best_params_comprehensive(task: str = 'regression'):
    """
    Train models with comprehensive tuning results.

    Parameters
    ----------
    task : str
        'regression' or 'classification'
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TRAINING WITH COMPREHENSIVE BEST PARAMETERS - {task.upper()}")
    logger.info(f"{'=' * 80}")

    # Load best parameters
    import json
    params_path = TABLES_DIR / f'best_params_{task}_comprehensive.json'

    if not params_path.exists():
        logger.error(f"Best params file not found: {params_path}")
        logger.error("Run comprehensive tuning first!")
        return

    with open(params_path, 'r') as f:
        best_params = json.load(f)

    # Load data
    if task == 'regression':
        data = prepare_regression_data()
    else:
        data = prepare_classification_data()

    # Train each model
    for model_name, params in best_params.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training {model_name.upper()} with best params")
        logger.info(f"{'=' * 60}")
        logger.info(f"Parameters: {params}")

        # Instantiate model
        if model_name == 'ridge':
            model = RidgeModel(task=task, **params)
        elif model_name == 'random_forest':
            model = RandomForestModel(task=task, **params)
        elif model_name == 'xgboost':
            model = XGBoostModel(task=task, **params)
        elif model_name == 'lightgbm':
            model = LightGBMModel(task=task, **params)
        else:
            logger.warning(f"Unknown model: {model_name}")
            continue

        # Train
        model.fit(data['X_train'], data['y_train'],
                  data['X_val'], data['y_val'])

        # Evaluate on all sets
        for set_name in ['train', 'val', 'test']:
            X = data[f'X_{set_name}']
            y = data[f'y_{set_name}']
            y_pred = model.predict(X)

            if task == 'regression':
                score = np.sqrt(mean_squared_error(y, y_pred))
                logger.info(f"  {set_name.capitalize()} RMSE: {score:.4f}°C")
            else:
                score = f1_score(y, y_pred)
                logger.info(f"  {set_name.capitalize()} F1: {score:.4f}")

        # Save
        save_path = MODELS_DIR / f'{model_name}_{task}_comprehensive.pkl'
        model.save(save_path)
        logger.info(f"--OK-- Saved: {save_path}")


def run_comprehensive_pipeline(use_early_stopping: bool = True):
    """
    Run complete comprehensive tuning pipeline.

    Parameters
    ----------
    use_early_stopping : bool
        Whether to use early stopping for large grids
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE HYPERPARAMETER TUNING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Early stopping: {'Enabled' if use_early_stopping else 'Disabled'}")

    # Regression
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: REGRESSION MODELS")
    logger.info("=" * 80)
    regression_results = tune_all_models_comprehensive(
        task='regression',
        use_early_stopping=use_early_stopping
    )
    save_comprehensive_results(regression_results, task='regression')
    train_with_best_params_comprehensive(task='regression')

    # Classification
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: CLASSIFICATION MODELS")
    logger.info("=" * 80)
    classification_results = tune_all_models_comprehensive(
        task='classification',
        use_early_stopping=use_early_stopping
    )
    save_comprehensive_results(classification_results, task='classification')
    train_with_best_params_comprehensive(task='classification')

    logger.info("\n" + "=" * 80)
    logger.info("--OK-- COMPREHENSIVE TUNING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nResults saved to: {TABLES_DIR}")
    logger.info(f"Models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(PROJECT_ROOT / 'comprehensive_tuning.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE HYPERPARAMETER TUNING")
    logger.info("No time constraints - thorough search")
    logger.info("=" * 80)

    # Run with early stopping enabled (recommended)
    # Set to False for exhaustive search (will take much longer)
    run_comprehensive_pipeline(use_early_stopping=True)