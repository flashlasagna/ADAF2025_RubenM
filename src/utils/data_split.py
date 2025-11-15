"""
Data splitting utilities for temporal cross-validation.
CRITICAL: Never shuffle time series data!
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import (
    TRAIN_END_DATE,
    VAL_END_DATE,
    REGRESSION_TARGET,
    FEATURES_DATASET_FILE,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE
)

logger = logging.getLogger(__name__)


def load_feature_dataset() -> pd.DataFrame:
    """
    Load the complete feature-engineered dataset.

    Returns
    -------
    pd.DataFrame
        Feature dataset with all engineered features
    """
    logger.info(f"Loading feature dataset from {FEATURES_DATASET_FILE}")
    df = pd.read_csv(FEATURES_DATASET_FILE, parse_dates=['date'])
    logger.info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def temporal_train_val_test_split(
        df: pd.DataFrame,
        train_end: str = TRAIN_END_DATE,
        val_end: str = VAL_END_DATE,
        target_col: str = 'target_temp_next_day'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
pd.Series, pd.Series, pd.Series]:
    """
    Split data temporally into train/validation/test sets.

    CRITICAL: Never shuffle! Time flows forward only.

    Split logic:
    - Train: start → train_end
    - Val: train_end → val_end
    - Test: val_end → end

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset with date column
    train_end : str
        End date for training set (inclusive)
    val_end : str
        End date for validation set (inclusive)
    target_col : str
        Name of target column

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEMPORAL TRAIN/VAL/TEST SPLIT")
    logger.info("=" * 60)

    # Ensure sorted by date
    df = df.sort_values('date').reset_index(drop=True)

    # Create masks
    train_mask = df['date'] <= train_end
    val_mask = (df['date'] > train_end) & (df['date'] <= val_end)
    test_mask = df['date'] > val_end

    # Split data
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(f"\nData split:")
    logger.info(f"  Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df):,} rows)")
    logger.info(f"  Val:   {val_df['date'].min().date()} to {val_df['date'].max().date()} ({len(val_df):,} rows)")
    logger.info(f"  Test:  {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df):,} rows)")

    # Separate features and target
    feature_cols = [col for col in df.columns
                    if col not in ['date', 'target_temp_next_day', 'target_rain_next_day', 'rain_today']]

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df[target_col]
    y_val = val_df[target_col]
    y_test = test_df[target_col]

    logger.info(f"\nFeatures: {len(feature_cols)}")
    logger.info(f"Target: {target_col}")

    # Verify no NaN in targets
    assert y_train.isna().sum() == 0, "NaN found in train target!"
    assert y_val.isna().sum() == 0, "NaN found in val target!"
    assert y_test.isna().sum() == 0, "NaN found in test target!"

    logger.info("✓ No NaN in targets")

    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_regression_data(df: pd.DataFrame = None) -> Dict:
    """
    Prepare data for temperature regression task.

    Target: next-day mean temperature (Geneva)

    Parameters
    ----------
    df : pd.DataFrame, optional
        Feature dataset. If None, loads from file.

    Returns
    -------
    dict
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("\n" + "=" * 60)
    logger.info("PREPARING REGRESSION DATA")
    logger.info("=" * 60)

    if df is None:
        df = load_feature_dataset()

    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(
        df, target_col='target_temp_next_day'
    )

    # Statistics
    logger.info(f"\nTarget statistics:")
    logger.info(f"  Train - Mean: {y_train.mean():.2f}°C, Std: {y_train.std():.2f}°C")
    logger.info(f"  Val   - Mean: {y_val.mean():.2f}°C, Std: {y_val.std():.2f}°C")
    logger.info(f"  Test  - Mean: {y_test.mean():.2f}°C, Std: {y_test.std():.2f}°C")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': list(X_train.columns)
    }


def prepare_classification_data(df: pd.DataFrame = None) -> Dict:
    """
    Prepare data for rain classification task.

    Target: binary rain indicator (1 if precip > 0.1mm)

    Parameters
    ----------
    df : pd.DataFrame, optional
        Feature dataset. If None, loads from file.

    Returns
    -------
    dict
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test
    """
    logger.info("\n" + "=" * 60)
    logger.info("PREPARING CLASSIFICATION DATA")
    logger.info("=" * 60)

    if df is None:
        df = load_feature_dataset()

    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(
        df, target_col='target_rain_next_day'
    )

    # Statistics
    logger.info(f"\nClass distribution:")
    logger.info(
        f"  Train - Rain: {y_train.sum():,} ({y_train.mean() * 100:.1f}%), No rain: {(~y_train.astype(bool)).sum():,}")
    logger.info(
        f"  Val   - Rain: {y_val.sum():,} ({y_val.mean() * 100:.1f}%), No rain: {(~y_val.astype(bool)).sum():,}")
    logger.info(
        f"  Test  - Rain: {y_test.sum():,} ({y_test.mean() * 100:.1f}%), No rain: {(~y_test.astype(bool)).sum():,}")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': list(X_train.columns)
    }


def save_split_datasets(df: pd.DataFrame = None):
    """
    Save train/val/test splits to separate CSV files.

    Useful for reproducibility and external tools.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Feature dataset. If None, loads from file.
    """
    logger.info("\n" + "=" * 60)
    logger.info("SAVING SPLIT DATASETS")
    logger.info("=" * 60)

    if df is None:
        df = load_feature_dataset()

    # Create masks
    train_mask = df['date'] <= TRAIN_END_DATE
    val_mask = (df['date'] > TRAIN_END_DATE) & (df['date'] <= VAL_END_DATE)
    test_mask = df['date'] > VAL_END_DATE

    # Split and save
    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    logger.info(f"✓ Saved train set: {TRAIN_FILE} ({len(train_df):,} rows)")
    logger.info(f"✓ Saved val set: {VAL_FILE} ({len(val_df):,} rows)")
    logger.info(f"✓ Saved test set: {TEST_FILE} ({len(test_df):,} rows)")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Test data splitting
    logger.info("Testing data splitting utilities...")

    # Load data
    df = load_feature_dataset()

    # Prepare regression data
    reg_data = prepare_regression_data(df)

    # Prepare classification data
    clf_data = prepare_classification_data(df)

    # Save splits
    save_split_datasets(df)

    logger.info("\n✓ Data splitting utilities tested successfully!")