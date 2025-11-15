"""
Lag feature engineering for weather forecasting.
Creates features from past values (t-1, t-2, t-7 days ago).
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import LAG_PERIODS

logger = logging.getLogger(__name__)


def add_lag_features(df: pd.DataFrame,
                     variables: List[str],
                     lags: List[int] = LAG_PERIODS) -> pd.DataFrame:
    """
    Add lagged features for specified variables.

    For each variable, creates features like:
    - var_lag1: yesterday's value
    - var_lag2: 2 days ago
    - var_lag7: 1 week ago

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with temporal data
    variables : List[str]
        List of variable names to create lags for
    lags : List[int]
        List of lag periods (default: [1, 2, 7])

    Returns
    -------
    pd.DataFrame
        Dataframe with lag features added
    """
    logger.info(f"Adding lag features for {len(variables)} variables...")
    logger.info(f"Lag periods: {lags}")

    df_features = df.copy()

    features_added = 0
    for var in variables:
        if var not in df.columns:
            logger.warning(f"  ⚠️ Variable {var} not found, skipping")
            continue

        for lag in lags:
            new_col = f'{var}_lag{lag}'
            df_features[new_col] = df_features[var].shift(lag)
            features_added += 1

    logger.info(f"✓ Added {features_added} lag features")

    return df_features


def get_meteorological_variables(df: pd.DataFrame) -> List[str]:
    """
    Get list of meteorological variables from dataframe.

    Identifies all variables with station suffixes (_gve, _puy).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with weather data

    Returns
    -------
    List[str]
        List of meteorological variable names
    """
    # Get all columns ending with _gve or _puy
    met_vars = [col for col in df.columns
                if col.endswith('_gve') or col.endswith('_puy')]

    # Exclude non-numeric columns if any
    numeric_vars = [col for col in met_vars
                    if df[col].dtype in ['float64', 'int64']]

    return numeric_vars


def add_all_lag_features(df: pd.DataFrame,
                         lags: List[int] = LAG_PERIODS) -> pd.DataFrame:
    """
    Add lag features for all meteorological variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with weather data
    lags : List[int]
        List of lag periods

    Returns
    -------
    pd.DataFrame
        Dataframe with all lag features
    """
    logger.info("\n" + "=" * 60)
    logger.info("LAG FEATURE ENGINEERING")
    logger.info("=" * 60)

    # Get meteorological variables
    met_vars = get_meteorological_variables(df)
    logger.info(f"Found {len(met_vars)} meteorological variables")

    # Add lag features
    df_features = add_lag_features(df, met_vars, lags)

    # Summary
    lag_features = [col for col in df_features.columns if col not in df.columns]

    logger.info("\n" + "=" * 60)
    logger.info(f"LAG FEATURES COMPLETE: {len(lag_features)} features added")
    logger.info("=" * 60)
    logger.info(f"\nExpected: {len(met_vars)} vars × {len(lags)} lags = {len(met_vars) * len(lags)}")
    logger.info(f"Actual: {len(lag_features)}")

    return df_features


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Test on master dataset
    from src.utils.config import MASTER_DATASET_FILE

    logger.info("Loading master dataset...")
    df = pd.read_csv(MASTER_DATASET_FILE, parse_dates=['date'])

    logger.info(f"Original shape: {df.shape}")

    # Add lag features
    df_features = add_all_lag_features(df)

    logger.info(f"New shape: {df_features.shape}")
    logger.info(f"Features added: {df_features.shape[1] - df.shape[1]}")

    # Check for NaN values introduced by lagging
    logger.info("\n" + "=" * 60)
    logger.info("MISSING VALUES INTRODUCED BY LAGGING")
    logger.info("=" * 60)

    max_lag = max(LAG_PERIODS)
    logger.info(f"First {max_lag} rows will have NaN in lag features")
    logger.info(f"Missing values in first {max_lag} rows: {df_features.iloc[:max_lag].isna().sum().sum()}")
    logger.info(f"Missing values after row {max_lag}: {df_features.iloc[max_lag:].isna().sum().sum()}")

    # Show sample
    logger.info("\nSample lag features (row 10):")
    lag_cols = [col for col in df_features.columns if 'lag' in col][:6]
    print(df_features[['date'] + lag_cols].iloc[10:12])