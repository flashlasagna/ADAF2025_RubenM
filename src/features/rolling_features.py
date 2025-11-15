"""
Rolling feature engineering for weather forecasting.
Creates moving averages and standard deviations.
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

from src.utils.config import ROLLING_WINDOWS, ROLLING_VARS

logger = logging.getLogger(__name__)


def add_rolling_mean(df: pd.DataFrame,
                     variable: str,
                     window: int) -> pd.DataFrame:
    """
    Add rolling mean for a variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    variable : str
        Variable name
    window : int
        Rolling window size (days)

    Returns
    -------
    pd.DataFrame
        Dataframe with rolling mean added
    """
    new_col = f'{variable}_roll{window}_mean'
    df[new_col] = df[variable].rolling(window=window, min_periods=1).mean()

    return df


def add_rolling_std(df: pd.DataFrame,
                    variable: str,
                    window: int) -> pd.DataFrame:
    """
    Add rolling standard deviation for a variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    variable : str
        Variable name
    window : int
        Rolling window size (days)

    Returns
    -------
    pd.DataFrame
        Dataframe with rolling std added
    """
    new_col = f'{variable}_roll{window}_std'
    df[new_col] = df[variable].rolling(window=window, min_periods=1).std()

    return df


def add_rolling_features(df: pd.DataFrame,
                         variables: List[str],
                         windows: List[int] = ROLLING_WINDOWS,
                         add_std: bool = True) -> pd.DataFrame:
    """
    Add rolling statistics for specified variables.

    For each variable and window, creates:
    - var_roll{window}_mean: rolling average
    - var_roll{window}_std: rolling standard deviation (if add_std=True)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    variables : List[str]
        List of variable names
    windows : List[int]
        List of window sizes (default: [3, 7, 30])
    add_std : bool
        Whether to add standard deviation features

    Returns
    -------
    pd.DataFrame
        Dataframe with rolling features
    """
    logger.info(f"Adding rolling features for {len(variables)} variables...")
    logger.info(f"Windows: {windows}")
    logger.info(f"Include std: {add_std}")

    df_features = df.copy()

    features_added = 0
    for var in variables:
        if var not in df.columns:
            logger.warning(f"  ⚠️ Variable {var} not found, skipping")
            continue

        for window in windows:
            # Add rolling mean
            df_features = add_rolling_mean(df_features, var, window)
            features_added += 1

            # Add rolling std
            if add_std:
                df_features = add_rolling_std(df_features, var, window)
                features_added += 1

    logger.info(f"✓ Added {features_added} rolling features")

    return df_features


def get_key_variables_for_rolling(df: pd.DataFrame) -> List[str]:
    """
    Get key variables for rolling statistics.

    Based on config.ROLLING_VARS, finds corresponding station variables.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe

    Returns
    -------
    List[str]
        List of variable names for both stations
    """
    key_vars = []

    for base_var in ROLLING_VARS:
        # Add both station versions
        gve_var = f'{base_var}_gve'
        puy_var = f'{base_var}_puy'

        if gve_var in df.columns:
            key_vars.append(gve_var)
        if puy_var in df.columns:
            key_vars.append(puy_var)

    return key_vars


def add_all_rolling_features(df: pd.DataFrame,
                             windows: List[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Add rolling features for key meteorological variables.

    Focuses on temp, pressure, humidity as configured in config.py.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    windows : List[int]
        List of window sizes

    Returns
    -------
    pd.DataFrame
        Dataframe with all rolling features
    """
    logger.info("\n" + "=" * 60)
    logger.info("ROLLING FEATURE ENGINEERING")
    logger.info("=" * 60)

    # Get key variables
    key_vars = get_key_variables_for_rolling(df)
    logger.info(f"Found {len(key_vars)} key variables: {ROLLING_VARS}")

    # Add rolling features
    df_features = add_rolling_features(df, key_vars, windows, add_std=True)

    # Summary
    rolling_features = [col for col in df_features.columns if col not in df.columns]

    logger.info("\n" + "=" * 60)
    logger.info(f"ROLLING FEATURES COMPLETE: {len(rolling_features)} features added")
    logger.info("=" * 60)
    logger.info(
        f"\nExpected: {len(key_vars)} vars × {len(windows)} windows × 2 (mean+std) = {len(key_vars) * len(windows) * 2}")
    logger.info(f"Actual: {len(rolling_features)}")

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

    # Add rolling features
    df_features = add_all_rolling_features(df)

    logger.info(f"New shape: {df_features.shape}")
    logger.info(f"Features added: {df_features.shape[1] - df.shape[1]}")

    # Check for NaN values
    logger.info("\n" + "=" * 60)
    logger.info("MISSING VALUES INTRODUCED BY ROLLING")
    logger.info("=" * 60)

    max_window = max(ROLLING_WINDOWS)
    logger.info(f"First {max_window} rows may have NaN in rolling std")
    logger.info(f"(Rolling mean uses min_periods=1, so fewer NaNs)")

    # Show sample
    logger.info("\nSample rolling features (row 10):")
    roll_cols = [col for col in df_features.columns if 'roll' in col][:6]
    print(df_features[['date', 'temp_mean_gve'] + roll_cols].iloc[8:12])