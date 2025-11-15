"""
Temporal feature engineering for weather forecasting.
Creates cyclical encodings and time-based features.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical encodings for temporal features.

    Uses sine/cosine transformation to capture cyclical nature of time.
    This prevents discontinuity (e.g., Dec 31 → Jan 1).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with temporal columns

    Returns
    -------
    pd.DataFrame
        Dataframe with cyclical features added
    """
    logger.info("Adding cyclical temporal features...")

    df_features = df.copy()

    # Day of year (1-365/366)
    df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365.25)
    df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365.25)

    # Month (1-12)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)

    # Day of week (0-6)
    df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)

    logger.info("✓ Added 6 cyclical features:")
    logger.info("  - day_of_year (sin, cos)")
    logger.info("  - month (sin, cos)")
    logger.info("  - day_of_week (sin, cos)")

    return df_features


def add_season_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary season indicators.

    Seasons defined as:
    - Winter: Dec, Jan, Feb (12, 1, 2)
    - Spring: Mar, Apr, May (3, 4, 5)
    - Summer: Jun, Jul, Aug (6, 7, 8)
    - Fall: Sep, Oct, Nov (9, 10, 11)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with month column

    Returns
    -------
    pd.DataFrame
        Dataframe with season indicators
    """
    logger.info("Adding season indicators...")

    df_features = df.copy()

    # Create season indicators
    df_features['is_winter'] = df_features['month'].isin([12, 1, 2]).astype(int)
    df_features['is_spring'] = df_features['month'].isin([3, 4, 5]).astype(int)
    df_features['is_summer'] = df_features['month'].isin([6, 7, 8]).astype(int)
    df_features['is_fall'] = df_features['month'].isin([9, 10, 11]).astype(int)

    logger.info("✓ Added 4 season indicators")

    return df_features


def add_weekend_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weekend indicator.

    Note: This may not be relevant for weather prediction,
    but included for completeness. Can be removed if not useful.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with day_of_week column

    Returns
    -------
    pd.DataFrame
        Dataframe with weekend indicator
    """
    logger.info("Adding weekend indicator...")

    df_features = df.copy()
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)

    logger.info("✓ Added 1 weekend indicator")

    return df_features


def add_all_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all temporal features to dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with basic temporal columns (year, month, day_of_year, etc.)

    Returns
    -------
    pd.DataFrame
        Dataframe with all temporal features
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEMPORAL FEATURE ENGINEERING")
    logger.info("=" * 60)

    # Verify required columns exist
    required_cols = ['day_of_year', 'month', 'day_of_week']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Add features
    df_features = add_cyclical_features(df)
    df_features = add_season_indicators(df_features)
    df_features = add_weekend_indicator(df_features)

    # Summary
    temporal_features = [col for col in df_features.columns if col not in df.columns]

    logger.info("\n" + "=" * 60)
    logger.info(f"TEMPORAL FEATURES COMPLETE: {len(temporal_features)} features added")
    logger.info("=" * 60)
    logger.info(f"\nNew features: {temporal_features}")

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

    # Add temporal features
    df_features = add_all_temporal_features(df)

    logger.info(f"New shape: {df_features.shape}")
    logger.info(f"Features added: {df_features.shape[1] - df.shape[1]}")

    # Show sample
    logger.info("\nSample of new features:")
    new_cols = [col for col in df_features.columns if col not in df.columns]
    print(df_features[['date'] + new_cols].head())