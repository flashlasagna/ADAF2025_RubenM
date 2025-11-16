"""
Complete feature engineering pipeline.
Orchestrates all feature creation and target variable generation.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.features.temporal_features import add_all_temporal_features
from src.features.lag_features import add_all_lag_features
from src.features.rolling_features import add_all_rolling_features
from src.features.derived_features import add_all_derived_features
from src.features.cross_station import add_all_cross_station_features

from src.utils.config import (
    MASTER_DATASET_FILE,
    FEATURES_DATASET_FILE,
    REGRESSION_TARGET,
    RAIN_THRESHOLD
)

logger = logging.getLogger(__name__)


def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables for regression and classification.

    Regression target: next-day mean temperature
    Classification target: binary rain indicator (1 if rain > threshold)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with weather data

    Returns
    -------
    pd.DataFrame
        Dataframe with target variables added
    """
    logger.info("\n" + "=" * 60)
    logger.info("CREATING TARGET VARIABLES")
    logger.info("=" * 60)

    df_targets = df.copy()

    # Regression target: next-day mean temperature (Geneva)
    if REGRESSION_TARGET in df.columns:
        df_targets['target_temp_next_day'] = df_targets[REGRESSION_TARGET].shift(-1)
        logger.info(f"--OK-- Regression target: next-day {REGRESSION_TARGET}")

    # Classification target: rain binary (Geneva)
    if 'precipitation_gve' in df.columns:
        # Current day rain
        df_targets['rain_today'] = (df_targets['precipitation_gve'] > RAIN_THRESHOLD).astype(int)

        # Next day rain (prediction target)
        df_targets['target_rain_next_day'] = df_targets['rain_today'].shift(-1)

        rain_days = df_targets['rain_today'].sum()
        rain_pct = (rain_days / len(df_targets)) * 100

        logger.info(f"--OK-- Classification target: rain > {RAIN_THRESHOLD}mm")
        logger.info(f"  Rainy days: {rain_days} ({rain_pct:.1f}%)")
        logger.info(f"  Dry days: {len(df_targets) - rain_days} ({100 - rain_pct:.1f}%)")

    return df_targets


def handle_missing_values_from_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle NaN values introduced by feature engineering.

    Lag and rolling features create NaN in first rows.
    Target variables create NaN in last row.

    Strategy: Drop rows with NaN in targets or features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with features and targets

    Returns
    -------
    pd.DataFrame
        Dataframe with NaN rows removed
    """
    logger.info("\n" + "=" * 60)
    logger.info("HANDLING MISSING VALUES FROM FEATURE ENGINEERING")
    logger.info("=" * 60)

    initial_rows = len(df)
    logger.info(f"Initial rows: {initial_rows}")

    # Check NaN counts
    nan_counts = df.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]

    if len(cols_with_nan) > 0:
        logger.info(f"\nColumns with NaN:")
        for col, count in cols_with_nan.items():
            logger.info(f"  {col}: {count} NaN")

    # Drop rows with any NaN
    df_clean = df.dropna()

    final_rows = len(df_clean)
    rows_dropped = initial_rows - final_rows

    logger.info(f"\nFinal rows: {final_rows}")
    logger.info(f"Rows dropped: {rows_dropped}")
    logger.info(f"Percentage retained: {(final_rows / initial_rows) * 100:.2f}%")

    return df_clean


def build_complete_feature_set(save_output: bool = True) -> pd.DataFrame:
    """
    Build complete feature set from master dataset.

    Pipeline:
    1. Load master dataset
    2. Add temporal features
    3. Add lag features
    4. Add rolling features
    5. Add derived features
    6. Add cross-station features
    7. Create target variables
    8. Handle missing values
    9. Save feature dataset

    Parameters
    ----------
    save_output : bool
        Whether to save the feature dataset

    Returns
    -------
    pd.DataFrame
        Complete feature dataset
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 80)

    # Step 1: Load master dataset
    logger.info("\nStep 1: Loading master dataset...")
    df = pd.read_csv(MASTER_DATASET_FILE, parse_dates=['date'])
    logger.info(f"Loaded: {df.shape}")

    # Step 2: Temporal features
    logger.info("\nStep 2: Adding temporal features...")
    df = add_all_temporal_features(df)
    logger.info(f"Current shape: {df.shape}")

    # Step 3: Lag features
    logger.info("\nStep 3: Adding lag features...")
    df = add_all_lag_features(df)
    logger.info(f"Current shape: {df.shape}")

    # Step 4: Rolling features
    logger.info("\nStep 4: Adding rolling features...")
    df = add_all_rolling_features(df)
    logger.info(f"Current shape: {df.shape}")

    # Step 5: Derived features
    logger.info("\nStep 5: Adding derived features...")
    df = add_all_derived_features(df)
    logger.info(f"Current shape: {df.shape}")

    # Step 6: Cross-station features
    logger.info("\nStep 6: Adding cross-station features...")
    df = add_all_cross_station_features(df)
    logger.info(f"Current shape: {df.shape}")

    # Step 7: Create targets
    logger.info("\nStep 7: Creating target variables...")
    df = create_target_variables(df)
    logger.info(f"Current shape: {df.shape}")

    # Step 8: Handle missing values
    logger.info("\nStep 8: Handling missing values...")
    df = handle_missing_values_from_features(df)
    logger.info(f"Current shape: {df.shape}")

    # Step 9: Save
    if save_output:
        logger.info("\nStep 9: Saving feature dataset...")
        df.to_csv(FEATURES_DATASET_FILE, index=False)
        logger.info(f"--OK-- Saved to: {FEATURES_DATASET_FILE}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE ENGINEERING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nFinal dataset:")
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"  Features: {df.shape[1] - 3} (excluding date + 2 targets)")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    # List feature categories
    temporal_feats = [c for c in df.columns if any(x in c for x in ['_sin', '_cos', 'is_', 'season'])]
    lag_feats = [c for c in df.columns if 'lag' in c]
    roll_feats = [c for c in df.columns if 'roll' in c]
    derived_feats = [c for c in df.columns if
                     any(x in c for x in ['tendency', 'range', 'wind_u', 'wind_v', 'dew', 'vpd'])]
    cross_feats = [c for c in df.columns if any(x in c for x in ['_diff', 'gradient', 'for_'])]

    logger.info(f"\nFeature breakdown:")
    logger.info(f"  Temporal: {len(temporal_feats)}")
    logger.info(f"  Lag: {len(lag_feats)}")
    logger.info(f"  Rolling: {len(roll_feats)}")
    logger.info(f"  Derived: {len(derived_feats)}")
    logger.info(f"  Cross-station: {len(cross_feats)}")
    logger.info(f"  Original: 24 (meteorological variables)")
    logger.info(
        f"  Total: {len(temporal_feats) + len(lag_feats) + len(roll_feats) + len(derived_feats) + len(cross_feats) + 24}")

    return df


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(PROJECT_ROOT / 'feature_engineering.log'),
            logging.StreamHandler()
        ]
    )

    # Build complete feature set
    df = build_complete_feature_set(save_output=True)

    print("\n--OK-- Feature engineering pipeline completed successfully!")
    print(f"   Feature dataset: {FEATURES_DATASET_FILE}")