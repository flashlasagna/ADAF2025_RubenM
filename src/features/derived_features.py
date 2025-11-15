"""
Derived meteorological feature engineering.
Creates physically-meaningful features from existing variables.
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


def add_pressure_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pressure tendency (change from previous day).

    Pressure tendency is a key indicator of weather system movement.
    Falling pressure → weather deteriorating
    Rising pressure → weather improving

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with pressure columns

    Returns
    -------
    pd.DataFrame
        Dataframe with pressure tendency features
    """
    logger.info("Adding pressure tendency...")

    df_features = df.copy()

    # For both stations
    for suffix in ['_gve', '_puy']:
        pressure_col = f'pressure{suffix}'
        if pressure_col in df.columns:
            new_col = f'pressure_tendency{suffix}'
            df_features[new_col] = df_features[pressure_col].diff()

    logger.info("✓ Added pressure tendency features")

    return df_features


def add_temperature_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily temperature range (max - min).

    Large range → clear skies, continental conditions
    Small range → cloudy, maritime conditions

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with temp_max and temp_min columns

    Returns
    -------
    pd.DataFrame
        Dataframe with temperature range features
    """
    logger.info("Adding temperature range...")

    df_features = df.copy()

    # For both stations
    for suffix in ['_gve', '_puy']:
        temp_max_col = f'temp_max{suffix}'
        temp_min_col = f'temp_min{suffix}'

        if temp_max_col in df.columns and temp_min_col in df.columns:
            new_col = f'temp_range{suffix}'
            df_features[new_col] = df_features[temp_max_col] - df_features[temp_min_col]

    logger.info("✓ Added temperature range features")

    return df_features


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add wind u/v components from speed and direction.

    Converts polar coordinates (speed, direction) to Cartesian (u, v).
    This handles the 0°/360° discontinuity problem.

    u = east-west component (positive = east)
    v = north-south component (positive = north)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with wind_speed and wind_direction columns

    Returns
    -------
    pd.DataFrame
        Dataframe with wind component features
    """
    logger.info("Adding wind components...")

    df_features = df.copy()

    # For both stations
    for suffix in ['_gve', '_puy']:
        speed_col = f'wind_speed{suffix}'
        dir_col = f'wind_direction{suffix}'

        if speed_col in df.columns and dir_col in df.columns:
            # Convert direction to radians
            dir_rad = np.radians(df_features[dir_col])

            # Calculate components
            # Meteorological convention: direction wind comes FROM
            df_features[f'wind_u{suffix}'] = -df_features[speed_col] * np.sin(dir_rad)
            df_features[f'wind_v{suffix}'] = -df_features[speed_col] * np.cos(dir_rad)

    logger.info("✓ Added wind u/v components")

    return df_features


def add_dew_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dew point temperature using Magnus formula (simplified).

    Dew point = temperature at which air becomes saturated.
    Close to actual temp → high humidity, fog likely
    Far from actual temp → dry conditions

    Uses simplified approximation:
    Td ≈ T - ((100 - RH) / 5)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with temp_mean and humidity columns

    Returns
    -------
    pd.DataFrame
        Dataframe with dew point features
    """
    logger.info("Adding dew point...")

    df_features = df.copy()

    # For both stations
    for suffix in ['_gve', '_puy']:
        temp_col = f'temp_mean{suffix}'
        humidity_col = f'humidity{suffix}'

        if temp_col in df.columns and humidity_col in df.columns:
            new_col = f'dew_point{suffix}'
            # Simplified Magnus formula
            df_features[new_col] = df_features[temp_col] - ((100 - df_features[humidity_col]) / 5)

    logger.info("✓ Added dew point features")

    return df_features


def add_vapor_pressure_deficit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add vapor pressure deficit (VPD).

    VPD = saturation vapor pressure - actual vapor pressure
    Indicates atmospheric drying power (important for evaporation).

    Uses Magnus-Tetens formula.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with temp_mean and humidity columns

    Returns
    -------
    pd.DataFrame
        Dataframe with VPD features
    """
    logger.info("Adding vapor pressure deficit...")

    df_features = df.copy()

    # For both stations
    for suffix in ['_gve', '_puy']:
        temp_col = f'temp_mean{suffix}'
        humidity_col = f'humidity{suffix}'

        if temp_col in df.columns and humidity_col in df.columns:
            # Saturation vapor pressure (hPa)
            es = 0.611 * np.exp(17.27 * df_features[temp_col] /
                                (df_features[temp_col] + 237.3))

            # Actual vapor pressure
            ea = es * (df_features[humidity_col] / 100)

            # VPD
            new_col = f'vpd{suffix}'
            df_features[new_col] = es - ea

    logger.info("✓ Added VPD features")

    return df_features


def add_all_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived meteorological features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with base meteorological variables

    Returns
    -------
    pd.DataFrame
        Dataframe with all derived features
    """
    logger.info("\n" + "=" * 60)
    logger.info("DERIVED FEATURE ENGINEERING")
    logger.info("=" * 60)

    # Add each type of derived feature
    df_features = add_pressure_tendency(df)
    df_features = add_temperature_range(df_features)
    df_features = add_wind_components(df_features)
    df_features = add_dew_point(df_features)
    df_features = add_vapor_pressure_deficit(df_features)

    # Summary
    derived_features = [col for col in df_features.columns if col not in df.columns]

    logger.info("\n" + "=" * 60)
    logger.info(f"DERIVED FEATURES COMPLETE: {len(derived_features)} features added")
    logger.info("=" * 60)
    logger.info(f"\nNew features: {derived_features}")

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

    # Add derived features
    df_features = add_all_derived_features(df)

    logger.info(f"New shape: {df_features.shape}")
    logger.info(f"Features added: {df_features.shape[1] - df.shape[1]}")

    # Show sample
    logger.info("\nSample derived features:")
    derived_cols = [col for col in df_features.columns if col not in df.columns]
    print(df_features[['date'] + derived_cols[:5]].head(10))

    # Check for invalid values
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION")
    logger.info("=" * 60)
    for col in derived_cols:
        n_nan = df_features[col].isna().sum()
        n_inf = np.isinf(df_features[col]).sum()
        if n_nan > 0 or n_inf > 0:
            logger.warning(f"  {col}: {n_nan} NaN, {n_inf} Inf")
        else:
            logger.info(f"  ✓ {col}: valid")