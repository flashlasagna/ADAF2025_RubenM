"""
Cross-station feature engineering.
Exploits spatial relationships between Geneva and Pully stations.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import STATION_DISTANCE

logger = logging.getLogger(__name__)


def add_temperature_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temperature difference between stations.

    Indicates local effects (lake influence, urban heat island, etc.)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with temp columns from both stations

    Returns
    -------
    pd.DataFrame
        Dataframe with temperature difference features
    """
    logger.info("Adding temperature differences...")

    df_features = df.copy()

    # Mean temperature difference
    if 'temp_mean_gve' in df.columns and 'temp_mean_puy' in df.columns:
        df_features['temp_mean_diff'] = (df_features['temp_mean_gve'] -
                                         df_features['temp_mean_puy'])

    # Max temperature difference
    if 'temp_max_gve' in df.columns and 'temp_max_puy' in df.columns:
        df_features['temp_max_diff'] = (df_features['temp_max_gve'] -
                                        df_features['temp_max_puy'])

    # Min temperature difference
    if 'temp_min_gve' in df.columns and 'temp_min_puy' in df.columns:
        df_features['temp_min_diff'] = (df_features['temp_min_gve'] -
                                        df_features['temp_min_puy'])

    logger.info("--OK-- Added temperature difference features")

    return df_features


def add_pressure_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pressure gradient between stations.

    Pressure gradient drives wind and indicates weather system movement.
    Normalized by distance between stations.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with pressure columns from both stations

    Returns
    -------
    pd.DataFrame
        Dataframe with pressure gradient features
    """
    logger.info("Adding pressure gradient...")

    df_features = df.copy()

    if 'pressure_gve' in df.columns and 'pressure_puy' in df.columns:
        # Pressure difference (hPa)
        pressure_diff = df_features['pressure_gve'] - df_features['pressure_puy']

        # Gradient (hPa per 100km)
        df_features['pressure_gradient'] = (pressure_diff / STATION_DISTANCE) * 100

        # Also keep absolute difference
        df_features['pressure_diff'] = pressure_diff

    logger.info("--OK-- Added pressure gradient features")

    return df_features


def add_humidity_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add humidity difference between stations.

    Indicates moisture gradients (lake effect, local weather).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with humidity columns from both stations

    Returns
    -------
    pd.DataFrame
        Dataframe with humidity difference feature
    """
    logger.info("Adding humidity difference...")

    df_features = df.copy()

    if 'humidity_gve' in df.columns and 'humidity_puy' in df.columns:
        df_features['humidity_diff'] = (df_features['humidity_gve'] -
                                        df_features['humidity_puy'])

    logger.info("--OK-- Added humidity difference feature")

    return df_features


def add_spatial_precipitation_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add spatial lag for precipitation.

    If it rained in Geneva yesterday, might rain in Pully today
    (weather systems move ~ 50-100 km/day).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with precipitation columns

    Returns
    -------
    pd.DataFrame
        Dataframe with spatial lag features
    """
    logger.info("Adding spatial precipitation lags...")

    df_features = df.copy()

    if 'precipitation_gve' in df.columns:
        # Geneva yesterday → Pully predictor
        df_features['precip_gve_lag1_for_puy'] = df_features['precipitation_gve'].shift(1)

    if 'precipitation_puy' in df.columns:
        # Pully yesterday → Geneva predictor (less likely but possible)
        df_features['precip_puy_lag1_for_gve'] = df_features['precipitation_puy'].shift(1)

    logger.info("--OK-- Added spatial precipitation lag features")

    return df_features


def add_wind_convergence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add wind convergence/divergence indicator.

    If winds blow toward each other → convergence → lifting → rain
    If winds blow apart → divergence → sinking → clear

    Using u/v components if available.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with wind component columns

    Returns
    -------
    pd.DataFrame
        Dataframe with wind convergence features
    """
    logger.info("Adding wind convergence...")

    df_features = df.copy()

    # Check if wind components exist
    has_components = ('wind_u_gve' in df.columns and 'wind_v_gve' in df.columns and
                      'wind_u_puy' in df.columns and 'wind_v_puy' in df.columns)

    if has_components:
        # U-component difference (east-west)
        df_features['wind_u_diff'] = df_features['wind_u_gve'] - df_features['wind_u_puy']

        # V-component difference (north-south)
        df_features['wind_v_diff'] = df_features['wind_v_gve'] - df_features['wind_v_puy']

        logger.info("--OK-- Added wind component differences")
    else:
        logger.warning("--WARNING-- Wind components not found, skipping wind convergence")

    return df_features


def add_radiation_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add radiation difference between stations.

    Different radiation → different cloud cover or fog conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with radiation columns

    Returns
    -------
    pd.DataFrame
        Dataframe with radiation difference feature
    """
    logger.info("Adding radiation difference...")

    df_features = df.copy()

    if 'radiation_gve' in df.columns and 'radiation_puy' in df.columns:
        df_features['radiation_diff'] = (df_features['radiation_gve'] -
                                         df_features['radiation_puy'])

    logger.info("--OK-- Added radiation difference feature")

    return df_features


def add_all_cross_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all cross-station features.

    Exploits the panel data structure (2 stations).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with data from both stations

    Returns
    -------
    pd.DataFrame
        Dataframe with all cross-station features
    """
    logger.info("\n" + "=" * 60)
    logger.info("CROSS-STATION FEATURE ENGINEERING")
    logger.info("=" * 60)
    logger.info(f"Station distance: {STATION_DISTANCE} km")

    # Add each type of cross-station feature
    df_features = add_temperature_difference(df)
    df_features = add_pressure_gradient(df_features)
    df_features = add_humidity_difference(df_features)
    df_features = add_spatial_precipitation_lag(df_features)
    df_features = add_wind_convergence(df_features)
    df_features = add_radiation_difference(df_features)

    # Summary
    cross_features = [col for col in df_features.columns if col not in df.columns]

    logger.info("\n" + "=" * 60)
    logger.info(f"CROSS-STATION FEATURES COMPLETE: {len(cross_features)} features added")
    logger.info("=" * 60)
    logger.info(f"\nNew features: {cross_features}")

    return df_features


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Test on master dataset with derived features
    from src.utils.config import MASTER_DATASET_FILE
    from derived_features import add_all_derived_features

    logger.info("Loading master dataset...")
    df = pd.read_csv(MASTER_DATASET_FILE, parse_dates=['date'])

    # Need derived features first (for wind components)
    logger.info("Adding derived features first...")
    df = add_all_derived_features(df)

    logger.info(f"Original shape: {df.shape}")

    # Add cross-station features
    df_features = add_all_cross_station_features(df)

    logger.info(f"New shape: {df_features.shape}")
    logger.info(f"Features added: {df_features.shape[1] - df.shape[1]}")

    # Show sample
    logger.info("\nSample cross-station features:")
    cross_cols = [col for col in df_features.columns if col not in df.columns]
    print(df_features[['date'] + cross_cols[:5]].head(10))

    # Statistics
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE STATISTICS")
    logger.info("=" * 60)
    for col in cross_cols[:5]:
        logger.info(f"{col}:")
        logger.info(f"  Mean: {df_features[col].mean():.3f}")
        logger.info(f"  Std:  {df_features[col].std():.3f}")
        logger.info(f"  Min:  {df_features[col].min():.3f}")
        logger.info(f"  Max:  {df_features[col].max():.3f}")