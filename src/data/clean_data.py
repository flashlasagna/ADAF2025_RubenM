"""
Data cleaning functions for weather forecasting project.
Handles missing values, outliers, and data quality issues.
"""

import pandas as pd
import numpy as np
from scipy import interpolate
import logging
from pathlib import Path
from typing import Dict

# Import configuration
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import CORE_VARIABLES

logger = logging.getLogger(__name__)


def analyze_missing_values(df: pd.DataFrame, station_name: str) -> Dict:
    """
    Analyze missing values in dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to analyze
    station_name : str
        Name of station

    Returns
    -------
    Dict
        Dictionary with missing value statistics
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"MISSING VALUE ANALYSIS: {station_name}")
    logger.info(f"{'=' * 60}")

    missing_stats = {}

    for col in df.columns:
        if col not in ['date', 'station_abbr']:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100

            missing_stats[col] = {
                'count': missing_count,
                'percentage': missing_pct
            }

            if missing_pct == 0:
                status = "✅"
            elif missing_pct < 5:
                status = "⚠️"
            else:
                status = "❌"

            logger.info(f"  {status} {col:.<35} {missing_pct:>6.2f}%")

    return missing_stats


def handle_minimal_missing(df: pd.DataFrame,
                           columns: list,
                           method: str = 'ffill') -> pd.DataFrame:
    """
    Handle minimal missing values (<1%) using simple imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with missing values
    columns : list
        List of columns to impute
    method : str
        Imputation method ('ffill', 'bfill', 'interpolate')

    Returns
    -------
    pd.DataFrame
        Dataframe with imputed values
    """
    df_clean = df.copy()

    for col in columns:
        missing_before = df_clean[col].isna().sum()

        if missing_before > 0:
            if method == 'ffill':
                df_clean[col] = df_clean[col].ffill()
            elif method == 'bfill':
                df_clean[col] = df_clean[col].bfill()
            elif method == 'interpolate':
                df_clean[col] = df_clean[col].interpolate(method='linear')

            missing_after = df_clean[col].isna().sum()
            logger.info(f"  {col}: {missing_before} → {missing_after} ({method})")

    return df_clean


def interpolate_sunshine_using_radiation(df: pd.DataFrame,
                                         station_suffix: str) -> pd.DataFrame:
    """
    Interpolate missing sunshine values using radiation relationship.

    Uses polynomial fit of sunshine ~ radiation where both exist,
    then applies to missing sunshine values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with sunshine and radiation columns
    station_suffix : str
        Station suffix ('_gve' or '_puy')

    Returns
    -------
    pd.DataFrame
        Dataframe with interpolated sunshine values
    """
    logger.info(f"\nInterpolating sunshine for {station_suffix}...")

    df_clean = df.copy()

    sunshine_col = f'sunshine{station_suffix}'
    radiation_col = f'radiation{station_suffix}'
    sunshine_hours_col = f'sunshine_hours{station_suffix}'

    # Convert sunshine from minutes to hours
    df_clean[sunshine_hours_col] = df_clean[sunshine_col] / 60

    # Get data where both exist
    both_exist = (~df_clean[sunshine_hours_col].isna()) & \
                 (~df_clean[radiation_col].isna())

    sunshine_exist = df_clean.loc[both_exist, sunshine_hours_col].values
    radiation_exist = df_clean.loc[both_exist, radiation_col].values

    # Fit polynomial (degree 2) relationship
    z = np.polyfit(radiation_exist, sunshine_exist, deg=2)
    p = np.poly1d(z)

    logger.info(f"  Polynomial fit: sunshine = {z[0]:.6f}*rad² + {z[1]:.6f}*rad + {z[2]:.2f}")

    # Apply to missing values
    missing_mask = df_clean[sunshine_hours_col].isna()
    missing_count = missing_mask.sum()

    if missing_count > 0:
        df_clean.loc[missing_mask, sunshine_hours_col] = \
            p(df_clean.loc[missing_mask, radiation_col])

        # Clip to physical bounds (0-24 hours)
        df_clean[sunshine_hours_col] = df_clean[sunshine_hours_col].clip(lower=0, upper=24)

        logger.info(f"  ✓ Interpolated {missing_count:,} sunshine values")
        logger.info(
            f"  ✓ Range: {df_clean[sunshine_hours_col].min():.1f} to {df_clean[sunshine_hours_col].max():.1f} hours")

    # Drop original sunshine column
    df_clean = df_clean.drop(columns=[sunshine_col])

    return df_clean


def clean_geneva_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Geneva weather data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Geneva data

    Returns
    -------
    pd.DataFrame
        Cleaned Geneva data
    """
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING GENEVA DATA")
    logger.info("=" * 60)

    # Analyze missing values
    analyze_missing_values(df, "Geneva")

    # Handle minimal missing values
    minimal_missing_cols = ['pressure_gve', 'wind_direction_gve', 'wind_gust_gve']
    df_clean = handle_minimal_missing(df, minimal_missing_cols, method='ffill')

    # Convert sunshine to hours
    df_clean = interpolate_sunshine_using_radiation(df_clean, '_gve')

    # Verify no missing values remain
    logger.info("\n✅ Verification - Remaining Missing Values:")
    remaining_missing = df_clean.isna().sum()
    for col, missing in remaining_missing.items():
        if col not in ['date', 'station_abbr'] and missing > 0:
            logger.warning(f"  ⚠️ {col}: {missing} still missing!")
        elif col not in ['date', 'station_abbr']:
            logger.info(f"  ✅ {col}: 0 missing")

    return df_clean


def clean_pully_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Pully weather data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Pully data

    Returns
    -------
    pd.DataFrame
        Cleaned Pully data
    """
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING PULLY DATA")
    logger.info("=" * 60)

    # Analyze missing values
    analyze_missing_values(df, "Pully")

    # Handle sunshine (major missing values - 15%)
    df_clean = interpolate_sunshine_using_radiation(df, '_puy')

    # Handle minimal missing values
    minimal_missing_cols = ['pressure_puy', 'wind_direction_puy', 'wind_gust_puy']
    df_clean = handle_minimal_missing(df_clean, minimal_missing_cols, method='ffill')

    # Verify no missing values remain
    logger.info("\n✅ Verification - Remaining Missing Values:")
    remaining_missing = df_clean.isna().sum()
    for col, missing in remaining_missing.items():
        if col not in ['date', 'station_abbr'] and missing > 0:
            logger.warning(f"  ⚠️ {col}: {missing} still missing!")
        elif col not in ['date', 'station_abbr']:
            logger.info(f"  ✅ {col}: 0 missing")

    return df_clean


def detect_outliers(df: pd.DataFrame,
                    column: str,
                    method: str = 'iqr',
                    threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in a column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe
    column : str
        Column to check
    method : str
        Method ('iqr' or 'zscore')
    threshold : float
        Threshold for outlier detection

    Returns
    -------
    pd.Series
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = z_scores > threshold

    else:
        raise ValueError(f"Unknown method: {method}")

    return outliers


def validate_physical_constraints(df: pd.DataFrame, station_suffix: str) -> None:
    """
    Validate that values are within physical constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate
    station_suffix : str
        Station suffix
    """
    logger.info(f"\nValidating physical constraints for {station_suffix}...")

    constraints = {
        f'temp_mean{station_suffix}': (-50, 50),  # °C
        f'temp_max{station_suffix}': (-50, 50),
        f'temp_min{station_suffix}': (-50, 50),
        f'humidity{station_suffix}': (0, 100),  # %
        f'pressure{station_suffix}': (900, 1100),  # hPa
        f'precipitation{station_suffix}': (0, 500),  # mm/day
        f'radiation{station_suffix}': (0, 1000),  # W/m²
        f'sunshine_hours{station_suffix}': (0, 24),  # hours
        f'wind_speed{station_suffix}': (0, 50),  # m/s
        f'wind_direction{station_suffix}': (0, 360),  # degrees
        f'evaporation{station_suffix}': (0, 50),  # mm/day
        f'wind_gust{station_suffix}': (0, 100),  # m/s
    }

    violations = 0
    for col, (min_val, max_val) in constraints.items():
        if col in df.columns:
            below_min = (df[col] < min_val).sum()
            above_max = (df[col] > max_val).sum()

            if below_min > 0 or above_max > 0:
                logger.warning(f"  ⚠️ {col}: {below_min} below {min_val}, {above_max} above {max_val}")
                violations += 1

    if violations == 0:
        logger.info("  ✓ All values within physical constraints")
    else:
        logger.warning(f"  ⚠️ {violations} constraint violations found")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Import and clean data
    from load_data import load_both_stations

    gve, puy = load_both_stations()

    gve_clean = clean_geneva_data(gve)
    puy_clean = clean_pully_data(puy)

    # Validate
    validate_physical_constraints(gve_clean, '_gve')
    validate_physical_constraints(puy_clean, '_puy')

    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE")
    print("=" * 60)