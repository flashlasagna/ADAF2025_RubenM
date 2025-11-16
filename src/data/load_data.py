"""
Data loading functions for weather forecasting project.
Handles loading raw data from MeteoSwiss CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import (
    GENEVA_RAW_FILE, PULLY_RAW_FILE, CORE_VARIABLES,
    START_DATE, END_DATE
)

logger = logging.getLogger(__name__)


def load_raw_data(filepath: Path, station_name: str) -> pd.DataFrame:
    """
    Load raw MeteoSwiss data from CSV file.

    Parameters
    ----------
    filepath : Path
        Path to CSV file
    station_name : str
        Name of station (for logging)

    Returns
    -------
    pd.DataFrame
        Raw data with datetime index
    """
    logger.info(f"Loading {station_name} data from {filepath}")

    try:
        df = pd.read_csv(filepath, sep=';', encoding='latin-1')
        logger.info(f"  Loaded {len(df):,} records")

        # Convert timestamp
        df['date'] = pd.to_datetime(df['reference_timestamp'],
                                     format='%d.%m.%Y %H:%M')

        return df

    except Exception as e:
        logger.error(f"Error loading {station_name} data: {e}")
        raise


def filter_date_range(df: pd.DataFrame,
                      start_date: str = START_DATE,
                      end_date: str = END_DATE) -> pd.DataFrame:
    """
    Filter dataframe to specified date range.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'date' column
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    logger.info(f"Filtering to {start_date} - {end_date}")

    df_filtered = df[(df['date'] >= start_date) &
                     (df['date'] <= end_date)].copy()

    logger.info(f"  Retained {len(df_filtered):,} records")

    return df_filtered


def select_core_variables(df: pd.DataFrame,
                         station_suffix: str) -> pd.DataFrame:
    """
    Select core meteorological variables and rename.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe
    station_suffix : str
        Station suffix (e.g., '_gve', '_puy')

    Returns
    -------
    pd.DataFrame
        Dataframe with selected and renamed variables
    """
    logger.info(f"Selecting {len(CORE_VARIABLES)} core variables")

    # Select date, station, and core variables
    columns_to_keep = ['date', 'station_abbr'] + list(CORE_VARIABLES.keys())

    # Check which variables exist
    available_vars = [col for col in columns_to_keep if col in df.columns]
    missing_vars = [col for col in columns_to_keep if col not in df.columns]

    if missing_vars:
        logger.warning(f"  Missing variables: {missing_vars}")

    df_selected = df[available_vars].copy()

    # Rename variables
    rename_dict = {old: f"{new}{station_suffix}"
                   for old, new in CORE_VARIABLES.items()
                   if old in df.columns}

    df_selected.rename(columns=rename_dict, inplace=True)

    logger.info(f"  Selected {len(rename_dict)} variables")

    return df_selected


def load_geneva_data() -> pd.DataFrame:
    """
    Load and preprocess Geneva weather data.

    Returns
    -------
    pd.DataFrame
        Processed Geneva data (2000-2024)
    """
    logger.info("="*60)
    logger.info("LOADING GENEVA DATA")
    logger.info("="*60)

    # Load raw data
    df = load_raw_data(GENEVA_RAW_FILE, "Geneva")

    # Filter date range
    df = filter_date_range(df)

    # Select core variables
    df = select_core_variables(df, '_gve')

    logger.info(f"Geneva data ready: {df.shape}")

    return df


def load_pully_data() -> pd.DataFrame:
    """
    Load and preprocess Pully weather data.

    Returns
    -------
    pd.DataFrame
        Processed Pully data (2000-2024)
    """
    logger.info("="*60)
    logger.info("LOADING PULLY DATA")
    logger.info("="*60)

    # Load raw data
    df = load_raw_data(PULLY_RAW_FILE, "Pully")

    # Filter date range
    df = filter_date_range(df)

    # Select core variables
    df = select_core_variables(df, '_puy')

    logger.info(f"Pully data ready: {df.shape}")

    return df


def load_both_stations() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from both weather stations.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (Geneva data, Pully data)
    """
    logger.info("="*60)
    logger.info("LOADING BOTH STATIONS")
    logger.info("="*60)

    gve = load_geneva_data()
    puy = load_pully_data()

    logger.info(f"\nData loaded:")
    logger.info(f"  Geneva: {len(gve):,} records, {gve.shape[1]} columns")
    logger.info(f"  Pully:  {len(puy):,} records, {puy.shape[1]} columns")

    return gve, puy


def verify_data_integrity(df: pd.DataFrame, station_name: str) -> None:
    """
    Verify data integrity (no gaps, duplicates, etc.)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to verify
    station_name : str
        Name of station (for logging)
    """
    logger.info(f"\nVerifying {station_name} data integrity:")

    # Check for missing dates
    missing_dates = df['date'].isna().sum()
    logger.info(f"  Missing dates: {missing_dates}")

    # Check for duplicates
    duplicates = df['date'].duplicated().sum()
    logger.info(f"  Duplicate dates: {duplicates}")

    # Check date continuity
    df_sorted = df.sort_values('date')
    date_diffs = df_sorted['date'].diff()
    expected_diff = pd.Timedelta(days=1)
    gaps = date_diffs[date_diffs != expected_diff].dropna()

    if len(gaps) > 0:
        logger.warning(f"--WARNING-- Found {len(gaps)} date gaps!")
    else:
        logger.info(f"  --OK-- Perfect date continuity")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load data
    gve, puy = load_both_stations()

    # Verify integrity
    verify_data_integrity(gve, "Geneva")
    verify_data_integrity(puy, "Pully")

    print("\n" + "="*60)
    print("DATA LOADING COMPLETE")
    print("="*60)