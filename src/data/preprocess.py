"""
Data preprocessing pipeline for weather forecasting project.
Complete pipeline from raw data to merged master dataset.
"""

import pandas as pd
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_both_stations
from src.data.clean_data import clean_geneva_data, clean_pully_data
from src.utils.config import (
    MASTER_DATASET_FILE, 
    PROCESSED_DATA_DIR,
    GENEVA_CLEAN_FILE,
    PULLY_CLEAN_FILE,
    DATA_DICTIONARY_FILE
)

logger = logging.getLogger(__name__)


def merge_stations(gve: pd.DataFrame, puy: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Geneva and Pully datasets on date.
    
    Parameters
    ----------
    gve : pd.DataFrame
        Geneva data
    puy : pd.DataFrame
        Pully data
    
    Returns
    -------
    pd.DataFrame
        Merged dataset
    """
    logger.info("\n" + "="*60)
    logger.info("MERGING STATIONS")
    logger.info("="*60)
    
    # Drop station_abbr before merge
    gve_merge = gve.drop(columns=['station_abbr'], errors='ignore')
    puy_merge = puy.drop(columns=['station_abbr'], errors='ignore')
    
    # Merge on date (inner join)
    df = gve_merge.merge(puy_merge, on='date', how='inner')
    
    logger.info(f"Merged dataset: {len(df):,} records")
    logger.info(f"Total variables: {len(df.columns)-1} (excluding date)")
    
    # Verify merge integrity
    logger.info("\n--OK-- Merge Verification:")
    logger.info(f"  - No missing dates: {df['date'].isna().sum() == 0}")
    logger.info(f"  - No duplicate dates: {df['date'].duplicated().sum() == 0}")
    
    # Check date continuity
    df_sorted = df.sort_values('date')
    date_diffs = df_sorted['date'].diff()
    expected_diff = pd.Timedelta(days=1)
    gaps = date_diffs[date_diffs != expected_diff].dropna()
    
    if len(gaps) > 0:
        logger.warning(f"--WARNING-- Found {len(gaps)} date gaps!")
    else:
        logger.info(f"  --OK-- Perfect date continuity")
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic temporal features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with date column
    
    Returns
    -------
    pd.DataFrame
        Dataframe with temporal features added
    """
    logger.info("\nAdding basic temporal features...")
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    logger.info("--OK-- Added: year, month, day_of_year, quarter, day_of_week, week_of_year")
    
    return df


def create_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create data dictionary with variable descriptions and statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Master dataset
    
    Returns
    -------
    pd.DataFrame
        Data dictionary
    """
    logger.info("\nCreating data dictionary...")
    
    data_dict = {
        'Variable': list(df.columns),
        'Type': [str(df[col].dtype) for col in df.columns],
        'Non-Null': [df[col].notna().sum() for col in df.columns],
        'Null': [df[col].isna().sum() for col in df.columns],
        'Mean': [df[col].mean() if df[col].dtype in ['float64', 'int64'] else None 
                 for col in df.columns],
        'Std': [df[col].std() if df[col].dtype in ['float64', 'int64'] else None 
                for col in df.columns],
        'Min': [df[col].min() if df[col].dtype in ['float64', 'int64'] else None 
                for col in df.columns],
        'Max': [df[col].max() if df[col].dtype in ['float64', 'int64'] else None 
                for col in df.columns]
    }
    
    dict_df = pd.DataFrame(data_dict)
    
    return dict_df


def run_preprocessing_pipeline():
    """
    Run complete data preprocessing pipeline.
    
    Steps:
    1. Load raw data from both stations
    2. Clean data (handle missing values)
    3. Merge stations
    4. Add temporal features
    5. Save processed data
    6. Create data dictionary
    """
    logger.info("\n" + "="*80)
    logger.info("DATA PREPROCESSING PIPELINE")
    logger.info("="*80)
    
    # Step 1: Load data
    logger.info("\nSTEP 1: Loading raw data...")
    gve, puy = load_both_stations()
    
    # Step 2: Clean data
    logger.info("\nSTEP 2: Cleaning data...")
    gve_clean = clean_geneva_data(gve)
    puy_clean = clean_pully_data(puy)
    
    # Save cleaned individual station data
    gve_clean.to_csv(GENEVA_CLEAN_FILE, index=False)
    puy_clean.to_csv(PULLY_CLEAN_FILE, index=False)
    logger.info(f"\n--OK-- Saved: {GENEVA_CLEAN_FILE}")
    logger.info(f"--OK-- Saved: {PULLY_CLEAN_FILE}")
    
    # Step 3: Merge stations
    logger.info("\nSTEP 3: Merging stations...")
    df = merge_stations(gve_clean, puy_clean)
    
    # Step 4: Add temporal features
    logger.info("\nSTEP 4: Adding temporal features...")
    df = add_temporal_features(df)
    
    # Step 5: Save master dataset
    logger.info("\nSTEP 5: Saving master dataset...")
    df.to_csv(MASTER_DATASET_FILE, index=False)
    logger.info(f"--OK-- Saved: {MASTER_DATASET_FILE}")
    
    # Step 6: Create and save data dictionary
    logger.info("\nSTEP 6: Creating data dictionary...")
    data_dict = create_data_dictionary(df)
    data_dict.to_csv(DATA_DICTIONARY_FILE, index=False)
    logger.info(f"--OK-- Saved: {DATA_DICTIONARY_FILE}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nMaster dataset:")
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"  Duration: {(df['date'].max() - df['date'].min()).days} days")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"\nFiles saved to: {PROCESSED_DATA_DIR}")
    
    return df


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(PROJECT_ROOT / 'preprocessing.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run pipeline
    df = run_preprocessing_pipeline()
    
    print("\n--OK-- Preprocessing pipeline completed successfully!")
    print(f"   Master dataset: {MASTER_DATASET_FILE}")