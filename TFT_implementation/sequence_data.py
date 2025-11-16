"""
Sequence data preparation for Temporal Fusion Transformer (TFT).
Converts tabular weather data into sequences for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import FEATURES_DATASET_FILE, TRAIN_END_DATE, VAL_END_DATE

logger = logging.getLogger(__name__)


class SequenceDataGenerator:
    """
    Generate sequences for TFT training.

    TFT requires three types of inputs:
    1. Static covariates: Don't change over time (e.g., station info)
    2. Known future inputs: Known at prediction time (e.g., time features)
    3. Unknown inputs: Only known historically (e.g., temperature, rain)
    """

    def __init__(self, sequence_length: int = 30, prediction_horizon: int = 1):
        """
        Initialize sequence generator.

        Parameters
        ----------
        sequence_length : int
            Number of historical time steps (lookback window)
        prediction_horizon : int
            Number of future time steps to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Will be set during fit
        self.feature_names = None
        self.static_features = None
        self.known_features = None
        self.unknown_features = None
        self.target_col = None

        logger.info(f"Initialized SequenceDataGenerator:")
        logger.info(f"  Sequence length: {sequence_length} days")
        logger.info(f"  Prediction horizon: {prediction_horizon} day(s)")

    def identify_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize features into static, known, and unknown.

        Parameters
        ----------
        df : pd.DataFrame
            Feature dataframe

        Returns
        -------
        dict
            Dictionary with feature categorization
        """
        logger.info("\nCategorizing features for TFT...")

        all_features = [col for col in df.columns
                        if col not in ['date', 'target_temp_next_day',
                                       'target_rain_next_day', 'rain_today']]

        # Known future inputs: Time features (deterministic, always known)
        known_features = [col for col in all_features if any(x in col for x in [
            '_sin', '_cos', 'is_', 'year', 'month', 'day_of_year',
            'quarter', 'day_of_week', 'week_of_year', 'season'
        ])]

        # Static features: None in weather data (but we keep structure for TFT)
        static_features = []

        # Unknown inputs: Everything else (observed variables)
        unknown_features = [col for col in all_features
                            if col not in known_features and col not in static_features]

        logger.info(f"  Static features: {len(static_features)}")
        logger.info(f"  Known features: {len(known_features)} (time features)")
        logger.info(f"  Unknown features: {len(unknown_features)} (observed variables)")
        logger.info(f"  Total features: {len(all_features)}")

        return {
            'static': static_features,
            'known': known_features,
            'unknown': unknown_features,
            'all': all_features
        }

    def create_sequences(self, df: pd.DataFrame, target_col: str) -> Dict[str, np.ndarray]:
        """
        Create sequences from dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with features and target
        target_col : str
            Target column name

        Returns
        -------
        dict
            Dictionary with sequence arrays
        """
        logger.info(f"\nCreating sequences for {len(df)} samples...")

        # Identify feature types
        feature_types = self.identify_feature_types(df)
        self.static_features = feature_types['static']
        self.known_features = feature_types['known']
        self.unknown_features = feature_types['unknown']
        self.target_col = target_col

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Extract arrays
        known_array = df[self.known_features].values
        unknown_array = df[self.unknown_features].values
        target_array = df[target_col].values

        # Create sequences
        sequences_known = []
        sequences_unknown = []
        targets = []

        # Generate sequences with sliding window
        for i in range(len(df) - self.sequence_length - self.prediction_horizon + 1):
            # Historical window
            seq_known = known_array[i:i + self.sequence_length]
            seq_unknown = unknown_array[i:i + self.sequence_length]

            # Target (next day)
            target = target_array[i + self.sequence_length]

            sequences_known.append(seq_known)
            sequences_unknown.append(seq_unknown)
            targets.append(target)

        sequences_known = np.array(sequences_known)
        sequences_unknown = np.array(sequences_unknown)
        targets = np.array(targets)

        logger.info(f"  Generated {len(sequences_known)} sequences")
        logger.info(f"  Known features shape: {sequences_known.shape}")
        logger.info(f"  Unknown features shape: {sequences_unknown.shape}")
        logger.info(f"  Targets shape: {targets.shape}")

        return {
            'known': sequences_known,
            'unknown': sequences_unknown,
            'targets': targets,
            'feature_types': feature_types
        }

    def prepare_data_splits(self, df: pd.DataFrame, target_col: str) -> Dict:
        """
        Prepare train/val/test splits with sequences.

        Parameters
        ----------
        df : pd.DataFrame
            Complete feature dataset
        target_col : str
            Target column name ('target_temp_next_day' or 'target_rain_next_day')

        Returns
        -------
        dict
            Dictionary with train/val/test sequences
        """
        logger.info("\n" + "=" * 80)
        logger.info("PREPARING SEQUENCE DATA FOR TFT")
        logger.info("=" * 80)

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Create temporal splits
        train_mask = df['date'] <= TRAIN_END_DATE
        val_mask = (df['date'] > TRAIN_END_DATE) & (df['date'] <= VAL_END_DATE)
        test_mask = df['date'] > VAL_END_DATE

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        logger.info(f"\nTemporal split:")
        logger.info(
            f"  Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} days)")
        logger.info(f"  Val:   {val_df['date'].min().date()} to {val_df['date'].max().date()} ({len(val_df)} days)")
        logger.info(f"  Test:  {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} days)")

        # Create sequences for each split
        logger.info("\n" + "-" * 80)
        logger.info("TRAIN SET")
        logger.info("-" * 80)
        train_sequences = self.create_sequences(train_df, target_col)

        logger.info("\n" + "-" * 80)
        logger.info("VALIDATION SET")
        logger.info("-" * 80)
        val_sequences = self.create_sequences(val_df, target_col)

        logger.info("\n" + "-" * 80)
        logger.info("TEST SET")
        logger.info("-" * 80)
        test_sequences = self.create_sequences(test_df, target_col)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SEQUENCE PREPARATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nFinal sequence counts:")
        logger.info(f"  Train: {len(train_sequences['targets'])} sequences")
        logger.info(f"  Val:   {len(val_sequences['targets'])} sequences")
        logger.info(f"  Test:  {len(test_sequences['targets'])} sequences")

        return {
            'train': train_sequences,
            'val': val_sequences,
            'test': test_sequences,
            'feature_info': {
                'known_features': self.known_features,
                'unknown_features': self.unknown_features,
                'static_features': self.static_features,
                'n_known': len(self.known_features),
                'n_unknown': len(self.unknown_features),
                'n_static': len(self.static_features),
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon
            }
        }


def prepare_tft_data(task: str = 'regression',
                     sequence_length: int = 30) -> Dict:
    """
    Prepare data for TFT training.

    Parameters
    ----------
    task : str
        'regression' or 'classification'
    sequence_length : int
        Sequence length (lookback window)

    Returns
    -------
    dict
        Complete dataset ready for TFT
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"PREPARING TFT DATA - {task.upper()}")
    logger.info("=" * 80)

    # Load feature dataset
    logger.info(f"\nLoading feature dataset from {FEATURES_DATASET_FILE}")
    df = pd.read_csv(FEATURES_DATASET_FILE, parse_dates=['date'])
    logger.info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Select target
    if task == 'regression':
        target_col = 'target_temp_next_day'
    else:
        target_col = 'target_rain_next_day'

    logger.info(f"Target: {target_col}")

    # Create sequence generator
    generator = SequenceDataGenerator(
        sequence_length=sequence_length,
        prediction_horizon=1
    )

    # Prepare splits
    data = generator.prepare_data_splits(df, target_col)

    # Add task info
    data['task'] = task
    data['target_col'] = target_col

    return data


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Test sequence preparation
    logger.info("Testing sequence data preparation...")

    # Regression
    logger.info("\n" + "=" * 80)
    logger.info("REGRESSION DATA")
    logger.info("=" * 80)
    reg_data = prepare_tft_data(task='regression', sequence_length=30)

    # Classification
    logger.info("\n" + "=" * 80)
    logger.info("CLASSIFICATION DATA")
    logger.info("=" * 80)
    clf_data = prepare_tft_data(task='classification', sequence_length=30)

    logger.info("\n--OK-- Sequence data preparation complete!")