"""
Architecture Search for Temporal Fusion Transformer (TFT).
Systematic search over architecture and training hyperparameters.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys
import json
import time
from itertools import product
from sklearn.preprocessing import StandardScaler
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from sequence_data import prepare_tft_data
from tft_model import TemporalFusionTransformer

from src.utils.config import MODELS_DIR, TABLES_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================================
# DATA NORMALIZATION
# ============================================================================

def normalize_sequences(data: dict, task: str) -> dict:
    """
    Normalize sequence data using StandardScaler.
    Critical for deep learning performance!

    Parameters
    ----------
    data : dict
        Dictionary with train/val/test sequences
    task : str
        'regression' or 'classification'

    Returns
    -------
    dict
        Normalized data with scalers
    """
    logger.info("\n" + "=" * 80)
    logger.info("NORMALIZING SEQUENCES")
    logger.info("=" * 80)

    # Fit scalers on training data
    logger.info("\nFitting scalers on training data...")

    # Known features scaler
    n_samples_train = data['train']['known'].shape[0]
    n_seq = data['train']['known'].shape[1]
    n_known = data['train']['known'].shape[2]

    known_scaler = StandardScaler()
    known_train_flat = data['train']['known'].reshape(-1, n_known)
    known_scaler.fit(known_train_flat)

    # Unknown features scaler
    n_unknown = data['train']['unknown'].shape[2]
    unknown_scaler = StandardScaler()
    unknown_train_flat = data['train']['unknown'].reshape(-1, n_unknown)
    unknown_scaler.fit(unknown_train_flat)

    # Target scaler (regression only)
    if task == 'regression':
        target_scaler = StandardScaler()
        target_scaler.fit(data['train']['targets'].reshape(-1, 1))
    else:
        target_scaler = None

    logger.info("--OK-- Scalers fitted")

    # Transform all splits
    logger.info("\nTransforming all splits...")
    normalized_data = {
        'train': {},
        'val': {},
        'test': {},
        'feature_info': data['feature_info'],
        'task': data['task'],
        'target_col': data['target_col']
    }

    for split in ['train', 'val', 'test']:
        # Known features
        known_flat = data[split]['known'].reshape(-1, n_known)
        known_normalized = known_scaler.transform(known_flat)
        normalized_data[split]['known'] = known_normalized.reshape(
            data[split]['known'].shape
        )

        # Unknown features
        unknown_flat = data[split]['unknown'].reshape(-1, n_unknown)
        unknown_normalized = unknown_scaler.transform(unknown_flat)
        normalized_data[split]['unknown'] = unknown_normalized.reshape(
            data[split]['unknown'].shape
        )

        # Targets
        if task == 'regression':
            targets_normalized = target_scaler.transform(
                data[split]['targets'].reshape(-1, 1)
            ).flatten()
            normalized_data[split]['targets'] = targets_normalized
        else:
            normalized_data[split]['targets'] = data[split]['targets']

        # Keep feature_types
        if 'feature_types' in data[split]:
            normalized_data[split]['feature_types'] = data[split]['feature_types']

    logger.info("--OK-- All splits normalized")

    # Store scalers
    normalized_data['scalers'] = {
        'known': known_scaler,
        'unknown': unknown_scaler,
        'target': target_scaler
    }

    logger.info("\n" + "=" * 80)
    logger.info("NORMALIZATION COMPLETE")
    logger.info("=" * 80)

    return normalized_data


def create_tf_dataset(sequences: dict, batch_size: int = 32, shuffle: bool = True):
    """Create TensorFlow dataset from sequences."""
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'known': sequences['known'].astype(np.float32),
            'unknown': sequences['unknown'].astype(np.float32)
        },
        sequences['targets'].astype(np.float32)
    ))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=RANDOM_SEED)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ============================================================================
# ARCHITECTURE SEARCH
# ============================================================================

def train_single_config(config: dict, data: dict, task: str) -> dict:
    """
    Train TFT with a single configuration.

    Parameters
    ----------
    config : dict
        Hyperparameter configuration
    data : dict
        Normalized sequence data
    task : str
        'regression' or 'classification'

    Returns
    -------
    dict
        Training results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Training configuration:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"{'=' * 80}")

    # Create datasets
    train_dataset = create_tf_dataset(
        data['train'],
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_dataset = create_tf_dataset(
        data['val'],
        batch_size=config['batch_size'],
        shuffle=False
    )

    # Build model
    model = TemporalFusionTransformer(
        n_known_features=data['feature_info']['n_known'],
        n_unknown_features=data['feature_info']['n_unknown'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_lstm_layers=config['num_lstm_layers'],
        dropout_rate=config['dropout_rate'],
        task=task
    )

    # Compile
    if task == 'regression':
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
    else:
        # Add class weights for imbalanced classification
        train_targets = data['train']['targets']
        n_positive = np.sum(train_targets)
        n_negative = len(train_targets) - n_positive

        # Compute class weights
        total = len(train_targets)
        weight_for_0 = total / (2.0 * n_negative)
        weight_for_1 = total / (2.0 * n_positive)

        class_weight = {0: weight_for_0, 1: weight_for_1}

        logger.info(f"\nClass distribution:")
        logger.info(f"  No rain (0): {n_negative} ({n_negative / total * 100:.1f}%)")
        logger.info(f"  Rain (1): {n_positive} ({n_positive / total * 100:.1f}%)")
        logger.info(f"  Class weights: {class_weight}")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
    ]

    # Train
    start_time = time.time()

    if task == 'classification':
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config['max_epochs'],
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0
        )
    else:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config['max_epochs'],
            callbacks=callbacks,
            verbose=0
        )

    training_time = time.time() - start_time

    # Get best validation loss
    best_val_loss = min(history.history['val_loss'])
    epochs_trained = len(history.history['loss'])

    # Evaluate on test set
    test_dataset = create_tf_dataset(
        data['test'],
        batch_size=config['batch_size'],
        shuffle=False
    )
    test_results = model.evaluate(test_dataset, verbose=0)
    test_loss = test_results[0]

    # Convert test loss to RMSE for regression
    if task == 'regression':
        # Denormalize RMSE
        target_scaler = data['scalers']['target']
        test_rmse_normalized = np.sqrt(test_loss)
        test_rmse = test_rmse_normalized * target_scaler.scale_[0]
        metric_value = test_rmse
        metric_name = 'RMSE'
    else:
        # For classification, use AUC
        test_auc = test_results[2]  # AUC is 3rd metric
        metric_value = test_auc
        metric_name = 'AUC'

    logger.info(f"\n--OK-- Training complete:")
    logger.info(f"  Epochs: {epochs_trained}")
    logger.info(f"  Training time: {training_time:.1f}s")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")
    logger.info(f"  Test {metric_name}: {metric_value:.4f}")

    return {
        'config': config,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_metric': metric_value,
        'metric_name': metric_name,
        'epochs_trained': epochs_trained,
        'training_time': training_time,
        'history': history.history
    }


def architecture_search(task: str,
                        sequence_length: int = 30,
                        max_trials: int = 30) -> dict:
    """
    Perform architecture search for TFT.

    Parameters
    ----------
    task : str
        'regression' or 'classification'
    sequence_length : int
        Sequence length
    max_trials : int
        Maximum number of configurations to try

    Returns
    -------
    dict
        Search results
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"TFT ARCHITECTURE SEARCH - {task.upper()}")
    logger.info("=" * 80)
    logger.info(f"\nMax trials: {max_trials}")
    logger.info(f"Sequence length: {sequence_length}")

    # Load and normalize data
    logger.info("\n" + "-" * 80)
    logger.info("LOADING DATA")
    logger.info("-" * 80)

    data = prepare_tft_data(task=task, sequence_length=sequence_length)
    data = normalize_sequences(data, task=task)

    # Define search space
    logger.info("\n" + "-" * 80)
    logger.info("SEARCH SPACE")
    logger.info("-" * 80)

    search_space = {
        'hidden_dim': [64, 128, 256],
        'num_heads': [2, 4, 8],
        'num_lstm_layers': [1, 2],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.0001, 0.0005, 0.001],
        'batch_size': [32, 64],
        'max_epochs': [100],
        'patience': [15]
    }

    for param, values in search_space.items():
        logger.info(f"  {param}: {values}")

    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    all_configs = [dict(zip(keys, v)) for v in product(*values)]

    total_configs = len(all_configs)
    logger.info(f"\nTotal possible configurations: {total_configs}")

    # Limit to max_trials
    if total_configs > max_trials:
        logger.info(f"Sampling {max_trials} random configurations...")
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(total_configs, max_trials, replace=False)
        configs_to_try = [all_configs[i] for i in indices]
    else:
        configs_to_try = all_configs

    logger.info(f"Will try {len(configs_to_try)} configurations")

    # Run search
    logger.info("\n" + "=" * 80)
    logger.info("STARTING ARCHITECTURE SEARCH")
    logger.info("=" * 80)

    results = []
    best_metric = float('inf') if task == 'regression' else -float('inf')
    best_config = None

    for i, config in enumerate(configs_to_try, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TRIAL {i}/{len(configs_to_try)}")
        logger.info(f"{'=' * 80}")

        try:
            result = train_single_config(config, data, task)
            results.append(result)

            # Update best
            current_metric = result['test_metric']
            is_better = (current_metric < best_metric if task == 'regression'
                         else current_metric > best_metric)

            if is_better:
                improvement = abs(current_metric - best_metric)
                best_metric = current_metric
                best_config = config
                logger.info(f"\n!! NEW BEST! {result['metric_name']}: {current_metric:.4f} (↑{improvement:.4f})")

        except Exception as e:
            logger.error(f"Trial {i} failed: {e}")
            continue

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ARCHITECTURE SEARCH COMPLETE")
    logger.info("=" * 80)

    logger.info(f"\nTrials completed: {len(results)}/{len(configs_to_try)}")
    logger.info(f"Best {results[0]['metric_name']}: {best_metric:.4f}")
    logger.info(f"\nBest configuration:")
    for k, v in best_config.items():
        logger.info(f"  {k}: {v}")

    return {
        'task': task,
        'results': results,
        'best_config': best_config,
        'best_metric': best_metric,
        'metric_name': results[0]['metric_name'],
        'search_space': search_space,
        'trials_completed': len(results)
    }


def save_search_results(search_results: dict):
    """Save architecture search results."""
    task = search_results['task']

    logger.info(f"\n{'=' * 80}")
    logger.info("SAVING RESULTS")
    logger.info(f"{'=' * 80}")

    # Create tables directory
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Summary table
    summary_rows = []
    for result in search_results['results']:
        row = {
            'test_metric': result['test_metric'],
            'metric_name': result['metric_name'],
            'val_loss': result['best_val_loss'],
            'epochs': result['epochs_trained'],
            'time_seconds': result['training_time']
        }
        row.update(result['config'])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Sort by metric
    if task == 'regression':
        summary_df = summary_df.sort_values('test_metric', ascending=True)
    else:
        summary_df = summary_df.sort_values('test_metric', ascending=False)

    # Save summary
    summary_path = TABLES_DIR / f'tft_architecture_search_{task}.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"--OK-- Saved summary: {summary_path}")

    # Save best config
    best_config_path = TABLES_DIR / f'tft_best_config_{task}.json'
    with open(best_config_path, 'w') as f:
        json.dump(search_results['best_config'], f, indent=2)
    logger.info(f"--OK-- Saved best config: {best_config_path}")

    # Print top 5
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TOP 5 CONFIGURATIONS")
    logger.info(f"{'=' * 80}\n")

    print(summary_df.head(5).to_string(index=False))


def run_architecture_search_both_tasks(sequence_length: int = 30,
                                       max_trials: int = 30):
    """Run architecture search for both tasks."""
    logger.info("\n" + "=" * 80)
    logger.info("TFT ARCHITECTURE SEARCH - BOTH TASKS")
    logger.info("=" * 80)

    all_results = {}

    # Regression
    logger.info("\n\n" + "=" * 80)
    logger.info("TASK 1: REGRESSION")
    logger.info("=" * 80)

    reg_results = architecture_search(
        task='regression',
        sequence_length=sequence_length,
        max_trials=max_trials
    )
    save_search_results(reg_results)
    all_results['regression'] = reg_results

    # Classification
    logger.info("\n\n" + "=" * 80)
    logger.info("TASK 2: CLASSIFICATION")
    logger.info("=" * 80)

    clf_results = architecture_search(
        task='classification',
        sequence_length=sequence_length,
        max_trials=max_trials
    )
    save_search_results(clf_results)
    all_results['classification'] = clf_results

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("--OK-- BOTH TASKS COMPLETE!")
    logger.info("=" * 80)

    logger.info(f"\nRegression best RMSE: {reg_results['best_metric']:.4f}°C")
    logger.info(f"Classification best AUC: {clf_results['best_metric']:.4f}")

    return all_results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(PROJECT_ROOT / 'tft_architecture_search.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 80)
    logger.info("TFT ARCHITECTURE SEARCH")
    logger.info("=" * 80)

    # Run architecture search
    # 30 trials should take ~2-3 hours total
    results = run_architecture_search_both_tasks(
        sequence_length=30,
        max_trials=30
    )

    logger.info("\n" + "=" * 80)
    logger.info("--OK-- ARCHITECTURE SEARCH COMPLETE!")
    logger.info("=" * 80)