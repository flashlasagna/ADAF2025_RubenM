"""
Training pipeline for Temporal Fusion Transformer (TFT).
Handles training, validation, and model saving.
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Import TFT components
import sys

sys.path.insert(0, str(PROJECT_ROOT))
from sequence_data import prepare_tft_data
from tft_model import build_tft_model

from src.utils.config import MODELS_DIR, TABLES_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def create_tf_dataset(sequences: dict, batch_size: int = 32, shuffle: bool = True):
    """
    Create TensorFlow dataset from sequences.

    Parameters
    ----------
    sequences : dict
        Dictionary with 'known', 'unknown', and 'targets'
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle data

    Returns
    -------
    tf.data.Dataset
        TensorFlow dataset
    """
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


def train_tft(task='regression',
              sequence_length=30,
              hidden_dim=128,
              num_heads=4,
              num_lstm_layers=1,
              dropout_rate=0.1,
              batch_size=32,
              epochs=100,
              patience=15,
              learning_rate=0.001):
    """
    Train TFT model.

    Parameters
    ----------
    task : str
        'regression' or 'classification'
    sequence_length : int
        Sequence length
    hidden_dim : int
        Hidden dimension
    num_heads : int
        Number of attention heads
    num_lstm_layers : int
        Number of LSTM layers
    dropout_rate : float
        Dropout rate
    batch_size : int
        Batch size
    epochs : int
        Maximum epochs
    patience : int
        Early stopping patience
    learning_rate : float
        Learning rate

    Returns
    -------
    model, history, training_time
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"TRAINING TFT - {task.upper()}")
    logger.info("=" * 80)

    logger.info(f"\nHyperparameters:")
    logger.info(f"  Sequence length: {sequence_length}")
    logger.info(f"  Hidden dim: {hidden_dim}")
    logger.info(f"  Attention heads: {num_heads}")
    logger.info(f"  LSTM layers: {num_lstm_layers}")
    logger.info(f"  Dropout: {dropout_rate}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Max epochs: {epochs}")
    logger.info(f"  Early stopping patience: {patience}")

    # Prepare data
    logger.info("\n" + "-" * 80)
    logger.info("LOADING DATA")
    logger.info("-" * 80)
    data = prepare_tft_data(task=task, sequence_length=sequence_length)

    # Create TF datasets
    logger.info("\nCreating TensorFlow datasets...")
    train_dataset = create_tf_dataset(data['train'], batch_size=batch_size, shuffle=True)
    val_dataset = create_tf_dataset(data['val'], batch_size=batch_size, shuffle=False)
    test_dataset = create_tf_dataset(data['test'], batch_size=batch_size, shuffle=False)

    logger.info(f"  Train batches: {len(list(train_dataset))}")
    logger.info(f"  Val batches: {len(list(val_dataset))}")
    logger.info(f"  Test batches: {len(list(test_dataset))}")

    # Recreate datasets (consumed by len())
    train_dataset = create_tf_dataset(data['train'], batch_size=batch_size, shuffle=True)
    val_dataset = create_tf_dataset(data['val'], batch_size=batch_size, shuffle=False)

    # Build model
    logger.info("\n" + "-" * 80)
    logger.info("BUILDING MODEL")
    logger.info("-" * 80)

    model = build_tft_model(
        feature_info=data['feature_info'],
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_lstm_layers=num_lstm_layers,
        dropout_rate=dropout_rate,
        task=task
    )

    # Re-compile with specified learning rate
    if task == 'regression':
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train
    logger.info("\n" + "-" * 80)
    logger.info("TRAINING")
    logger.info("-" * 80)

    start_time = time.time()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time

    logger.info(f"\n✓ Training complete in {training_time / 60:.2f} minutes")
    logger.info(f"  Best epoch: {len(history.history['loss']) - patience}")
    logger.info(f"  Best val_loss: {min(history.history['val_loss']):.4f}")

    # Evaluate on test set
    logger.info("\n" + "-" * 80)
    logger.info("TEST SET EVALUATION")
    logger.info("-" * 80)

    test_dataset = create_tf_dataset(data['test'], batch_size=batch_size, shuffle=False)
    test_results = model.evaluate(test_dataset, verbose=1)

    logger.info(f"\nTest results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        logger.info(f"  {metric_name}: {value:.4f}")

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f'tft_{task}.h5'
    model.save(model_path)
    logger.info(f"\n✓ Model saved: {model_path}")

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_path = TABLES_DIR / f'tft_{task}_training_history.csv'
    history_df.to_csv(history_path, index=False)
    logger.info(f"✓ Training history saved: {history_path}")

    # Save hyperparameters
    hyperparams = {
        'task': task,
        'sequence_length': sequence_length,
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'num_lstm_layers': num_lstm_layers,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs_trained': len(history.history['loss']),
        'training_time_minutes': training_time / 60,
        'best_val_loss': float(min(history.history['val_loss'])),
        'test_loss': float(test_results[0])
    }

    hyperparams_path = TABLES_DIR / f'tft_{task}_hyperparameters.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    logger.info(f"✓ Hyperparameters saved: {hyperparams_path}")

    return model, history, training_time


def train_both_tasks(sequence_length=30,
                     hidden_dim=128,
                     num_heads=4,
                     num_lstm_layers=1,
                     dropout_rate=0.1,
                     batch_size=32,
                     epochs=100,
                     patience=15):
    """
    Train TFT for both regression and classification tasks.

    Parameters
    ----------
    sequence_length : int
        Sequence length
    hidden_dim : int
        Hidden dimension
    num_heads : int
        Number of attention heads
    num_lstm_layers : int
        Number of LSTM layers
    dropout_rate : float
        Dropout rate
    batch_size : int
        Batch size
    epochs : int
        Maximum epochs
    patience : int
        Early stopping patience

    Returns
    -------
    dict
        Results for both tasks
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING TFT FOR BOTH TASKS")
    logger.info("=" * 80)

    results = {}

    # Train regression
    logger.info("\n\n" + "=" * 80)
    logger.info("TASK 1: REGRESSION (Temperature Prediction)")
    logger.info("=" * 80)

    reg_model, reg_history, reg_time = train_tft(
        task='regression',
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_lstm_layers=num_lstm_layers,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience
    )

    results['regression'] = {
        'model': reg_model,
        'history': reg_history,
        'training_time': reg_time
    }

    # Train classification
    logger.info("\n\n" + "=" * 80)
    logger.info("TASK 2: CLASSIFICATION (Rain Prediction)")
    logger.info("=" * 80)

    clf_model, clf_history, clf_time = train_tft(
        task='classification',
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_lstm_layers=num_lstm_layers,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience
    )

    results['classification'] = {
        'model': clf_model,
        'history': clf_history,
        'training_time': clf_time
    }

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("--OK-- BOTH TASKS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nRegression:")
    logger.info(f"  Training time: {reg_time / 60:.2f} minutes")
    logger.info(f"  Best val loss: {min(reg_history.history['val_loss']):.4f}")

    logger.info(f"\nClassification:")
    logger.info(f"  Training time: {clf_time / 60:.2f} minutes")
    logger.info(f"  Best val loss: {min(clf_history.history['val_loss']):.4f}")

    logger.info(f"\nTotal time: {(reg_time + clf_time) / 60:.2f} minutes")

    return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(PROJECT_ROOT / 'tft_training.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 80)
    logger.info("TEMPORAL FUSION TRANSFORMER (TFT) TRAINING")
    logger.info("=" * 80)

    # Train both tasks with default hyperparameters
    results = train_both_tasks(
        sequence_length=30,
        hidden_dim=128,
        num_heads=4,
        num_lstm_layers=1,
        dropout_rate=0.1,
        batch_size=32,
        epochs=100,
        patience=15
    )

    logger.info("\n" + "=" * 80)
    logger.info("--OK-- TFT TRAINING PIPELINE COMPLETE!")
    logger.info("=" * 80)