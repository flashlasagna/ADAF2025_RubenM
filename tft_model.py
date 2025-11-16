"""
Temporal Fusion Transformer (TFT) implementation in Keras.
Based on: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
(Lim et al., 2021)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ============================================================================
# TFT COMPONENTS
# ============================================================================

class GatedResidualNetwork(layers.Layer):
    """
    Gated Residual Network (GRN) - Core building block of TFT.
    Applies non-linear processing with gating and skip connections.
    """

    def __init__(self, hidden_dim, output_dim=None, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.dropout_rate = dropout_rate

        # Layers
        self.dense1 = layers.Dense(hidden_dim, activation='elu')
        self.dense2 = layers.Dense(self.output_dim)
        self.dropout = layers.Dropout(dropout_rate)
        self.gate = layers.Dense(self.output_dim, activation='sigmoid')
        self.layer_norm = layers.LayerNormalization()

        # Skip connection (if dimensions differ)
        self.skip_layer = None

    def build(self, input_shape):
        # Add skip connection if needed
        if input_shape[-1] != self.output_dim:
            self.skip_layer = layers.Dense(self.output_dim, use_bias=False)
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Non-linear processing
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)

        # Gating
        gate = self.gate(inputs)
        x = x * gate

        # Skip connection
        if self.skip_layer is not None:
            skip = self.skip_layer(inputs)
        else:
            skip = inputs

        # Add and normalize
        output = self.layer_norm(x + skip)

        return output


class VariableSelectionNetwork(layers.Layer):
    """
    Variable Selection Network (VSN) - Selects relevant features.
    Uses attention mechanism to weight input variables.
    """

    def __init__(self, hidden_dim, num_inputs, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_inputs = num_inputs
        self.dropout_rate = dropout_rate

        # GRN for each variable
        self.variable_grns = [
            GatedResidualNetwork(hidden_dim, hidden_dim, dropout_rate)
            for _ in range(num_inputs)
        ]

        # GRN for variable selection weights
        self.weight_grn = GatedResidualNetwork(hidden_dim, num_inputs, dropout_rate)

        # Softmax for attention weights
        self.softmax = layers.Softmax()

    def call(self, inputs, training=None):
        # inputs shape: (batch, time, num_inputs, feature_dim)
        # or (batch, num_inputs, feature_dim) for static

        # Process each variable through its GRN
        processed_vars = []
        for i, grn in enumerate(self.variable_grns):
            var = inputs[..., i, :]  # Extract variable i
            processed = grn(var, training=training)
            processed_vars.append(processed)

        # Stack processed variables
        processed = tf.stack(processed_vars, axis=-2)  # (batch, [...], num_inputs, hidden_dim)

        # Flatten for attention computation
        flattened = tf.concat(processed_vars, axis=-1)  # (batch, [...], num_inputs * hidden_dim)

        # Compute variable selection weights
        weights = self.weight_grn(flattened, training=training)  # (batch, [...], num_inputs)
        weights = self.softmax(weights)

        # Apply weights
        weights_expanded = tf.expand_dims(weights, axis=-1)  # (batch, [...], num_inputs, 1)
        selected = processed * weights_expanded  # (batch, [...], num_inputs, hidden_dim)

        # Sum across variables
        output = tf.reduce_sum(selected, axis=-2)  # (batch, [...], hidden_dim)

        return output, weights


class InterpretableMultiHeadAttention(layers.Layer):
    """
    Interpretable Multi-Head Attention - Modified for interpretability.
    Uses shared value weights across heads for clearer interpretation.
    """

    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // num_heads

        # Query, Key, Value projections
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]

        # Linear projections
        q = self.wq(inputs)  # (batch, seq_len, d_model)
        k = self.wk(inputs)
        v = self.wv(inputs)

        # Split heads
        q = self.split_heads(q, batch_size)  # (batch, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Scale
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        # Attention output
        attention_output = tf.matmul(attention_weights, v)

        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model))

        # Final linear projection
        output = self.dense(concat_attention)

        return output, attention_weights


# ============================================================================
# COMPLETE TFT MODEL
# ============================================================================

class TemporalFusionTransformer(keras.Model):
    """
    Complete Temporal Fusion Transformer model.

    Architecture:
    1. Variable Selection Networks (for known and unknown inputs)
    2. LSTM Encoder-Decoder (for temporal processing)
    3. Multi-Head Attention (for long-term dependencies)
    4. Gated Residual Networks (throughout)
    5. Quantile outputs (for uncertainty estimation)
    """

    def __init__(self,
                 n_known_features,
                 n_unknown_features,
                 hidden_dim=128,
                 num_heads=4,
                 num_lstm_layers=1,
                 dropout_rate=0.1,
                 output_quantiles=[0.1, 0.5, 0.9],
                 task='regression',
                 **kwargs):
        """
        Initialize TFT model.

        Parameters
        ----------
        n_known_features : int
            Number of known future input features
        n_unknown_features : int
            Number of unknown input features
        hidden_dim : int
            Hidden dimension size
        num_heads : int
            Number of attention heads
        num_lstm_layers : int
            Number of LSTM layers
        dropout_rate : float
            Dropout rate
        output_quantiles : list
            Quantiles for prediction intervals
        task : str
            'regression' or 'classification'
        """
        super().__init__(**kwargs)

        self.n_known_features = n_known_features
        self.n_unknown_features = n_unknown_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout_rate
        self.output_quantiles = output_quantiles
        self.task = task

        logger.info(f"\nInitializing TFT model:")
        logger.info(f"  Known features: {n_known_features}")
        logger.info(f"  Unknown features: {n_unknown_features}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Attention heads: {num_heads}")
        logger.info(f"  LSTM layers: {num_lstm_layers}")
        logger.info(f"  Task: {task}")

        # Embedding layers for input features
        self.known_embedding = layers.Dense(hidden_dim)
        self.unknown_embedding = layers.Dense(hidden_dim)

        # Variable Selection Networks
        # Note: Simplified - assuming already embedded features
        # In full TFT, each feature would be embedded separately

        # LSTM Encoder (processes historical sequence)
        self.lstm_encoder = keras.Sequential([
            layers.LSTM(hidden_dim, return_sequences=True, dropout=dropout_rate)
            for _ in range(num_lstm_layers)
        ])

        # Gated Residual Network after LSTM
        self.post_lstm_grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout_rate)

        # Multi-Head Attention
        self.attention = InterpretableMultiHeadAttention(hidden_dim, num_heads, dropout_rate)

        # Post-attention GRN
        self.post_attention_grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout_rate)

        # Output layers
        if task == 'regression':
            # Quantile outputs for uncertainty
            self.output_layers = [
                layers.Dense(1, name=f'quantile_{q}')
                for q in output_quantiles
            ]
        else:
            # Binary classification
            self.output_layer = layers.Dense(1, activation='sigmoid', name='classification_output')

    def call(self, inputs, training=None):
        """
        Forward pass.

        Parameters
        ----------
        inputs : dict
            Dictionary with 'known' and 'unknown' keys
        training : bool
            Training mode flag

        Returns
        -------
        outputs
            Model predictions
        """
        known_inputs = inputs['known']  # (batch, seq_len, n_known)
        unknown_inputs = inputs['unknown']  # (batch, seq_len, n_unknown)

        # Embed inputs
        known_embedded = self.known_embedding(known_inputs)  # (batch, seq_len, hidden_dim)
        unknown_embedded = self.unknown_embedding(unknown_inputs)  # (batch, seq_len, hidden_dim)

        # Combine known and unknown inputs
        combined = known_embedded + unknown_embedded  # Simple addition for now

        # LSTM encoding
        lstm_output = self.lstm_encoder(combined, training=training)
        lstm_output = self.post_lstm_grn(lstm_output, training=training)

        # Self-attention over sequence
        attention_output, attention_weights = self.attention(lstm_output, training=training)
        attention_output = self.post_attention_grn(attention_output, training=training)

        # Take last time step for prediction
        final_hidden = attention_output[:, -1, :]  # (batch, hidden_dim)

        # Generate outputs
        if self.task == 'regression':
            # Multiple quantile predictions
            outputs = [layer(final_hidden) for layer in self.output_layers]
            outputs = tf.concat(outputs, axis=-1)  # (batch, num_quantiles)
        else:
            # Classification
            outputs = self.output_layer(final_hidden)  # (batch, 1)

        return outputs

    def get_config(self):
        return {
            'n_known_features': self.n_known_features,
            'n_unknown_features': self.n_unknown_features,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_lstm_layers': self.num_lstm_layers,
            'dropout_rate': self.dropout_rate,
            'output_quantiles': self.output_quantiles,
            'task': self.task
        }


def build_tft_model(feature_info: dict,
                    hidden_dim: int = 128,
                    num_heads: int = 4,
                    num_lstm_layers: int = 1,
                    dropout_rate: float = 0.1,
                    task: str = 'regression') -> TemporalFusionTransformer:
    """
    Build TFT model from feature information.

    Parameters
    ----------
    feature_info : dict
        Dictionary with feature counts
    hidden_dim : int
        Hidden dimension
    num_heads : int
        Number of attention heads
    num_lstm_layers : int
        Number of LSTM layers
    dropout_rate : float
        Dropout rate
    task : str
        'regression' or 'classification'

    Returns
    -------
    TemporalFusionTransformer
        Compiled TFT model
    """
    model = TemporalFusionTransformer(
        n_known_features=feature_info['n_known'],
        n_unknown_features=feature_info['n_unknown'],
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_lstm_layers=num_lstm_layers,
        dropout_rate=dropout_rate,
        task=task
    )

    # Compile model
    if task == 'regression':
        # Quantile loss (we'll use median prediction for main metric)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Simplified - full TFT uses quantile loss
            metrics=['mae']
        )
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

    logger.info("\n--OK-- TFT model built and compiled")

    return model


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("Testing TFT model architecture...")

    # Test model creation
    feature_info = {
        'n_known': 11,  # Example: temporal features
        'n_unknown': 162,  # Example: observed features
        'sequence_length': 30
    }

    model = build_tft_model(feature_info, hidden_dim=64, task='regression')

    # Test forward pass
    batch_size = 8
    seq_len = 30

    dummy_input = {
        'known': tf.random.normal((batch_size, seq_len, feature_info['n_known'])),
        'unknown': tf.random.normal((batch_size, seq_len, feature_info['n_unknown']))
    }

    output = model(dummy_input, training=False)
    logger.info(f"\nTest output shape: {output.shape}")

    model.summary()

    logger.info("\n--OK-- TFT model architecture test complete!")