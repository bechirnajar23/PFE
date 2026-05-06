"""
Universal Bi-LSTM 3-Day Loader
===============================

This module rebuilds the Bi-LSTM architecture from scratch and loads
pre-trained weights. Works with ANY Keras version (2.x, 3.x).

Usage:
    from bilstm_loader import load_bilstm_3d
    
    model, scaler = load_bilstm_3d(
        weights_path="bilstm_3d.weights.h5",
        scaler_path="transfer_scaler.pkl"
    )
    
    # Predict
    X_seq = ...  # (N, 24, 13) array
    y_prob = model.predict(X_seq)
"""

import json
import numpy as np
import joblib


def build_bilstm_architecture():
    """
    Manually rebuild the exact Bi-LSTM architecture.
    This avoids any Keras version-specific serialization issues.
    """
    try:
        from tensorflow import keras
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
    except ImportError:
        from keras.models import Model
        from keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
    
    # Architecture hardcoded to match trained model
    inp = Input(shape=(24, 13), name='input')
    x = Bidirectional(LSTM(48, return_sequences=True, dropout=0.3),
                       name='bilstm_1')(inp)
    x = Bidirectional(LSTM(24, return_sequences=False, dropout=0.3),
                       name='bilstm_2')(x)
    x = Dense(32, activation='relu', name='dense_1')(x)
    x = Dropout(0.3, name='dropout_1')(x)
    x = Dense(16, activation='relu', name='dense_2')(x)
    x = Dropout(0.2, name='dropout_2')(x)
    out = Dense(1, activation='sigmoid', name='prediction')(x)
    
    model = Model(inputs=inp, outputs=out, name='bilstm_3day')
    return model


def load_bilstm_3d(weights_path, scaler_path, metadata_path=None):
    """
    Load the Bi-LSTM 3-day model.
    
    Args:
        weights_path: Path to .weights.h5 file
        scaler_path: Path to transfer_scaler.pkl
        metadata_path: Optional path to bilstm_3d_metadata.json
        
    Returns:
        (model, scaler) or (model, scaler, metadata) if metadata_path provided
    """
    # Build architecture from scratch
    model = build_bilstm_architecture()
    
    # Load weights
    model.load_weights(weights_path)
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    if metadata_path:
        with open(metadata_path) as f:
            metadata = json.load(f)
        return model, scaler, metadata
    
    return model, scaler


# Quick test when run directly
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python bilstm_loader.py <weights.h5> <scaler.pkl>")
        sys.exit(1)
    
    model, scaler = load_bilstm_3d(sys.argv[1], sys.argv[2])
    print(f"✓ Model loaded: {model.input_shape} → {model.output_shape}")
    print(f"✓ Total params: {model.count_params():,}")
    print(f"✓ Scaler loaded: {scaler.n_features_in_} features")
