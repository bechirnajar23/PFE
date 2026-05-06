"""
LSTM Long-Horizon Training Scaffold (3-7 days ahead)
=====================================================

⚠️ CURRENT STATUS: NOT TRAINABLE YET
This script REQUIRES at least 30 days of continuous HGW telemetry to produce
a meaningful model. With your current 7.5 days, label generation breaks down
(positive rate ≈ 89% which makes the model useless).

WHEN TO RUN:
  Wait until you've collected 30+ days of data (preferably 60+ for 7-day horizon).
  Then run this script as-is. The pipeline will detect the data sufficiency,
  train the LSTM, and integrate with the multi-horizon predictor automatically.

DATA REQUIREMENTS:
  - 3-day horizon  →  minimum 21 days of data (7x horizon for stable labels)
  - 7-day horizon  →  minimum 49 days of data

ARCHITECTURE:
  - Lookback:    7 days at 1-hour granularity (168 timesteps)
  - Features:    16 (same as the synthetic-data Bi-LSTM)
  - Model:       Simple unidirectional LSTM (32 → 16 units)
  - Loss:        Focal loss (handles class imbalance)
  - Output:      Single sigmoid (probability of incident in horizon)

USAGE:
  python train_lstm_long_horizon.py --horizon-days 3
  python train_lstm_long_horizon.py --horizon-days 7
"""

import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')


# =============================================================================
# Pre-flight check
# =============================================================================
def check_data_sufficiency(df, horizon_days):
    """Verify there is enough continuous data to train a model at this horizon."""
    df = df.sort_values('timestamp').reset_index(drop=True)
    duration_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400

    min_required = horizon_days * 7
    if duration_days < min_required:
        return False, (
            f"Need at least {min_required:.0f} days of data for {horizon_days}-day horizon, "
            f"got only {duration_days:.1f} days. Continue collecting data."
        )

    # Check sampling regularity
    df['gap'] = df['timestamp'].diff().dt.total_seconds()
    big_gaps = df['gap'] > 86400  # > 1 day gap
    if big_gaps.sum() > horizon_days:
        return False, (
            f"Too many >1-day gaps in data ({big_gaps.sum()}). "
            f"Collect more continuous telemetry first."
        )

    # Check incident diversity
    is_urgent = (df['LOCAL_STATUS'] == 'URGENT').astype(int)
    horizon_samples = horizon_days * 24 * 60  # at 1-min granularity
    future = is_urgent.iloc[::-1].rolling(horizon_samples, min_periods=1).max().iloc[::-1]
    pos_rate = future.mean()

    if pos_rate > 0.85:
        return False, (
            f"Label saturated: {pos_rate:.1%} positives. "
            f"Need data with calmer periods to make {horizon_days}-day prediction useful."
        )
    if pos_rate < 0.02:
        return False, (
            f"Too few incidents ({pos_rate:.1%}). "
            f"Wait until more incidents accumulate."
        )

    return True, f"Data OK: {duration_days:.1f} days, {pos_rate:.1%} positive rate"


# =============================================================================
# LSTM model definition
# =============================================================================
def build_lstm(seq_len, n_features, units1=32, units2=16, dropout=0.3, lr=5e-4):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    import tensorflow.keras.backend as K

    def focal_loss(gamma=2.0, alpha=0.25):
        def loss(y_true, y_pred):
            y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
            pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_t = tf.where(K.equal(y_true, 1), alpha, 1 - alpha)
            return -K.mean(alpha_t * K.pow(1.0 - pt, gamma) * K.log(pt))
        return loss

    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(units1, return_sequences=True, dropout=dropout),
        LSTM(units2, return_sequences=False, dropout=dropout),
        Dense(16, activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=focal_loss(2.0, 0.25),
        metrics=[tf.keras.metrics.AUC(curve='PR', name='prauc'),
                 tf.keras.metrics.AUC(name='auc')],
    )
    return model


# =============================================================================
# Sequence building
# =============================================================================
def build_sequences(df, features, label, lookback_hours, horizon_min, stride_h=1):
    """Build (X, y) sequences. Hourly granularity for long horizons."""
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Resample to 1-hour granularity for memory efficiency
    df_idx = df.set_index('timestamp')
    df_h = df_idx[features].resample('1h').mean().ffill().dropna()
    is_urgent = (df_idx['LOCAL_STATUS'] == 'URGENT').astype(int).resample('1h').max().fillna(0)
    df_h['is_urgent'] = is_urgent.reindex(df_h.index).fillna(0)

    # Generate label
    horizon_samples = horizon_min // 60  # in hourly samples
    future = df_h['is_urgent'].iloc[::-1].rolling(horizon_samples, min_periods=1).max().iloc[::-1]
    df_h[label] = future.shift(-1).fillna(0).astype(int)
    df_h.loc[df_h['is_urgent'] == 1, label] = 0

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_h[features].values).astype(np.float32)

    # Build sequences
    X_seq, y_seq = [], []
    for i in range(lookback_hours, len(df_h), stride_h):
        X_seq.append(X_scaled[i - lookback_hours:i])
        y_seq.append(df_h[label].iloc[i])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32), scaler


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon-days', type=int, choices=[3, 7], default=3,
                        help='Prediction horizon in days')
    parser.add_argument('--input-csv', default='data/real/real_hgw_preprocessed.csv')
    parser.add_argument('--output-dir', default='data/real/long_horizon')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if not Path(args.input_csv).exists():
        print(f'ERROR: {args.input_csv} not found')
        sys.exit(1)

    df = pd.read_csv(args.input_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Pre-flight check
    print('=' * 70)
    print(f'LSTM LONG-HORIZON TRAINING — {args.horizon_days} DAYS')
    print('=' * 70)
    ok, msg = check_data_sufficiency(df, args.horizon_days)
    print(f'\nPre-flight check: {msg}')
    if not ok:
        print('\nABORTING. The model would not be useful.')
        print('\nRECOMMENDATIONS:')
        print('  1. Continue collecting telemetry from your HGW')
        print('  2. Re-run this script when the data window grows')
        print('  3. In the meantime, rely on the short-horizon CatBoost models')
        print('     (15min, 30min, 1h, 6h) which work well on your current data')
        sys.exit(0)

    # Build sequences
    horizon_min = args.horizon_days * 24 * 60
    lookback_hours = args.horizon_days * 24  # match horizon for context

    features = [
        'cpu_load', 'mem_used_pct', 'ping_latency', 'packet_loss', 'wan_status',
        'cpu_slope_5min', 'ram_slope_5min', 'cpu_mean_5min', 'mem_mean_5min',
        'cpu_std_30min', 'mem_std_30min', 'health_score',
    ]
    label = f'incident_in_{args.horizon_days}d'

    print(f'\nBuilding sequences...')
    print(f'  Lookback: {lookback_hours} hours')
    print(f'  Horizon:  {args.horizon_days} days = {horizon_min} minutes')
    X, y, scaler = build_sequences(df, features, label, lookback_hours, horizon_min)
    print(f'  Sequences: {X.shape}, positives: {int(y.sum())} ({y.mean()*100:.1f}%)')

    if y.sum() < 30:
        print(f'\nABORTING: too few positive sequences ({int(y.sum())} < 30)')
        sys.exit(0)

    # Stratified train/test split
    from sklearn.model_selection import train_test_split
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=0.30, random_state=42, stratify=y
    )
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_te, y_te = X[idx_te], y[idx_te]
    print(f'  Train: {X_tr.shape}, pos: {int(y_tr.sum())}')
    print(f'  Test:  {X_te.shape}, pos: {int(y_te.sum())}')

    # Build & train
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    tf.keras.utils.set_random_seed(42)

    model = build_lstm(lookback_hours, len(features))
    model.summary()

    val_size = max(20, int(len(X_tr) * 0.15))
    X_train, X_val = X_tr[:-val_size], X_tr[-val_size:]
    y_train, y_val = y_tr[:-val_size], y_tr[-val_size:]

    model_path = out_dir / f'lstm_{args.horizon_days}d_real.keras'
    callbacks = [
        ModelCheckpoint(filepath=str(model_path),
                        monitor='val_prauc', mode='max',
                        save_best_only=True, verbose=0),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50, batch_size=32,
        callbacks=callbacks, verbose=2,
    )

    # Evaluate
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score, precision_recall_curve, confusion_matrix)
    y_prob = model.predict(X_te, batch_size=32, verbose=0).flatten()
    prec, rec, thresh = precision_recall_curve(y_te, y_prob)
    denom = prec + rec
    f1s = np.where(denom == 0, 0, 2 * prec * rec / np.maximum(denom, 1e-9))
    best_th = float(thresh[int(np.argmax(f1s[:-1]))]) if len(thresh) > 0 else 0.5
    y_pred = (y_prob >= best_th).astype(int)
    cm = confusion_matrix(y_te, y_pred)

    print(f'\n=== LSTM {args.horizon_days}-day on REAL DATA ===')
    print(f'  ROC-AUC:   {roc_auc_score(y_te, y_prob):.4f}')
    print(f'  PR-AUC:    {average_precision_score(y_te, y_prob):.4f}')
    print(f'  F1 (best): {f1_score(y_te, y_pred):.4f}')
    print(f'  Threshold: {best_th:.4f}')
    print(f'  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}')

    # Save metadata
    import joblib
    joblib.dump(scaler, out_dir / f'lstm_{args.horizon_days}d_scaler.pkl')

    metadata = {
        'model_type': 'LSTM (unidirectional)',
        'horizon_days': args.horizon_days,
        'horizon_min': horizon_min,
        'lookback_hours': lookback_hours,
        'features': features,
        'metrics': {
            'roc_auc': float(roc_auc_score(y_te, y_prob)),
            'pr_auc': float(average_precision_score(y_te, y_prob)),
            'f1': float(f1_score(y_te, y_pred)),
            'threshold': best_th,
            'tp': int(cm[1, 1]), 'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]), 'tn': int(cm[0, 0]),
        },
    }
    with open(out_dir / f'lstm_{args.horizon_days}d_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'\nSaved → {model_path}')
    print(f'Saved → {out_dir}/lstm_{args.horizon_days}d_metadata.json')
    print(f'Saved → {out_dir}/lstm_{args.horizon_days}d_scaler.pkl')


if __name__ == '__main__':
    main()
