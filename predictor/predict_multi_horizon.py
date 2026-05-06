# ============================================
# MULTI-HORIZON + LSTM (PRO VERSION CLEAN)
# ============================================

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# =========================
# ML / DL IMPORTS
# =========================
import tensorflow as tf
import joblib

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, precision_recall_curve, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from catboost import CatBoostClassifier

# =========================
# PATHS
# =========================
INPUT_CSV = "/app/data/data_train/hgw_short_term.csv"
OUT_DIR = Path("/app/data/real/multi_horizon")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LSTM_MODEL_PATH = "/app/predictor/long_horizon_dl/lstm_3day.keras"
LSTM_SCALER_PATH = "/app/predictor/long_horizon_dl/lstm_scaler.pkl"

# =========================
# FEATURES
# =========================

# 🔵 LSTM (long-term)
LSTM_FEATURES = [
    'cpu_load', 'mem_used_pct', 'ping_latency',
    'packet_loss', 'wan_status',
    'cpu_mean_24h', 'ram_mean_24h',
    'cpu_std_24h', 'ram_std_24h',
    'cpu_slope_6h', 'ram_slope_6h',
    'wan_instability_6h', 'health_score'
]

# 🟢 CatBoost (short-term)
CB_FEATURES = [
    'cpu_load', 'mem_used_pct', 'ping_latency', 'packet_loss',
    'wan_status', 'reboot_event', 'recovery_phase',
    'cwmp_rss_mb', 'dhcp_rss_mb', 'nemo_rss_mb',
    'cpu_slope_5min', 'cpu_slope_30min', 'ram_slope_5min', 'ram_slope_30min',
    'cpu_mean_5min', 'cpu_mean_30min', 'cpu_std_30min', 'cpu_max_30min',
    'mem_mean_5min', 'mem_mean_30min', 'mem_std_30min', 'mem_max_30min',
    'ping_mean_5min', 'ping_mean_30min', 'ping_max_5min', 'loss_mean_5min',
    'wan_instability_5min',
    'cpu_lag1m', 'cpu_lag3m', 'cpu_lag5m', 'cpu_lag10m', 'cpu_lag15m',
    'mem_lag1m', 'mem_lag3m', 'mem_lag5m', 'mem_lag10m', 'mem_lag15m',
    'sin_hour', 'cos_hour', 'cpu_x_mem', 'saturation_idx', 'mem_headroom', 'health_score',
]

LOOKBACK = 24
HORIZONS_MIN = [15, 30, 60, 360]

# =========================
# LOAD LSTM
# =========================
print("🔵 Loading LSTM...")

lstm_model = None
lstm_scaler = None

if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_SCALER_PATH):
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    lstm_scaler = joblib.load(LSTM_SCALER_PATH)
    print("✅ LSTM loaded")
else:
    print("⚠️ LSTM model not found — using fallback")
# =========================
# LSTM PREDICTION
# =========================
def predict_lstm(df_recent):
    if lstm_model is None or lstm_scaler is None:
        return 0.0

    if len(df_recent) < LOOKBACK:
        return 0.0

    X = df_recent[LSTM_FEATURES].values[-LOOKBACK:]
    X = lstm_scaler.transform(X)
    X = np.expand_dims(X, axis=0)

    return float(lstm_model.predict(X, verbose=0)[0][0])

# =========================
# LABEL GENERATION
# =========================
def add_horizon_label(df, horizon_min):
    label_col = f'incident_in_{horizon_min}min'
    out = []

    for sid, group in df.groupby('session_id'):
        g = group.sort_values('timestamp').copy().reset_index(drop=True)

        is_urgent = (g['LOCAL_STATUS'] == 'URGENT').astype(int)
        future = is_urgent.iloc[::-1].rolling(horizon_min, min_periods=1).max().iloc[::-1]

        g[label_col] = future.shift(-1).fillna(0).astype(int)
        g.loc[is_urgent == 1, label_col] = 0

        out.append(g)

    return pd.concat(out, ignore_index=True)

# =========================
# THRESHOLDS
# =========================
def find_thresholds(y_true, y_prob):
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)

    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    idx = np.argmax(f1[:-1])

    return float(thresh[idx]) if len(thresh) > 0 else 0.5

# =========================
# TRAIN CATBOOST
# =========================
def train_one_horizon(df, horizon):

    label = f'incident_in_{horizon}min'

    if label not in df.columns:
        df = add_horizon_label(df, horizon)

    df_ml = df.dropna(subset=CB_FEATURES + [label]).copy()

    X = df_ml[CB_FEATURES].values
    y = df_ml[label].values

    if y.sum() < 30:
        print(f"⚠️ Skipping {horizon}min (not enough positives)")
        return None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function='Logloss',
        eval_metric='PRAUC',
        early_stopping_rounds=50,
        verbose=0
    )

    model.fit(X_tr, y_tr, eval_set=(X_te, y_te))

    y_prob = model.predict_proba(X_te)[:, 1]

    threshold = find_thresholds(y_te, y_prob)

    print(f"✅ {horizon}min | PR-AUC: {average_precision_score(y_te, y_prob):.4f}")

    model.save_model(str(OUT_DIR / f"catboost_{horizon}min.cbm"))

    return threshold

# =========================
# MAIN
# =========================
def main():

    print("📊 Loading data...")
    df = pd.read_csv(INPUT_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    thresholds = {}

    for h in HORIZONS_MIN:
        th = train_one_horizon(df, h)
        if th:
            thresholds[h] = th

    print("\n🚀 ALL MODELS TRAINED")
    print(thresholds)

def predict_all_horizons(df_recent):
    """
    Retourne toutes les prédictions multi-horizon
    """

    predictions = {}

    # 🔵 LSTM (long terme)
    predictions["3d"] = predict_lstm(df_recent)

    # 🟢 CatBoost (court terme)
    # (si tes modèles sont chargés ailleurs)
    try:
        for h in [15, 30, 60, 360]:
            if f"model_{h}" in globals():
                X = df_recent[CB_FEATURES].iloc[-1:].values
                prob = globals()[f"model_{h}"].predict_proba(X)[0][1]
                predictions[f"{h}min"] = float(prob)
            else:
                predictions[f"{h}min"] = 0.0
    except:
        predictions.update({
            "15min": 0.0,
            "30min": 0.0,
            "60min": 0.0,
            "360min": 0.0
        })

    return predictions


# =========================
if __name__ == "__main__":
    main()