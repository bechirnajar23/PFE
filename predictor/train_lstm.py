import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score
import joblib
import os

np.random.seed(42)
tf.random.set_seed(42)

# =========================
# CONFIG
# =========================

DATA_PATH = "/app/data/data_train/hgw_long_term.csv"
MODEL_PATH = "/app/predictor/long_horizon_dl/lstm_3day.keras"
SCALER_PATH = "/app/predictor/long_horizon_dl/lstm_scaler.pkl"

FEATURES = [
    'cpu_load', 'mem_used_pct', 'ping_latency',
    'packet_loss', 'wan_status',
    'cpu_mean_24h', 'ram_mean_24h',
    'cpu_std_24h', 'ram_std_24h',
    'cpu_slope_6h', 'ram_slope_6h',
    'wan_instability_6h', 'health_score'
]

TARGET = "incident_in_72h"
LOOKBACK = 24


# =========================
# DATA
# =========================

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values(["gateway_id", "timestamp"]).reset_index(drop=True)
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


# =========================
# PREPROCESS
# =========================

def build_sequences(df, scaler):
    X_scaled = scaler.transform(df[FEATURES]).astype(np.float32)
    y = df[TARGET].values

    X_seq, y_seq = [], []

    for i in range(LOOKBACK, len(df)):
        X_seq.append(X_scaled[i-LOOKBACK:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


# =========================
# MODEL
# =========================

def build_lstm():
    inp = Input(shape=(LOOKBACK, len(FEATURES)))

    x = LSTM(48, return_sequences=True, dropout=0.3)(inp)
    x = LSTM(24, return_sequences=False, dropout=0.3)(x)

    x = BatchNormalization()(x)

    x = Dense(24, activation='relu')(x)
    x = Dropout(0.4)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(curve="PR", name="prauc")
        ]
    )

    return model


# =========================
# TRAIN
# =========================

def train():

    print("📊 Loading data...")
    df = load_data()

    n = len(df)
    train_df = df.iloc[:int(n*0.7)]
    val_df   = df.iloc[int(n*0.7):int(n*0.85)]
    test_df  = df.iloc[int(n*0.85):]

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURES])

    X_tr, y_tr = build_sequences(train_df, scaler)
    X_va, y_va = build_sequences(val_df, scaler)
    X_te, y_te = build_sequences(test_df, scaler)

    print("📈 Building model...")
    model = build_lstm()

    es = EarlyStopping(
        monitor="val_prauc",
        patience=3,
        restore_best_weights=True
    )

    lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5
    )

    print("🚀 Training...")
    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=15,
        batch_size=128,
        callbacks=[es, lr],
        verbose=1
    )

    print("📊 Evaluating...")
    y_prob = model.predict(X_te).flatten()

    precision, recall, thresholds = precision_recall_curve(y_te, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)

    best_idx = f1_scores[:-1].argmax()
    best_threshold = thresholds[best_idx]

    print(f"Best threshold: {best_threshold:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("✅ Model saved")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    train()