"""
03c_retrain_bilstm_safe.py — Re-train Bi-LSTM with checkpointing
==================================================================
Skips Optuna (we already know best params from previous run) and trains
the final model with ModelCheckpoint so the model is saved after EVERY
epoch — no risk of losing it to a post-training crash.

Best params from previous run (PR-AUC val 0.9511):
  - Use them directly via --units1 / --units2 / --dropout / --lr
  - Default values below are the ones Optuna selected

Usage:
    python 03c_retrain_bilstm_safe.py --epochs 12 --batch-size 128

If training is interrupted, the best-so-far model is in
data/bilstm_72h_checkpoint.keras and you can resume or evaluate it.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument("--data",       default="data/hgw_long_term.csv")
parser.add_argument("--out-dir",    default="data")
parser.add_argument("--horizon",    default="72h", choices=["72h", "7d"])
parser.add_argument("--lookback",   type=int, default=21)
parser.add_argument("--lookback-hours-step", type=int, default=2)
parser.add_argument("--train-stride", type=int, default=4)
parser.add_argument("--test-stride",  type=int, default=2)
parser.add_argument("--epochs",     type=int, default=12,
                     help="Default 12 — we know it converges around epoch 8")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--units1",     type=int, default=48,
                     help="Default from Optuna best trial")
parser.add_argument("--units2",     type=int, default=24)
parser.add_argument("--dropout",    type=float, default=0.3)
parser.add_argument("--lr",         type=float, default=6.15e-04)
args = parser.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
LABEL_COL = f"incident_in_{args.horizon}"


# =============================================================
# 1. LOAD + BUILD SEQUENCES
# =============================================================
print("=" * 70)
print(f"Bi-LSTM Re-training (SAFE) — {args.horizon}")
print("=" * 70)

df = pd.read_csv(args.data, parse_dates=["timestamp"])
df = df.sort_values(["gateway_id", "timestamp"]).reset_index(drop=True)
SPH = 2

INPUT_FEATURES = [
    "cpu_load", "mem_used_pct", "ping_latency", "packet_loss",
    "cwmp_rss_mb", "dhcp_rss_mb", "nemo_rss_mb",
    "wan_status", "cpu_slope_6h", "ram_slope_6h",
    "cpu_mean_24h", "ram_mean_24h", "cpu_std_24h", "ram_std_24h",
    "wan_instability_6h", "health_score",
]
LOOKBACK_STEPS = args.lookback * 24 * SPH
SUBSAMPLE      = args.lookback_hours_step * SPH
SEQ_LEN        = LOOKBACK_STEPS // SUBSAMPLE
print(f"Lookback: {args.lookback}d → {SEQ_LEN} timesteps × {len(INPUT_FEATURES)} feats")

print("\nBuilding sequences...")
from sklearn.preprocessing import StandardScaler
t0 = time.time()

X_tr_all, y_tr_all = [], []
X_te_all, y_te_all = [], []
for gw, group in df.groupby("gateway_id"):
    group = group.sort_values("timestamp").reset_index(drop=True)
    sp = int(len(group) * 0.75)
    while sp > len(group) * 0.50:
        if group.iloc[sp:][LABEL_COL].sum() >= 50:
            break
        sp = int(sp * 0.95)

    scaler = StandardScaler()
    X_tr_raw = group.iloc[:sp][INPUT_FEATURES].fillna(0).values
    X_te_raw = group.iloc[sp:][INPUT_FEATURES].fillna(0).values
    scaler.fit(X_tr_raw)
    X_full = np.vstack([scaler.transform(X_tr_raw),
                          scaler.transform(X_te_raw)]).astype(np.float32)
    labels = group[LABEL_COL].values

    for i in range(LOOKBACK_STEPS, sp, args.train_stride):
        X_tr_all.append(X_full[i-LOOKBACK_STEPS:i:SUBSAMPLE])
        y_tr_all.append(labels[i])
    pos = np.where(labels[LOOKBACK_STEPS:sp] == 1)[0] + LOOKBACK_STEPS
    for i in pos:
        if i % args.train_stride != 0:
            X_tr_all.append(X_full[i-LOOKBACK_STEPS:i:SUBSAMPLE])
            y_tr_all.append(labels[i])

    for i in range(sp, len(group), args.test_stride):
        if i - LOOKBACK_STEPS < 0:
            continue
        X_te_all.append(X_full[i-LOOKBACK_STEPS:i:SUBSAMPLE])
        y_te_all.append(labels[i])

X_tr = np.asarray(X_tr_all, dtype=np.float32)
y_tr = np.asarray(y_tr_all, dtype=np.float32)
X_te = np.asarray(X_te_all, dtype=np.float32)
y_te = np.asarray(y_te_all, dtype=np.float32)
print(f"  Built in {time.time()-t0:.0f}s")
print(f"  X_tr: {X_tr.shape}  pos: {int(y_tr.sum())} ({y_tr.mean()*100:.1f}%)")
print(f"  X_te: {X_te.shape}  pos: {int(y_te.sum())} ({y_te.mean()*100:.1f}%)")

# Balanced 3:1 undersampling
pos_idx = np.where(y_tr == 1)[0]
neg_idx = np.where(y_tr == 0)[0]
n_pos = len(pos_idx)
neg_kept = np.random.choice(neg_idx, min(n_pos*3, len(neg_idx)), replace=False)
keep = np.sort(np.concatenate([pos_idx, neg_kept]))
X_tr_bal = X_tr[keep]
y_tr_bal = y_tr[keep]
print(f"  Balanced: {X_tr_bal.shape}  pos rate: {y_tr_bal.mean()*100:.1f}%")
del X_tr, y_tr  # free memory


# =============================================================
# 2. MODEL ARCHITECTURE
# =============================================================
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Bidirectional, Dense,
                                       Dropout, Layer)
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                          ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

print(f"\nTensorFlow {tf.__version__}")

# Get serializable decorator
try:
    register_keras = tf.keras.saving.register_keras_serializable
except AttributeError:
    try:
        from keras.saving import register_keras_serializable as register_keras
    except ImportError:
        def register_keras(*a, **k):
            def deco(cls): return cls
            return deco


@register_keras()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="W", shape=(input_shape[-1], 1),
                                   initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="b", shape=(input_shape[1], 1),
                                   initializer="zeros", trainable=True)
        super().build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        context = K.sum(x * a, axis=1)
        return [context, K.squeeze(a, -1)]
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[-1]),
                (input_shape[0], input_shape[1])]
    def get_config(self):
        return super().get_config()


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(K.equal(y_true, 1), alpha, 1 - alpha)
        return -K.mean(alpha_t * K.pow(1.0 - pt, gamma) * K.log(pt))
    return loss


def build_bilstm():
    inp = Input(shape=(SEQ_LEN, len(INPUT_FEATURES)))
    x = Bidirectional(LSTM(args.units1, return_sequences=True,
                              dropout=args.dropout, recurrent_dropout=0.1))(inp)
    x = Bidirectional(LSTM(args.units2, return_sequences=True,
                              dropout=args.dropout))(x)
    context, attn = AttentionLayer(name="attention")(x)
    x = Dense(32, activation="relu")(context)
    x = Dropout(args.dropout)(x)
    out = Dense(1, activation="sigmoid", name="prediction")(x)
    model = Model(inputs=inp, outputs=[out, attn])
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss={"prediction": focal_loss(gamma=2.0, alpha=0.25),
              "attention":  lambda y_true, y_pred: 0.0 * K.sum(y_pred)},
        loss_weights={"prediction": 1.0, "attention": 0.0},
        metrics={"prediction": [tf.keras.metrics.AUC(curve="PR", name="prauc"),
                                  tf.keras.metrics.AUC(name="auc")]}
    )
    return model


# =============================================================
# 3. TRAINING WITH CHECKPOINTING (model saved after every epoch)
# =============================================================
print(f"\nTraining: epochs={args.epochs}, batch={args.batch_size}, "
      f"units1={args.units1}, units2={args.units2}, lr={args.lr:.2e}")

tf.keras.utils.set_random_seed(42)
model = build_bilstm()
model.summary()

ckpt_path = OUT / f"bilstm_{args.horizon}.keras"
print(f"\nCheckpoint path: {ckpt_path}")
print(f"  → Model saved after every epoch where val_prauc improves")

callbacks = [
    ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor="val_prediction_prauc",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1,
    ),
]

dummy_train = np.zeros((len(y_tr_bal), SEQ_LEN), dtype=np.float32)
t0 = time.time()
history = model.fit(
    X_tr_bal,
    {"prediction": y_tr_bal, "attention": dummy_train},
    validation_split=0.15,
    epochs=args.epochs,
    batch_size=args.batch_size,
    callbacks=callbacks,
    verbose=2,
)
train_time = time.time() - t0
print(f"\nTraining done in {train_time:.0f}s ({len(history.history['loss'])} epochs)")
print(f"Best model saved at: {ckpt_path}")
print(f"\n>>> Now run: python 03b_evaluate_bilstm.py --skip-mc")
print(f"    to compute the final test metrics + predictions CSV.")
