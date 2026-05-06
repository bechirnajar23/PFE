"""
03b_evaluate_bilstm.py — Evaluate a saved Bi-LSTM (no retraining)
==================================================================
Loads the trained model from data/bilstm_72h.keras, runs predictions
in batched fashion (avoids OOM), computes MC Dropout uncertainty,
and writes the metadata + predictions files that the original training
script tried to produce before crashing.

Usage:
    python 03b_evaluate_bilstm.py --horizon 72h
    python 03b_evaluate_bilstm.py --batch-size 256 --mc-passes 10
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
parser.add_argument("--data",        default="data/hgw_long_term.csv")
parser.add_argument("--model-path",  default=None,
                     help="Default: data/bilstm_<horizon>.keras")
parser.add_argument("--out-dir",     default="data")
parser.add_argument("--horizon",     default="72h", choices=["72h", "7d"])
parser.add_argument("--lookback",    type=int, default=21)
parser.add_argument("--lookback-hours-step", type=int, default=2)
parser.add_argument("--test-stride", type=int, default=2)
parser.add_argument("--batch-size",  type=int, default=256)
parser.add_argument("--mc-passes",   type=int, default=10)
parser.add_argument("--mc-batch",    type=int, default=128,
                     help="Smaller batch for MC dropout to avoid OOM")
parser.add_argument("--skip-mc",     action="store_true",
                     help="Skip MC Dropout (faster but no uncertainty)")
args = parser.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
LABEL_COL = f"incident_in_{args.horizon}"
MODEL_PATH = args.model_path or f"data/bilstm_{args.horizon}.keras"


# =============================================================
# 1. LOAD DATA + REBUILD SEQUENCES
# =============================================================
print("=" * 70)
print(f"Bi-LSTM Evaluation — {args.horizon}")
print("=" * 70)

df = pd.read_csv(args.data, parse_dates=["timestamp"])
df = df.sort_values(["gateway_id", "timestamp"]).reset_index(drop=True)
SPH = 2
print(f"Loaded: {len(df):,} rows  ({df['gateway_id'].nunique()} gateways)")

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
print(f"Lookback: {args.lookback}d → {SEQ_LEN} timesteps × {len(INPUT_FEATURES)} features")

print("\nBuilding test sequences (skipping training portion)...")
from sklearn.preprocessing import StandardScaler

X_te_all, y_te_all, gw_te_all, ts_te_all = [], [], [], []
for gw, group in df.groupby("gateway_id"):
    group = group.sort_values("timestamp").reset_index(drop=True)
    sp = int(len(group) * 0.75)
    while sp > len(group) * 0.50:
        if group.iloc[sp:][LABEL_COL].sum() >= 50:
            break
        sp = int(sp * 0.95)

    scaler = StandardScaler()
    X_train_raw = group.iloc[:sp][INPUT_FEATURES].fillna(0).values
    X_test_raw  = group.iloc[sp:][INPUT_FEATURES].fillna(0).values
    scaler.fit(X_train_raw)
    X_full_scaled = np.vstack([
        scaler.transform(X_train_raw),
        scaler.transform(X_test_raw),
    ]).astype(np.float32)
    labels = group[LABEL_COL].values
    ts     = group["timestamp"].values

    for i in range(sp, len(group), args.test_stride):
        if i - LOOKBACK_STEPS < 0:
            continue
        X_te_all.append(X_full_scaled[i-LOOKBACK_STEPS:i:SUBSAMPLE])
        y_te_all.append(labels[i])
        gw_te_all.append(gw)
        ts_te_all.append(ts[i])

X_te = np.asarray(X_te_all, dtype=np.float32)
y_te = np.asarray(y_te_all, dtype=np.float32)
gw_te = np.asarray(gw_te_all)
ts_te = pd.to_datetime(ts_te_all)
print(f"  Test sequences: {X_te.shape}  positives: {int(y_te.sum())} ({y_te.mean()*100:.2f}%)")


# =============================================================
# 2. LOAD MODEL
# =============================================================
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
print(f"\nTensorFlow {tf.__version__}")

# Get the register decorator (path varies between TF versions)
try:
    register_keras = tf.keras.saving.register_keras_serializable
except AttributeError:
    try:
        from keras.saving import register_keras_serializable as register_keras
    except ImportError:
        # Fallback: no-op decorator (loading still works with custom_objects)
        def register_keras(*a, **k):
            def deco(cls): return cls
            return deco

# Reconstruct custom AttentionLayer (must match training script)
@register_keras()
class AttentionLayer(layers.Layer):
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


print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"AttentionLayer": AttentionLayer},
    compile=False,
    safe_mode=False,
)
print(f"  Model loaded: {model.count_params():,} parameters")


# =============================================================
# 3. BATCHED PREDICTION (avoids OOM)
# =============================================================
print(f"\nPredicting on {len(X_te):,} sequences (batch_size={args.batch_size})...")
t0 = time.time()
y_prob_list = []
attn_list = []
n_batches = (len(X_te) + args.batch_size - 1) // args.batch_size
for i in range(n_batches):
    chunk = X_te[i*args.batch_size : (i+1)*args.batch_size]
    yp_chunk, attn_chunk = model.predict(chunk, verbose=0, batch_size=args.batch_size)
    y_prob_list.append(yp_chunk.flatten())
    attn_list.append(attn_chunk)
    if i % 20 == 0:
        print(f"  batch {i+1}/{n_batches}", flush=True)

y_prob = np.concatenate(y_prob_list)
attn_weights = np.concatenate(attn_list, axis=0)
print(f"  Done in {time.time()-t0:.0f}s")


# =============================================================
# 4. BATCHED MC DROPOUT (uncertainty estimation)
# =============================================================
mc_mean = y_prob.copy()
mc_std  = np.zeros_like(y_prob)

if not args.skip_mc:
    print(f"\nRunning MC Dropout ({args.mc_passes} passes, batch={args.mc_batch})...")
    t0 = time.time()
    mc_predictions = []
    for pass_idx in range(args.mc_passes):
        pass_results = []
        for i in range(0, len(X_te), args.mc_batch):
            chunk = X_te[i : i+args.mc_batch]
            yp_chunk, _ = model(chunk, training=True)
            pass_results.append(yp_chunk.numpy().flatten())
        mc_predictions.append(np.concatenate(pass_results))
        print(f"  pass {pass_idx+1}/{args.mc_passes} done")

    mc_array = np.stack(mc_predictions, axis=0)
    mc_mean = mc_array.mean(axis=0)
    mc_std  = mc_array.std(axis=0)
    print(f"  MC Dropout done in {time.time()-t0:.0f}s")
    print(f"  Mean uncertainty σ = {mc_std.mean():.4f}")


# =============================================================
# 5. METRICS + THRESHOLD TUNING
# =============================================================
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                confusion_matrix, precision_recall_curve,
                                classification_report)

prec, rec, thresh = precision_recall_curve(y_te, y_prob)
denom = 4*prec + rec
f2 = np.where(denom == 0, 0, (5*prec*rec) / np.maximum(denom, 1e-9))
best_th = float(thresh[int(np.argmax(f2[:-1]))]) if len(thresh) > 0 else 0.5
y_pred = (y_prob >= best_th).astype(int)
cm = confusion_matrix(y_te, y_pred)

print(f"\n{'='*60}\nFINAL METRICS — Bi-LSTM {args.horizon}\n{'='*60}")
print(f"  ROC-AUC:   {roc_auc_score(y_te, y_prob):.4f}")
print(f"  PR-AUC:    {average_precision_score(y_te, y_prob):.4f}")
print(f"  F1 (best): {f1_score(y_te, y_pred):.4f}")
print(f"  Threshold: {best_th:.4f}  (F2-optimized)")
print(f"\n  Confusion matrix:")
print(f"           Pred=0   Pred=1")
print(f"  True=0   {cm[0,0]:>6}   {cm[0,1]:>6}")
print(f"  True=1   {cm[1,0]:>6}   {cm[1,1]:>6}")
print(f"\n{classification_report(y_te, y_pred, target_names=['Normal', 'Incident'])}")

# Per-gateway breakdown
print(f"\nPer-gateway test PR-AUC:")
for gw in np.unique(gw_te):
    m = gw_te == gw
    if m.sum() > 0 and y_te[m].sum() > 0:
        gw_prauc = average_precision_score(y_te[m], y_prob[m])
        print(f"  {gw}: {gw_prauc:.4f}  (n={m.sum()}, pos={int(y_te[m].sum())})")


# =============================================================
# 6. SAVE ARTIFACTS
# =============================================================
# Predictions CSV
preds_df = pd.DataFrame({
    "timestamp": ts_te,
    "gateway_id": gw_te,
    "y_true":  y_te.astype(int),
    "y_prob":  y_prob,
    "y_pred":  y_pred,
    "mc_mean": mc_mean,
    "mc_std":  mc_std,
})
preds_path = OUT / f"bilstm_{args.horizon}_predictions.csv"
preds_df.to_csv(preds_path, index=False)
print(f"\nPredictions → {preds_path}  ({len(preds_df):,} rows)")

# Attention summary on positives
pos_idx = np.where(y_te == 1)[0]
attn_pos = attn_weights[pos_idx].mean(axis=0).tolist() if len(pos_idx) > 0 else []
attn_neg = attn_weights[y_te == 0].mean(axis=0).tolist() if (y_te == 0).any() else []

# Metadata
metadata = {
    "horizon": args.horizon,
    "label_column": LABEL_COL,
    "model_type": "Bi-LSTM with Attention + Focal Loss + MC Dropout",
    "model_path": str(MODEL_PATH),
    "architecture": {
        "lookback_days": args.lookback,
        "subsample_hours": args.lookback_hours_step,
        "seq_len": SEQ_LEN,
        "n_features": len(INPUT_FEATURES),
        "input_features": INPUT_FEATURES,
        "total_params": int(model.count_params()),
    },
    "metrics": {
        "roc_auc": round(float(roc_auc_score(y_te, y_prob)), 4),
        "pr_auc":  round(float(average_precision_score(y_te, y_prob)), 4),
        "f1":      round(float(f1_score(y_te, y_pred)), 4),
        "threshold": round(best_th, 4),
        "tp": int(cm[1,1]), "fp": int(cm[0,1]),
        "fn": int(cm[1,0]), "tn": int(cm[0,0]),
        "mc_dropout_mean_std": round(float(mc_std.mean()), 4),
    },
    "test_set": {
        "sequences": int(len(X_te)),
        "positives": int(y_te.sum()),
        "positive_rate": round(float(y_te.mean()), 4),
    },
    "attention_summary": {
        "seq_len": SEQ_LEN,
        "mean_on_positives": [round(float(v), 5) for v in attn_pos],
        "mean_on_negatives": [round(float(v), 5) for v in attn_neg],
    },
}
meta_path = OUT / f"bilstm_{args.horizon}_metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata    → {meta_path}")
print("\nDone.")
