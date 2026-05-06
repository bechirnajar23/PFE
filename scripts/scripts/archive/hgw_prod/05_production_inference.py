"""
Production Inference Pipeline
==============================
Real-time scoring for HGW telemetry using trained models.

Loads:
  - data/catboost_24h.cbm          (short-term ML model)
  - data/bilstm_72h.keras          (long-term DL model)

For each gateway, generates a prediction payload:
  {
    "gateway_id": "HGW_001",
    "timestamp":  "2026-04-30T12:00:00",
    "health_score": 73.4,
    "alerts": {
        "24h": {"prob": 0.05, "fire": false, "threshold": 0.69},
        "72h": {"prob": 0.18, "fire": false, "threshold": 0.53,
                 "uncertainty": 0.04, "attention_peak_step": 144}
    },
    "top_reasons": [
        {"feature": "cwmp_ma72h",      "value": 245.3, "shap": +0.32},
        {"feature": "saturation_idx",  "value": 0.87,  "shap": +0.21}
    ]
  }

Outputs:
  - data/predictions_live.json  — per-gateway latest scores
  - data/grafana_metrics.csv    — flat CSV for Grafana datasource
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument("--data",         default="data/hgw_short_term.csv",
                     help="Recent telemetry to score")
parser.add_argument("--data-long",    default="data/hgw_long_term.csv",
                     help="Long-term telemetry (30-min step) for LSTM")
parser.add_argument("--out-dir",      default="data")
parser.add_argument("--catboost-model", default="data/catboost_24h.cbm")
parser.add_argument("--lstm-model",     default="data/bilstm_72h.keras")
parser.add_argument("--latest-only",  action="store_true",
                     help="Score only the most recent N rows per gateway")
parser.add_argument("--n-latest",     type=int, default=24,
                     help="Number of recent rows to score per gateway (default 24)")
args = parser.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================
# HEALTH SCORE FUNCTION (Grafana-ready)
# =============================================================
def compute_health_score(cpu, mem, ping, loss, ttf_pred=None, alert_24h_prob=None):
    """
    Composite health score: 100% (healthy) -> 0% (crash imminent).
    Now factors in 24h alert probability for risk-aware scoring.
    """
    cpu, mem, ping, loss = [np.asarray(x, dtype=float) for x in [cpu, mem, ping, loss]]
    n_cpu = np.clip((cpu - 20) / 70, 0, 1)
    n_mem = np.clip((mem - 35) / 55, 0, 1)
    n_ping = np.clip((ping - 20) / 200, 0, 1)
    n_loss = np.clip(loss / 15, 0, 1)
    composite = 0.35 * n_mem + 0.30 * n_cpu + 0.20 * n_ping + 0.15 * n_loss

    if ttf_pred is not None:
        n_ttf = np.clip(1.0 - np.asarray(ttf_pred, dtype=float) / 720, 0, 1)
        composite = 0.5 * composite + 0.3 * n_ttf
        if alert_24h_prob is not None:
            composite += 0.2 * np.clip(np.asarray(alert_24h_prob, dtype=float), 0, 1)
    elif alert_24h_prob is not None:
        composite = 0.7 * composite + 0.3 * np.clip(np.asarray(alert_24h_prob, dtype=float), 0, 1)

    return np.round((1.0 - np.clip(composite, 0, 1)) * 100, 1)


# =============================================================
# 1. LOAD MODELS
# =============================================================
print("=" * 70)
print("Production Inference Pipeline")
print("=" * 70)

print(f"\nLoading CatBoost model from {args.catboost_model}...")
from catboost import CatBoostClassifier
cb_model = CatBoostClassifier()
cb_model.load_model(args.catboost_model)

# Load CatBoost metadata to get threshold + feature list
cb_meta_path = OUT / "catboost_24h_metadata.json"
with open(cb_meta_path) as f:
    cb_meta = json.load(f)
cb_threshold = cb_meta["metrics"]["threshold"]
print(f"  CatBoost threshold: {cb_threshold:.4f}")

print(f"\nLoading Bi-LSTM model from {args.lstm_model}...")
import tensorflow as tf
from tensorflow.keras.models import load_model

# Need custom objects (focal loss, AttentionLayer) for loading
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(K.equal(y_true, 1), alpha, 1 - alpha)
        return -K.mean(alpha_t * K.pow(1.0 - pt, gamma) * K.log(pt))
    return loss

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

lstm_model = load_model(args.lstm_model,
    custom_objects={"loss": focal_loss(), "AttentionLayer": AttentionLayer},
    compile=False)

# Build inference model that exposes attention
from tensorflow.keras.models import Model
attention_layer_output = None
for layer in lstm_model.layers:
    if isinstance(layer, AttentionLayer):
        attention_layer_output = layer.output
        break

if attention_layer_output is not None and isinstance(attention_layer_output, list):
    infer_model = Model(inputs=lstm_model.input,
                          outputs=[lstm_model.output, attention_layer_output[1]])
else:
    infer_model = lstm_model
print("  Bi-LSTM loaded with attention extraction")

# Load LSTM metadata
lstm_meta_path = OUT / "bilstm_72h_metadata.json"
with open(lstm_meta_path) as f:
    lstm_meta = json.load(f)
lstm_threshold = lstm_meta["metrics"]["threshold"]
LOOKBACK_STEPS = lstm_meta["architecture"]["lookback_days"] * 24 * 2  # 30-min step
SUBSAMPLE = lstm_meta["architecture"]["subsample_hours"] * 2
SEQ_LEN = lstm_meta["architecture"]["seq_len"]
INPUT_FEATURES = lstm_meta["architecture"]["input_features"]
print(f"  LSTM threshold: {lstm_threshold:.4f}  seq_len: {SEQ_LEN}")


# =============================================================
# 2. LOAD RECENT DATA
# =============================================================
print(f"\nLoading short-term data from {args.data}...")
df_short = pd.read_csv(args.data, parse_dates=["timestamp"])
df_short = df_short.sort_values(["gateway_id", "timestamp"]).reset_index(drop=True)
print(f"  {len(df_short):,} rows  ({df_short['gateway_id'].nunique()} gateways)")

print(f"Loading long-term data from {args.data_long}...")
df_long = pd.read_csv(args.data_long, parse_dates=["timestamp"])
df_long = df_long.sort_values(["gateway_id", "timestamp"]).reset_index(drop=True)
print(f"  {len(df_long):,} rows  ({df_long['gateway_id'].nunique()} gateways)")


# =============================================================
# 3. CATBOOST PREDICTIONS (24h)
# =============================================================
print(f"\n--- Scoring with CatBoost (24h) ---")

def build_catboost_features(df, sph=1):
    """Replicate features from training pipeline."""
    out = []
    for gw, group in df.groupby("gateway_id"):
        g = group.copy().sort_values("timestamp").reset_index(drop=True)
        d = {}
        for col, sh in [("cpu_load","cpu"), ("mem_used_pct","mem"),
                          ("ping_latency","ping"), ("packet_loss","loss"),
                          ("cwmp_rss_mb","cwmp")]:
            for lag_h in [1, 3, 6, 12, 24, 72]:
                d[f"{sh}_lag{lag_h}h"] = g[col].shift(lag_h * sph)
            for win_h in [6, 24, 72]:
                w = win_h * sph
                d[f"{sh}_ma{win_h}h"]  = g[col].rolling(w, min_periods=1).mean()
                d[f"{sh}_std{win_h}h"] = g[col].rolling(w, min_periods=1).std()
                d[f"{sh}_max{win_h}h"] = g[col].rolling(w, min_periods=1).max()
            d[f"{sh}_ewm12"]    = g[col].ewm(span=12*sph, adjust=False).mean()
            d[f"{sh}_slope6h"]  = g[col].diff(6*sph) / 6
            d[f"{sh}_slope24h"] = g[col].diff(24*sph) / 24
        d["cpu_x_mem"]       = g["cpu_load"] * g["mem_used_pct"] / 10000
        d["saturation_idx"]  = (g["cpu_load"]/88 + g["mem_used_pct"]/90) / 2
        d["mem_headroom"]    = np.clip(90.0 - g["mem_used_pct"], 0, 90)
        d["cwmp_share_mem"]  = g["cwmp_rss_mb"] / 936.0
        d["sin_hour"]   = np.sin(2*np.pi*g["hour"]/24)
        d["cos_hour"]   = np.cos(2*np.pi*g["hour"]/24)
        d["sin_dow"]    = np.sin(2*np.pi*g["dow"]/7)
        d["cos_dow"]    = np.cos(2*np.pi*g["dow"]/7)
        d["sin_month"]  = np.sin(2*np.pi*g["timestamp"].dt.month/12)
        d["cos_month"]  = np.cos(2*np.pi*g["timestamp"].dt.month/12)
        d["is_weekend"] = (g["dow"] >= 5).astype(int)
        d["wan_status"] = g["wan_status"]
        d["wan_outage_streak"] = (
            g["wan_status"].eq(0)
            .groupby((g["wan_status"] != g["wan_status"].shift()).cumsum())
            .cumcount()
        ) * g["wan_status"].eq(0).astype(int)
        feats = pd.DataFrame(d, index=g.index)
        cats = g[["gateway_id", "firmware", "region", "isp"]].copy()
        existing = g[["cpu_mean_24h","ram_mean_24h","cpu_std_24h","ram_std_24h",
                       "cpu_slope_6h","ram_slope_6h","wan_instability_6h",
                       "cwmp_rss_mb","dhcp_rss_mb","nemo_rss_mb",
                       "cpu_load","mem_used_pct","ping_latency","packet_loss",
                       "reboot_event","recovery_phase","timestamp"]].copy()
        out.append(pd.concat([cats, existing, feats], axis=1))
    return pd.concat(out).sort_values(["timestamp", "gateway_id"]).reset_index(drop=True)

df_feat = build_catboost_features(df_short).fillna(0)
# Score only the recent window if requested
if args.latest_only:
    df_feat = df_feat.groupby("gateway_id").tail(args.n_latest).reset_index(drop=True)
    df_short = df_short.groupby("gateway_id").tail(args.n_latest).reset_index(drop=True)

# Reproduce TRAINING column order: NUM_FEATURES first, then CAT_FEATURES
CAT_FEATURES = ["gateway_id", "firmware", "region", "isp"]
TARGET_COLS = ["timestamp"]
NUM_FEATURES = [c for c in df_feat.columns if c not in TARGET_COLS + CAT_FEATURES]
X_score = df_feat[NUM_FEATURES + CAT_FEATURES]

print(f"  Scoring {len(X_score):,} rows...")
cb_probs = cb_model.predict_proba(X_score)[:, 1]
cb_alerts = (cb_probs >= cb_threshold).astype(int)
print(f"  Alerts firing: {cb_alerts.sum():,}  ({cb_alerts.mean()*100:.2f}%)")


# =============================================================
# 4. BI-LSTM PREDICTIONS (72h) — only on recent window
# =============================================================
print(f"\n--- Scoring with Bi-LSTM (72h) ---")

# Build sequences for the most recent timestamp per gateway
from sklearn.preprocessing import StandardScaler

lstm_predictions = {}
for gw, group in df_long.groupby("gateway_id"):
    group = group.sort_values("timestamp").reset_index(drop=True)
    if len(group) < LOOKBACK_STEPS + 1:
        continue
    # Fit scaler on first 75% (mimic training)
    sp = int(len(group) * 0.75)
    scaler = StandardScaler()
    scaler.fit(group.iloc[:sp][INPUT_FEATURES].fillna(0).values)
    full_scaled = scaler.transform(group[INPUT_FEATURES].fillna(0).values)

    # Score the last N timesteps
    n_to_score = min(args.n_latest, len(group) - LOOKBACK_STEPS)
    sequences = []
    timestamps_scored = []
    for i in range(len(group) - n_to_score, len(group)):
        if i - LOOKBACK_STEPS < 0:
            continue
        sequences.append(full_scaled[i-LOOKBACK_STEPS:i:SUBSAMPLE])
        timestamps_scored.append(group.iloc[i]["timestamp"])
    if not sequences:
        continue
    X_seq = np.asarray(sequences, dtype=np.float32)

    # Predict (with attention)
    out = infer_model(X_seq, training=False)
    if isinstance(out, list):
        probs, attn = out[0].numpy().flatten(), out[1].numpy()
    else:
        probs = out.numpy().flatten()
        attn = np.zeros((len(probs), SEQ_LEN))

    # MC dropout uncertainty (5 forward passes)
    mc_runs = []
    for _ in range(5):
        out_mc = infer_model(X_seq, training=True)
        mc_runs.append(out_mc[0].numpy().flatten() if isinstance(out_mc, list) else out_mc.numpy().flatten())
    mc_std = np.stack(mc_runs).std(axis=0)

    # Attention peak per row
    attn_peak = attn.argmax(axis=1) if attn.ndim == 2 else np.zeros(len(probs))

    lstm_predictions[gw] = {
        "timestamps":  timestamps_scored,
        "probs":       probs,
        "alerts":      (probs >= lstm_threshold).astype(int),
        "uncertainty": mc_std,
        "attention_peak_step": attn_peak,
    }

n_lstm_alerts = sum(int(p["alerts"].sum()) for p in lstm_predictions.values())
print(f"  LSTM alerts firing: {n_lstm_alerts:,} across {len(lstm_predictions)} gateways")


# =============================================================
# 5. BUILD PER-GATEWAY PAYLOAD
# =============================================================
print(f"\n--- Building Grafana payload ---")

# Pull SHAP top-features from CatBoost metadata for explanation
cb_top_features = list(cb_meta.get("top15_features_gain", {}).keys())[:5]

predictions_live = {}
grafana_rows = []

for gw in df_short["gateway_id"].unique():
    gw_short = df_short[df_short["gateway_id"] == gw].iloc[-1]
    gw_idx_in_score = df_feat[df_feat["gateway_id"] == gw].index
    if len(gw_idx_in_score) == 0:
        continue
    last_score_idx = gw_idx_in_score[-1]
    cb_prob = float(cb_probs[df_feat.index.get_loc(last_score_idx)])
    cb_alert = bool(cb_prob >= cb_threshold)

    if gw in lstm_predictions:
        ls = lstm_predictions[gw]
        last = -1
        lstm_prob = float(ls["probs"][last])
        lstm_unc  = float(ls["uncertainty"][last])
        lstm_alert = bool(ls["alerts"][last])
        attn_peak = int(ls["attention_peak_step"][last])
    else:
        lstm_prob, lstm_unc, lstm_alert, attn_peak = 0.0, 0.0, False, 0

    # Health score blends both alarms
    hs = float(compute_health_score(
        gw_short["cpu_load"], gw_short["mem_used_pct"],
        gw_short["ping_latency"], gw_short["packet_loss"],
        ttf_pred=None, alert_24h_prob=cb_prob,
    ))

    # Top reasons (using gain importance values from training metadata)
    top_reasons = [
        {"feature": feat, "value": float(gw_short.get(feat, X_score.iloc[last_score_idx].get(feat, 0)))}
        for feat in cb_top_features[:3] if feat in gw_short.index or feat in X_score.columns
    ]

    payload = {
        "gateway_id": gw,
        "timestamp":  str(gw_short["timestamp"]),
        "firmware":   str(gw_short.get("firmware", "")),
        "region":     str(gw_short.get("region", "")),
        "health_score": hs,
        "current_metrics": {
            "cpu_load":     round(float(gw_short["cpu_load"]), 2),
            "mem_used_pct": round(float(gw_short["mem_used_pct"]), 2),
            "ping_latency": round(float(gw_short["ping_latency"]), 2),
            "packet_loss":  round(float(gw_short["packet_loss"]), 3),
            "wan_status":   int(gw_short["wan_status"]),
        },
        "alerts": {
            "24h": {
                "model": "CatBoost",
                "prob":  round(cb_prob, 4),
                "fire":  cb_alert,
                "threshold": round(cb_threshold, 4),
            },
            "72h": {
                "model": "Bi-LSTM",
                "prob":  round(lstm_prob, 4),
                "fire":  lstm_alert,
                "threshold": round(lstm_threshold, 4),
                "uncertainty_std": round(lstm_unc, 4),
                "attention_peak_step": attn_peak,
            },
        },
        "top_reasons": top_reasons,
    }
    predictions_live[gw] = payload
    grafana_rows.append({
        "timestamp":     payload["timestamp"],
        "gateway_id":    gw,
        "health_score":  hs,
        "alert_24h_prob": cb_prob,
        "alert_24h_fire": int(cb_alert),
        "alert_72h_prob": lstm_prob,
        "alert_72h_fire": int(lstm_alert),
        "alert_72h_uncertainty": lstm_unc,
        "cpu_load":     payload["current_metrics"]["cpu_load"],
        "mem_used_pct": payload["current_metrics"]["mem_used_pct"],
        "wan_status":   payload["current_metrics"]["wan_status"],
    })


# =============================================================
# 6. SAVE OUTPUTS
# =============================================================
out_json = OUT / "predictions_live.json"
with open(out_json, "w") as f:
    json.dump(predictions_live, f, indent=2, default=str)
print(f"  Predictions JSON -> {out_json}")

out_csv = OUT / "grafana_metrics.csv"
pd.DataFrame(grafana_rows).to_csv(out_csv, index=False)
print(f"  Grafana CSV      -> {out_csv}")

# Summary
print(f"\n{'='*60}")
print(f"INFERENCE SUMMARY")
print(f"{'='*60}")
for gw, p in predictions_live.items():
    icon = "🔴" if p["alerts"]["24h"]["fire"] or p["alerts"]["72h"]["fire"] else "🟢"
    print(f"  {icon} {gw}  health={p['health_score']:>5.1f}%  "
          f"24h={p['alerts']['24h']['prob']:.3f}{'!' if p['alerts']['24h']['fire'] else ' '}  "
          f"72h={p['alerts']['72h']['prob']:.3f}±{p['alerts']['72h']['uncertainty_std']:.3f}"
          f"{'!' if p['alerts']['72h']['fire'] else ''}")

print("\nDone.")
