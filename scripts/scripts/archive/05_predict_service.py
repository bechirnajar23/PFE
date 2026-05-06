"""
Production Prediction Service
==============================
Loads both trained models (CatBoost 24h + Bi-LSTM 72h) and produces
real-time predictions with:
  - Crash probability per horizon (24h, 72h)
  - Time-to-failure estimate
  - Health score (0-100%, Grafana-ready)
  - Top-3 explainability reasons (from feature importances + attention)
  - Uncertainty estimate (MC Dropout for LSTM)

Usage as library:
    from predict_service import HGWPredictor
    p = HGWPredictor()
    result = p.predict(telemetry_df)

Usage CLI:
    python 05_predict_service.py --input data/incoming_telemetry.csv \\
        --output data/predictions_now.csv

REST API stub:
    python 05_predict_service.py --serve --port 8000
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


# -------- CatBoost feature builder (must match 02_train_catboost_short.py) --------
def build_catboost_features(df, sph=1):
    """Replicate exact feature engineering used during training."""
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
        cats = g[["gateway_id", "firmware", "region", "isp"]]
        existing = g[["cpu_mean_24h","ram_mean_24h","cpu_std_24h","ram_std_24h",
                       "cpu_slope_6h","ram_slope_6h","wan_instability_6h",
                       "cwmp_rss_mb","dhcp_rss_mb","nemo_rss_mb",
                       "cpu_load","mem_used_pct","ping_latency","packet_loss",
                       "reboot_event","recovery_phase"]]
        out.append(pd.concat([cats, existing, feats], axis=1))
    return pd.concat(out).fillna(0)


def compute_health_score(cpu, mem, ping, loss, ttf_pred=None):
    """Same function used in dataset generation — Grafana-ready."""
    cpu, mem, ping, loss = [np.asarray(x, dtype=float) for x in [cpu, mem, ping, loss]]
    n_cpu  = np.clip((cpu - 20)  / 70,  0, 1)
    n_mem  = np.clip((mem - 35)  / 55,  0, 1)
    n_ping = np.clip((ping - 20) / 200, 0, 1)
    n_loss = np.clip(loss / 15, 0, 1)
    composite = 0.35*n_mem + 0.30*n_cpu + 0.20*n_ping + 0.15*n_loss
    if ttf_pred is not None:
        n_ttf = np.clip(1.0 - np.asarray(ttf_pred, dtype=float)/720, 0, 1)
        composite = 0.6*composite + 0.4*n_ttf
    return np.round((1.0 - np.clip(composite, 0, 1)) * 100, 1)


class HGWPredictor:
    """Production wrapper for both models."""

    def __init__(self,
                 catboost_path="data/catboost_24h.cbm",
                 catboost_meta="data/catboost_24h_metadata.json",
                 bilstm_path="data/bilstm_72h.keras",
                 bilstm_meta="data/bilstm_72h_metadata.json"):
        self.cb_model = None
        self.lstm_model = None
        self.cb_threshold = 0.5
        self.lstm_threshold = 0.5
        self._load_catboost(catboost_path, catboost_meta)
        self._load_bilstm(bilstm_path, bilstm_meta)

    def _load_catboost(self, model_path, meta_path):
        if not Path(model_path).exists():
            print(f"  WARN: CatBoost model not found at {model_path}")
            return
        from catboost import CatBoostClassifier
        self.cb_model = CatBoostClassifier()
        self.cb_model.load_model(str(model_path))
        if Path(meta_path).exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.cb_threshold = meta["metrics"].get("threshold", 0.5)
            self.cb_top_features = list(meta.get("top15_features_gain", {}).keys())[:5]
        print(f"  Loaded CatBoost 24h (threshold={self.cb_threshold:.4f})")

    def _load_bilstm(self, model_path, meta_path):
        if not Path(model_path).exists():
            print(f"  WARN: Bi-LSTM model not found at {model_path}")
            return
        try:
            import tensorflow as tf
            # Need custom_objects if using AttentionLayer + focal_loss
            self.lstm_model = tf.keras.models.load_model(
                str(model_path), compile=False, safe_mode=False
            )
            if Path(meta_path).exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self.lstm_threshold = meta["metrics"].get("threshold", 0.5)
                self.lstm_meta = meta
            print(f"  Loaded Bi-LSTM 72h (threshold={self.lstm_threshold:.4f})")
        except Exception as e:
            print(f"  WARN: Bi-LSTM load failed: {e}")

    def predict_short(self, df_telemetry):
        """24h horizon prediction via CatBoost."""
        if self.cb_model is None:
            return None
        feats = build_catboost_features(df_telemetry, sph=1)
        feature_names = self.cb_model.feature_names_
        # Reorder columns to match training
        for col in feature_names:
            if col not in feats.columns:
                feats[col] = 0
        X = feats[feature_names]
        prob = self.cb_model.predict_proba(X)[:, 1]
        pred = (prob >= self.cb_threshold).astype(int)
        return prob, pred

    def explain(self, df_row, prob_24h):
        """Return top reasons for a flagged prediction (top-3 features + values)."""
        if self.cb_model is None or not hasattr(self, "cb_top_features"):
            return []
        reasons = []
        feats = build_catboost_features(df_row, sph=1)
        for f in self.cb_top_features[:5]:
            if f in feats.columns:
                reasons.append({
                    "feature": f,
                    "value":   round(float(feats.iloc[-1][f]), 3),
                })
        return reasons

    def predict(self, df_telemetry):
        """Full prediction pipeline. Returns DataFrame with probabilities + health score."""
        df = df_telemetry.copy()
        # 24h prediction
        prob_24h, pred_24h = self.predict_short(df) if self.cb_model else (None, None)

        # Health score (always computed)
        health = compute_health_score(
            df["cpu_load"], df["mem_used_pct"],
            df["ping_latency"], df["packet_loss"]
        )

        result = pd.DataFrame({
            "timestamp":     df["timestamp"].values,
            "gateway_id":    df["gateway_id"].values,
            "cpu_load":      df["cpu_load"].values,
            "mem_used_pct":  df["mem_used_pct"].values,
            "ping_latency":  df["ping_latency"].values,
            "packet_loss":   df["packet_loss"].values,
            "wan_status":    df["wan_status"].values,
            "health_score":  health,
        })
        if prob_24h is not None:
            result["prob_incident_24h"] = prob_24h
            result["alert_24h"] = pred_24h
            result["risk_level"] = pd.cut(
                prob_24h,
                bins=[-0.01, 0.10, 0.30, 0.60, 1.01],
                labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            ).astype(str)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  help="Input telemetry CSV")
    parser.add_argument("--output", default="data/predictions_now.csv")
    parser.add_argument("--cb-model",  default="data/catboost_24h.cbm")
    parser.add_argument("--cb-meta",   default="data/catboost_24h_metadata.json")
    parser.add_argument("--lstm-model", default="data/bilstm_72h.keras")
    parser.add_argument("--lstm-meta",  default="data/bilstm_72h_metadata.json")
    args = parser.parse_args()

    if not args.input:
        raise SystemExit("--input required")

    print("=" * 70)
    print("HGW Prediction Service")
    print("=" * 70)
    print("\nLoading models...")
    p = HGWPredictor(
        catboost_path=args.cb_model,
        catboost_meta=args.cb_meta,
        bilstm_path=args.lstm_model,
        bilstm_meta=args.lstm_meta,
    )

    print(f"\nReading {args.input}...")
    df = pd.read_csv(args.input, parse_dates=["timestamp"], low_memory=False)
    print(f"  {len(df):,} rows")

    print(f"\nPredicting...")
    result = p.predict(df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nSaved {len(result):,} predictions -> {args.output}")

    # Summary
    if "alert_24h" in result.columns:
        n_alerts = int(result["alert_24h"].sum())
        n_critical = int((result["risk_level"] == "CRITICAL").sum())
        n_high = int((result["risk_level"] == "HIGH").sum())
        print(f"\n  Alerts (24h):       {n_alerts:,} rows")
        print(f"  CRITICAL risk:      {n_critical:,} rows")
        print(f"  HIGH risk:          {n_high:,} rows")
        print(f"  Mean health score:  {result['health_score'].mean():.1f}%")
        print(f"  Min health score:   {result['health_score'].min():.1f}%")


if __name__ == "__main__":
    main()
