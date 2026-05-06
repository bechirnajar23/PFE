"""
HGW Incident Predictor — Production Inference Script
=====================================================

Drop-in module for predicting HGW incidents 30 minutes ahead.
Plug into your Telnet collector loop. No external state required.

Usage:
    from predict_incident_prod import IncidentPredictor

    predictor = IncidentPredictor(
        model_path="catboost_30min_real.cbm",
        bundle_path="production_bundle.json",
    )

    # In the collector loop (every 1-5 minutes):
    df_recent = fetch_last_60_min_telemetry()  # standard collector schema
    result = predictor.predict(df_recent)

    if result["prediction"] == 1:
        send_alert(result)

Output schema:
    {
        "prediction": 1,                          # 0 = normal, 1 = incident likely
        "probability": 0.9234,                    # [0,1]
        "confidence_level": "INCIDENT_LIKELY",    # LOW_RISK | WATCH | INCIDENT_LIKELY | INCIDENT_VERY_LIKELY
        "threshold_used": 0.7833,
        "horizon_min": 30,
        "top_features": [                         # SHAP-ranked drivers
            {"feature": "cpu_mean_30min", "value": 78.4, "shap": +0.62, "direction": "increases_risk"},
            ...
        ]
    }
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool


# =============================================================================
# Schema mapping — universal Telnet adapter
# =============================================================================
def map_real_to_standard(df_raw, gateway_id="HGW_REAL_001"):
    """Maps the Telnet collector CSV schema → standard model schema."""
    df = pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df["gateway_id"] = gateway_id
    df["firmware"] = "unknown"
    df["cpu_load"] = df_raw["CPU_USAGE_PERCENT"].astype(float)
    df["mem_used_pct"] = df_raw["MEM_USAGE_PERCENT"].astype(float)
    df["ping_latency"] = (
        pd.to_numeric(df_raw["NET_LATENCY_MS"], errors="coerce")
        .ffill()
        .fillna(50.0)
    )
    df["wan_status"] = (df_raw["WAN_STATE"] == "UP").astype(int)
    df["packet_loss"] = (df_raw["NET_PING_STATUS"] == "FAIL").astype(int) * 100.0
    df["cwmp_rss_mb"] = 0.0
    df["dhcp_rss_mb"] = 0.0
    df["nemo_rss_mb"] = 0.0

    dhcp_run = (df_raw["DHCP_PROCESS_STATUS"] == "RUNNING").astype(int).values
    df["reboot_event"] = np.concatenate(
        [[0], (dhcp_run[1:] == 1) & (dhcp_run[:-1] == 0)]
    ).astype(int)
    df["recovery_phase"] = 0

    if "session_id" in df_raw.columns:
        df["session_id"] = df_raw["session_id"].values
    else:
        df["session_id"] = 0
    return df.sort_values("timestamp").reset_index(drop=True)


# =============================================================================
# Resampling
# =============================================================================
def resample_per_session(df, freq="1min"):
    out = []
    numeric_cols = [
        "cpu_load", "mem_used_pct", "ping_latency", "packet_loss",
        "cwmp_rss_mb", "dhcp_rss_mb", "nemo_rss_mb", "wan_status",
        "reboot_event", "recovery_phase",
    ]
    for sid, group in df.groupby("session_id"):
        if len(group) < 2:
            continue
        g = group.set_index("timestamp").sort_index()
        g_num = g[numeric_cols].resample(freq).mean()
        merged = g_num.ffill().dropna(subset=["cpu_load"])
        merged["session_id"] = sid
        merged["gateway_id"] = group["gateway_id"].iloc[0]
        out.append(merged.reset_index())
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


# =============================================================================
# Feature engineering
# =============================================================================
def compute_health_score(cpu, mem, ping, loss):
    cpu, mem, ping, loss = [np.asarray(x, dtype=float) for x in [cpu, mem, ping, loss]]
    n_cpu = np.clip((cpu - 20) / 70, 0, 1)
    n_mem = np.clip((mem - 35) / 55, 0, 1)
    n_ping = np.clip((ping - 20) / 200, 0, 1)
    n_loss = np.clip(loss / 15, 0, 1)
    composite = 0.35 * n_mem + 0.30 * n_cpu + 0.20 * n_ping + 0.15 * n_loss
    return np.round((1.0 - np.clip(composite, 0, 1)) * 100, 1)


def build_features(df):
    out = []
    for sid, group in df.groupby("session_id"):
        g = group.copy().sort_values("timestamp").reset_index(drop=True)
        g["cpu_slope_30min"] = g["cpu_load"].diff(30).fillna(0) / 30
        g["ram_slope_30min"] = g["mem_used_pct"].diff(30).fillna(0) / 30
        g["cpu_slope_5min"] = g["cpu_load"].diff(5).fillna(0) / 5
        g["ram_slope_5min"] = g["mem_used_pct"].diff(5).fillna(0) / 5
        g["cpu_mean_5min"] = g["cpu_load"].rolling(5, min_periods=1).mean()
        g["cpu_mean_30min"] = g["cpu_load"].rolling(30, min_periods=1).mean()
        g["cpu_std_30min"] = g["cpu_load"].rolling(30, min_periods=1).std().fillna(0)
        g["cpu_max_30min"] = g["cpu_load"].rolling(30, min_periods=1).max()
        g["mem_mean_5min"] = g["mem_used_pct"].rolling(5, min_periods=1).mean()
        g["mem_mean_30min"] = g["mem_used_pct"].rolling(30, min_periods=1).mean()
        g["mem_std_30min"] = g["mem_used_pct"].rolling(30, min_periods=1).std().fillna(0)
        g["mem_max_30min"] = g["mem_used_pct"].rolling(30, min_periods=1).max()
        g["ping_mean_5min"] = g["ping_latency"].rolling(5, min_periods=1).mean()
        g["ping_mean_30min"] = g["ping_latency"].rolling(30, min_periods=1).mean()
        g["ping_max_5min"] = g["ping_latency"].rolling(5, min_periods=1).max()
        g["loss_mean_5min"] = g["packet_loss"].rolling(5, min_periods=1).mean()
        g["wan_instability_5min"] = g["wan_status"].eq(0).rolling(5, min_periods=1).mean()
        for lag in [1, 3, 5, 10, 15]:
            g[f"cpu_lag{lag}m"] = g["cpu_load"].shift(lag).bfill()
            g[f"mem_lag{lag}m"] = g["mem_used_pct"].shift(lag).bfill()
        g["hour"] = g["timestamp"].dt.hour
        g["dow"] = g["timestamp"].dt.dayofweek
        g["sin_hour"] = np.sin(2 * np.pi * g["hour"] / 24)
        g["cos_hour"] = np.cos(2 * np.pi * g["hour"] / 24)
        g["cpu_x_mem"] = g["cpu_load"] * g["mem_used_pct"] / 10000
        g["saturation_idx"] = (g["cpu_load"] / 88 + g["mem_used_pct"] / 90) / 2
        g["mem_headroom"] = np.clip(90.0 - g["mem_used_pct"], 0, 90)
        g["health_score"] = compute_health_score(
            g["cpu_load"].fillna(0), g["mem_used_pct"].fillna(0),
            g["ping_latency"].fillna(50), g["packet_loss"].fillna(0),
        )
        out.append(g)
    return pd.concat(out, ignore_index=True)


# =============================================================================
# Predictor class
# =============================================================================
class IncidentPredictor:
    """Production-ready HGW incident predictor."""

    def __init__(
        self,
        model_path: str = "catboost_30min_real.cbm",
        bundle_path: str = "production_bundle.json",
        threshold_strategy: str = "balanced_F1",
    ):
        """
        Args:
            model_path: Path to the trained CatBoost model (.cbm)
            bundle_path: Path to production_bundle.json (contains thresholds + features)
            threshold_strategy: One of "balanced_F1", "high_recall_F2", "high_precision_F0.5"
        """
        self.model = CatBoostClassifier()
        self.model.load_model(str(Path(model_path)))

        with open(bundle_path) as f:
            self.bundle = json.load(f)

        self.features = self.bundle["features"]
        self.threshold_strategy = threshold_strategy
        self.threshold = self.bundle["thresholds"][threshold_strategy]
        self.horizon_min = self.bundle["horizon_min"]

    def predict(self, df_recent_window: pd.DataFrame) -> dict:
        """
        Args:
            df_recent_window: DataFrame in the standard Telnet collector schema.
                              Should contain at least the last 30 minutes of telemetry
                              (≈150 rows at 12s sampling).
        Returns:
            dict with prediction, probability, confidence_level, top_features, etc.
        """
        try:
            df_w = df_recent_window.copy().sort_values("timestamp").reset_index(drop=True)
            df_w["gap_to_prev"] = df_w["timestamp"].diff().dt.total_seconds().fillna(0)
            df_w["session_id"] = (df_w["gap_to_prev"] > 300).cumsum()

            df_std = map_real_to_standard(df_w)
            df_1m = resample_per_session(df_std, "1min")

            if len(df_1m) < 30:
                return {
                    "prediction": None,
                    "error": f"Not enough data after resampling: got {len(df_1m)} min, need 30+",
                }

            df_f = build_features(df_1m)
            last_row = df_f.iloc[[-1]][self.features]

            if last_row.isna().any().any():
                return {"prediction": None, "error": "NaN in features after engineering"}

            prob = float(self.model.predict_proba(last_row)[0, 1])
            pred = int(prob >= self.threshold)

            if prob < 0.30:
                confidence = "LOW_RISK"
            elif prob < self.threshold:
                confidence = "WATCH"
            elif prob < 0.85:
                confidence = "INCIDENT_LIKELY"
            else:
                confidence = "INCIDENT_VERY_LIKELY"

            shap_v = self.model.get_feature_importance(Pool(last_row), type="ShapValues")[0, :-1]
            contribs = list(zip(self.features, shap_v, last_row.iloc[0].values))
            contribs.sort(key=lambda x: -abs(x[1]))
            top = [
                {
                    "feature": f,
                    "value": float(v),
                    "shap": float(s),
                    "direction": "increases_risk" if s > 0 else "decreases_risk",
                }
                for f, s, v in contribs[:5]
            ]

            return {
                "prediction": pred,
                "probability": round(prob, 4),
                "confidence_level": confidence,
                "threshold_used": round(self.threshold, 4),
                "threshold_strategy": self.threshold_strategy,
                "horizon_min": self.horizon_min,
                "top_features": top,
                "timestamp": df_w["timestamp"].iloc[-1].isoformat(),
            }
        except Exception as e:
            return {"prediction": None, "error": str(e)}


# =============================================================================
# CLI usage example
# =============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict_incident_prod.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    predictor = IncidentPredictor()

    # Take the last 60 minutes worth of data as the "recent window"
    cutoff = df["timestamp"].max() - pd.Timedelta(minutes=60)
    df_recent = df[df["timestamp"] >= cutoff]

    result = predictor.predict(df_recent)
    print(json.dumps(result, indent=2, default=str))
