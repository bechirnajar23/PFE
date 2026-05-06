"""
HGW Predictive Service — DB-backed
====================================
Reads recent telemetry from TimescaleDB, runs both models,
writes predictions back into hgw_predictions.

Run as a cron job or systemd timer (every 5 minutes recommended):

    python predict_db.py --gateway-id HGW_001
    python predict_db.py --loop --interval 300

Models loaded:
  - data/catboost_24h.cbm        (short-term, 24h)
  - data/bilstm_72h.keras        (long-term, 72h)
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("predict_db")

try:
    import psycopg2
    import psycopg2.extras
    HAS_PG = True
except ImportError:
    HAS_PG = False


# Reuse feature engineering + health score from training script
sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_health_score(cpu, mem, ping, loss, ttf_pred=None):
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


def build_catboost_features(df, sph=1):
    """Build features matching the training script (must stay in sync)."""
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
        d["cpu_x_mem"]      = g["cpu_load"] * g["mem_used_pct"] / 10000
        d["saturation_idx"] = (g["cpu_load"]/88 + g["mem_used_pct"]/90) / 2
        d["mem_headroom"]   = np.clip(90.0 - g["mem_used_pct"], 0, 90)
        d["cwmp_share_mem"] = g["cwmp_rss_mb"] / 936.0
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
        cats = g[["gateway_id", "firmware", "region", "isp"]] if "region" in g.columns \
                else g[["gateway_id", "firmware"]].assign(region="UNK", isp="UNK")
        existing = g[["cpu_mean_24h","ram_mean_24h","cpu_std_24h","ram_std_24h",
                       "cpu_slope_6h","ram_slope_6h","wan_instability_6h",
                       "cwmp_rss_mb","dhcp_rss_mb","nemo_rss_mb",
                       "cpu_load","mem_used_pct","ping_latency","packet_loss"]]
        # Add missing eng cols if not in DB
        for col in ["reboot_event","recovery_phase"]:
            if col not in g.columns:
                g[col] = 0
        existing = pd.concat([existing, g[["reboot_event","recovery_phase"]]], axis=1)
        out.append(pd.concat([cats, existing, feats], axis=1))
    return pd.concat(out).fillna(0)


def fetch_recent_telemetry(conn, gateway_id, hours=72):
    """Fetch last N hours of telemetry from the DB and add derived columns."""
    sql = """
        SELECT timestamp, gateway_id, firmware,
               cpu_load, mem_used_pct, ping_latency, packet_loss, wan_status,
               cwmp_rss_mb, dhcp_rss_mb, nemo_rss_mb,
               EXTRACT(HOUR FROM timestamp) AS hour,
               EXTRACT(DOW  FROM timestamp) AS dow
          FROM hgw_telemetry
         WHERE gateway_id = %s
           AND timestamp >= NOW() - INTERVAL %s
         ORDER BY timestamp
    """
    df = pd.read_sql(sql, conn, params=(gateway_id, f"{hours} hours"))
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Add region/isp if missing in DB
    if "region" not in df.columns:
        df["region"] = "UNK"
    if "isp" not in df.columns:
        df["isp"] = "UNK"

    # Compute the rolling features used by the model (per-gateway)
    sph = 1  # assume hourly buckets after collector smoothing
    df["cpu_mean_24h"] = df["cpu_load"].rolling(24, min_periods=1).mean()
    df["ram_mean_24h"] = df["mem_used_pct"].rolling(24, min_periods=1).mean()
    df["cpu_std_24h"]  = df["cpu_load"].rolling(24, min_periods=1).std().fillna(0)
    df["ram_std_24h"]  = df["mem_used_pct"].rolling(24, min_periods=1).std().fillna(0)
    df["cpu_slope_6h"] = df["cpu_load"].diff(6) / 6
    df["ram_slope_6h"] = df["mem_used_pct"].diff(6) / 6
    df["wan_instability_6h"] = df["wan_status"].eq(0).rolling(6, min_periods=1).mean()
    df["health_score"] = compute_health_score(df["cpu_load"], df["mem_used_pct"],
                                                 df["ping_latency"], df["packet_loss"])
    return df.fillna(0)


def insert_predictions(conn, df_preds):
    """Write predictions to hgw_predictions."""
    rows = [
        (
            r["timestamp"], r["gateway_id"], r["model_version"],
            r["prob_incident_24h"], r["alert_24h"],
            r.get("prob_incident_72h"), r.get("alert_72h"),
            r.get("ttf_hours_pred"), r["health_score"], r["risk_level"],
            json.dumps(r.get("top_reasons", [])),
            r.get("mc_dropout_std"),
        )
        for _, r in df_preds.iterrows()
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO hgw_predictions
               (timestamp, gateway_id, model_version,
                prob_incident_24h, alert_24h,
                prob_incident_72h, alert_72h,
                ttf_hours_pred, health_score, risk_level,
                top_reasons, mc_dropout_std)
               VALUES %s
               ON CONFLICT (timestamp, gateway_id) DO UPDATE SET
                  prob_incident_24h = EXCLUDED.prob_incident_24h,
                  alert_24h         = EXCLUDED.alert_24h,
                  health_score      = EXCLUDED.health_score,
                  risk_level        = EXCLUDED.risk_level,
                  top_reasons       = EXCLUDED.top_reasons""",
            rows
        )
    conn.commit()


def predict_for_gateway(conn, gateway_id, cb_model, cb_threshold,
                          cb_features_order, cb_top_features, model_version):
    df = fetch_recent_telemetry(conn, gateway_id, hours=72)
    if df.empty:
        log.warning(f"No recent telemetry for {gateway_id}")
        return None

    # CatBoost prediction on the latest row
    feats = build_catboost_features(df.tail(72), sph=1)  # last 72h
    for col in cb_features_order:
        if col not in feats.columns:
            feats[col] = 0
    X = feats[cb_features_order]
    prob_24h = cb_model.predict_proba(X)[:, 1]
    alert_24h = (prob_24h >= cb_threshold).astype(int)

    # Top reasons (latest row)
    last_feats = feats.iloc[-1]
    top_reasons = []
    for f in cb_top_features[:5]:
        if f in last_feats.index:
            top_reasons.append({"feature": f, "value": float(last_feats[f])})

    # Health score — use latest reading
    health = float(df.iloc[-1]["health_score"])

    # Risk level
    p_now = float(prob_24h[-1])
    if   p_now >= 0.6:  risk = "CRITICAL"
    elif p_now >= 0.3:  risk = "HIGH"
    elif p_now >= 0.1:  risk = "MEDIUM"
    else:               risk = "LOW"

    pred = {
        "timestamp":         df.iloc[-1]["timestamp"],
        "gateway_id":        gateway_id,
        "model_version":     model_version,
        "prob_incident_24h": float(p_now),
        "alert_24h":         int(alert_24h[-1]),
        "prob_incident_72h": None,
        "alert_72h":         None,
        "ttf_hours_pred":    None,
        "health_score":      health,
        "risk_level":        risk,
        "top_reasons":       top_reasons,
        "mc_dropout_std":    None,
    }
    return pd.DataFrame([pred])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dsn",     default=os.getenv("DB_DSN",
                          "postgresql://hgw_admin:changeme@localhost:5432/hgw"))
    parser.add_argument("--cb-model",   default="data/catboost_24h.cbm")
    parser.add_argument("--cb-meta",    default="data/catboost_24h_metadata.json")
    parser.add_argument("--gateway-id", default=None,
                          help="Predict for one gateway. Omit to predict for all.")
    parser.add_argument("--loop",       action="store_true")
    parser.add_argument("--interval",   type=int, default=300)
    parser.add_argument("--model-version", default="catboost_24h_v1")
    args = parser.parse_args()

    if not HAS_PG:
        log.error("pip install psycopg2-binary")
        sys.exit(1)

    # Load CatBoost model
    from catboost import CatBoostClassifier
    cb_model = CatBoostClassifier()
    cb_model.load_model(args.cb_model)
    log.info(f"Loaded CatBoost: {args.cb_model}")

    cb_threshold = 0.5
    cb_top_features = []
    if Path(args.cb_meta).exists():
        with open(args.cb_meta) as f:
            meta = json.load(f)
        cb_threshold = meta["metrics"].get("threshold", 0.5)
        cb_top_features = list(meta.get("top15_features_gain", {}).keys())
    log.info(f"Threshold: {cb_threshold:.4f}  Top features: {cb_top_features[:3]}")

    cb_features_order = list(cb_model.feature_names_)

    conn = psycopg2.connect(args.db_dsn)
    log.info(f"Connected: {args.db_dsn.split('@')[-1]}")

    def run_once():
        if args.gateway_id:
            gateways = [args.gateway_id]
        else:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT gateway_id FROM hgw_telemetry "
                              "WHERE timestamp >= NOW() - INTERVAL '6 hours'")
                gateways = [r[0] for r in cur.fetchall()]

        for gw in gateways:
            try:
                df_pred = predict_for_gateway(
                    conn, gw, cb_model, cb_threshold,
                    cb_features_order, cb_top_features, args.model_version
                )
                if df_pred is not None:
                    insert_predictions(conn, df_pred)
                    r = df_pred.iloc[0]
                    log.info(f"  {gw}: p24h={r['prob_incident_24h']:.3f}  "
                              f"health={r['health_score']:.0f}%  risk={r['risk_level']}")
            except Exception as e:
                log.exception(f"{gw} prediction failed: {e}")

    run_once()
    while args.loop:
        time.sleep(args.interval)
        run_once()

    conn.close()


if __name__ == "__main__":
    main()
