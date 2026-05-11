import os
import json
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text


DB_DSN = os.getenv(
    "DB_DSN",
    "postgresql://hgw_user:hgw_password@timescaledb:5432/hgw_monitoring",
)
GRAFANA_PUBLIC_URL = os.getenv("GRAFANA_PUBLIC_URL", "http://localhost:3000").rstrip("/")
GRAFANA_INTERNAL_URL = os.getenv("GRAFANA_INTERNAL_URL", "http://grafana:3000").rstrip("/")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173",
    ).split(",")
    if origin.strip()
]

engine = create_engine(DB_DSN, pool_pre_ping=True)

app = FastAPI(
    title="HGW Predictive Maintenance API",
    version="1.0.0",
    description="Backend API for the HGW predictive maintenance frontend.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def scalar(query: str, default=None):
    try:
        with engine.begin() as conn:
            return conn.execute(text(query)).scalar() or default
    except Exception:
        return default


def rows(query: str, limit: int = 20):
    try:
        with engine.begin() as conn:
            result = conn.execute(text(query), {"limit": limit})
            return [dict(row._mapping) for row in result]
    except Exception:
        return []


def clean_value(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def clean_rows(items):
    return [{key: clean_value(value) for key, value in item.items()} for item in items]


def decode_json_payload(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def grafana_status():
    try:
        response = requests.get(f"{GRAFANA_INTERNAL_URL}/api/health", timeout=3)
        if response.ok:
            return {"status": "UP", "message": "Grafana disponible"}
        return {"status": "WARN", "message": f"Grafana HTTP {response.status_code}"}
    except Exception as exc:
        return {"status": "DOWN", "message": str(exc)}


@app.get("/health")
def health():
    db_ok = scalar("SELECT 1", default=None) == 1
    return {
        "status": "UP" if db_ok else "WARN",
        "api_time": datetime.now(timezone.utc).isoformat(),
        "database": "UP" if db_ok else "DOWN",
        "grafana": grafana_status(),
    }


@app.get("/api/dashboard-config")
def dashboard_config():
    return {
        "grafanaBaseUrl": GRAFANA_PUBLIC_URL,
        "dashboards": {
            "monitoring": {
                "label": "Monitoring",
                "title": "Monitoring HGW",
                "description": "Etat courant, CPU, memoire, latence, WAN et evenements metier.",
                "url": f"{GRAFANA_PUBLIC_URL}/d/hgw-monitoring/hgw-monitoring?orgId=1&from=now-30m&to=now&refresh=5s&kiosk",
                "openUrl": f"{GRAFANA_PUBLIC_URL}/d/hgw-monitoring/hgw-monitoring?orgId=1&from=now-30m&to=now&refresh=5s",
            },
            "predictions": {
                "label": "Predictions",
                "title": "Predictions multi-horizon",
                "description": "Probabilites, seuils, diagnostic LSTM 3 jours et explications.",
                "url": f"{GRAFANA_PUBLIC_URL}/d/hgw-predictions/hgw-predictions?orgId=1&from=now-6h&to=now&refresh=5s&kiosk",
                "openUrl": f"{GRAFANA_PUBLIC_URL}/d/hgw-predictions/hgw-predictions?orgId=1&from=now-6h&to=now&refresh=5s",
            },
        },
    }


@app.get("/api/summary")
def summary():
    latest = clean_rows(
        rows(
            """
            SELECT
                timestamp,
                COALESCE(local_status, 'UNKNOWN') AS local_status,
                COALESCE(status_reason, 'unknown') AS status_reason,
                cpu_usage_percent,
                mem_usage_percent,
                net_latency_ms,
                wan_state,
                alert_explanation
            FROM monitor_snapshots
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            limit=1,
        )
    )

    prediction = clean_rows(
        rows(
            """
            SELECT
                timestamp,
                COALESCE(MAX(probability), 0) AS max_probability,
                COUNT(*) FILTER (WHERE alert IS TRUE) AS active_model_alerts,
                COALESCE(MAX(decision_level), 'OK') AS decision_level,
                COALESCE(MAX(decision_message), 'Systeme stable') AS decision_message
            FROM predictions_log
            WHERE timestamp = (SELECT MAX(timestamp) FROM predictions_log)
            GROUP BY timestamp
            LIMIT 1
            """,
            limit=1,
        )
    )

    counts = {
        "snapshots": scalar("SELECT COUNT(*) FROM monitor_snapshots", 0),
        "predictions": scalar("SELECT COUNT(*) FROM predictions_log", 0),
        "critical_last_hour": scalar(
            """
            SELECT COUNT(*)
            FROM monitor_snapshots
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
              AND local_status IN ('URGENT', 'CRITICAL')
            """,
            0,
        ),
    }

    return {
        "latestSnapshot": latest[0] if latest else None,
        "latestPrediction": prediction[0] if prediction else None,
        "counts": counts,
        "services": {
            "api": {"status": "UP", "message": "Backend API disponible"},
            "database": {"status": "UP" if scalar("SELECT 1", None) == 1 else "DOWN"},
            "grafana": grafana_status(),
        },
    }


@app.get("/api/predictions/latest")
def latest_predictions(limit: int = 20):
    return {
        "items": clean_rows(
            rows(
                """
                SELECT
                    timestamp,
                    gateway_id,
                    horizon,
                    probability,
                    threshold,
                    alert,
                    decision_level,
                    decision_message,
                    decision_explanation,
                    explainer_json,
                    predictions_json->>'dl_error' AS diagnostic_dl
                FROM predictions_log
                ORDER BY timestamp DESC
                LIMIT :limit
                """,
                limit=max(1, min(limit, 100)),
            )
        )
    }


@app.get("/api/xai/latest")
def latest_xai(limit: int = 8):
    items = clean_rows(
        rows(
            """
            SELECT
                timestamp,
                gateway_id,
                horizon,
                probability,
                threshold,
                decision_level,
                decision_message,
                decision_explanation,
                explainer_json
            FROM predictions_log
            WHERE timestamp = (SELECT MAX(timestamp) FROM predictions_log)
              AND explainer_json IS NOT NULL
            ORDER BY
                CASE horizon
                    WHEN '15min' THEN 1
                    WHEN '30min' THEN 2
                    WHEN '60min' THEN 3
                    WHEN '360min' THEN 4
                    WHEN '3 jours' THEN 5
                    ELSE 9
                END
            LIMIT :limit
            """,
            limit=max(1, min(limit, 20)),
        )
    )

    visible_items = []
    for item in items:
        payload = decode_json_payload(item.get("explainer_json")) or {}
        shap_payload = payload.get("shap")
        if not shap_payload:
            continue
        item["explainer_json"] = payload
        item["business_explanation"] = payload.get("business_explanation")
        item["business"] = payload.get("business")
        item["shap"] = shap_payload
        item["xai_summary"] = payload.get("summary") or shap_payload.get("summary")
        visible_items.append(item)

    return {"items": visible_items}


@app.get("/api/series/monitoring")
def monitoring_series(limit: int = 240):
    items = rows(
        """
        SELECT
            timestamp,
            cpu_usage_percent,
            mem_usage_percent,
            net_latency_ms,
            wan_rx_rate_kbps,
            wan_tx_rate_kbps,
            local_status
        FROM monitor_snapshots
        ORDER BY timestamp DESC
        LIMIT :limit
        """,
        limit=max(10, min(limit, 1000)),
    )
    return {"items": list(reversed(clean_rows(items)))}


@app.get("/api/series/predictions")
def prediction_series(limit: int = 500):
    items = rows(
        """
        SELECT
            timestamp,
            horizon,
            probability,
            threshold,
            alert,
            decision_level
        FROM predictions_log
        ORDER BY timestamp DESC
        LIMIT :limit
        """,
        limit=max(10, min(limit, 2000)),
    )
    return {"items": list(reversed(clean_rows(items)))}


@app.get("/api/events/recent")
def recent_events(limit: int = 10):
    return {
        "items": clean_rows(
            rows(
                """
                SELECT
                    timestamp,
                    local_status,
                    status_reason,
                    cpu_usage_percent,
                    mem_usage_percent,
                    net_latency_ms,
                    alert_explanation
                FROM monitor_snapshots
                WHERE local_status IN ('URGENT', 'CRITICAL')
                   OR alert_eligible IS TRUE
                ORDER BY timestamp DESC
                LIMIT :limit
                """,
                limit=max(1, min(limit, 50)),
            )
        )
    }
