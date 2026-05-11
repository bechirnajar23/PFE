import json
import os
import smtplib
import subprocess
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from test_models import load_all_models, normalize_input_dataframe, predict_at_timestamp


DB_URL = os.getenv("DB_DSN", "postgresql://hgw_user:hgw_password@timescaledb:5432/hgw_monitoring")
PREDICTION_INTERVAL_SECONDS = int(os.getenv("PREDICTION_INTERVAL_SECONDS", "300"))
PREDICTION_LOOKBACK_HOURS = int(os.getenv("PREDICTION_LOOKBACK_HOURS", "25"))
PREDICTION_FALLBACK_ROWS = int(os.getenv("PREDICTION_FALLBACK_ROWS", "20000"))
MIN_ROWS_FOR_PREDICTION = int(os.getenv("MIN_ROWS_FOR_PREDICTION", "60"))

RETRAIN_ENABLED = os.getenv("RETRAIN_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
RETRAIN_INTERVAL_DAYS = int(os.getenv("RETRAIN_INTERVAL_DAYS", "7"))
RETRAIN_RUN_ON_START = os.getenv("RETRAIN_RUN_ON_START", "false").strip().lower() in {"1", "true", "yes", "on"}
RETRAIN_COMMAND = os.getenv("RETRAIN_COMMAND", "python train_multi_horizon.py")
RETRAIN_STATE_FILE = Path(os.getenv("RETRAIN_STATE_FILE", "/app/data/retrain_state.json"))

ALERT_EMAIL_TO = [x.strip() for x in os.getenv("ALERT_EMAIL_TO", "").split(",") if x.strip()]
ALERT_COOLDOWN_MINUTES = int(os.getenv("ALERT_COOLDOWN_MINUTES", "60"))
ALERT_STATE_FILE = Path(os.getenv("ALERT_STATE_FILE", "/app/data/alert_state.json"))

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "hgw-alerts@example.com").strip()

MODEL_VERSION = os.getenv("MODEL_VERSION", "continuous_5min_weekly_retrain")
NOTIFY_STATUSES = {"URGENT", "CRITICAL"}


engine = create_engine(DB_URL, pool_pre_ping=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_state(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Cannot read state {path}: {exc}")
    return {}


def save_state(path: Path, state: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] Cannot write state {path}: {exc}")


def ensure_predictions_schema() -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS predictions_log (
        timestamp TIMESTAMPTZ NOT NULL,
        gateway_id TEXT DEFAULT 'HGW_001',
        horizon TEXT,
        horizon_min INTEGER,
        probability DOUBLE PRECISION,
        threshold DOUBLE PRECISION,
        alert BOOLEAN,
        decision_level TEXT,
        decision_message TEXT,
        decision_explanation TEXT,
        explainer_json JSONB,
        model_version TEXT,
        predictions_json JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS gateway_id TEXT DEFAULT 'HGW_001';
    ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS horizon TEXT;
    ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS horizon_min INTEGER;
    ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS probability DOUBLE PRECISION;
    ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS threshold DOUBLE PRECISION;
    ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS alert BOOLEAN;
    ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS decision_level TEXT;
    ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS decision_message TEXT;
    ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS decision_explanation TEXT;
    ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS explainer_json JSONB;
    ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS model_version TEXT;
    ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS predictions_json JSONB;
    ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
    CREATE INDEX IF NOT EXISTS idx_predictions_log_time ON predictions_log (timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_predictions_log_alert ON predictions_log (alert, timestamp DESC);
    """
    with engine.begin() as conn:
        for stmt in [part.strip() for part in ddl.split(";") if part.strip()]:
            conn.execute(text(stmt))


def fetch_recent_snapshots() -> pd.DataFrame:
    since = utc_now() - timedelta(hours=PREDICTION_LOOKBACK_HOURS)
    query = text(
        """
        SELECT *
        FROM monitor_snapshots
        WHERE timestamp >= :since
        ORDER BY timestamp ASC
        """
    )
    df = pd.read_sql(query, engine, params={"since": since})
    if len(df) >= MIN_ROWS_FOR_PREDICTION:
        return df

    fallback_query = text(
        """
        SELECT *
        FROM monitor_snapshots
        ORDER BY timestamp DESC
        LIMIT :limit
        """
    )
    df = pd.read_sql(fallback_query, engine, params={"limit": PREDICTION_FALLBACK_ROWS})
    return df.sort_values("timestamp").reset_index(drop=True)


def get_gateway_id(df_raw: pd.DataFrame) -> str:
    for col in ["gateway_id", "serial_number", "SERIAL_NUMBER"]:
        if col in df_raw.columns and df_raw[col].notna().any():
            return str(df_raw[col].dropna().iloc[-1])
    return "HGW_001"


def decision_from_results(results: dict) -> tuple[str, str, str | None]:
    business = results.get("business_alert") or {}
    if business.get("alert") and business.get("level") == "CRITICAL":
        return "CRITICAL", business.get("message", "Etat critique detecte"), "business"

    alerts = [name for name, pred in results.get("predictions", {}).items() if pred.get("alert")]
    if alerts:
        return "PREDICTED_INCIDENT", f"Incident predit ({alerts[0]})", alerts[0]

    if business.get("alert"):
        return business.get("level", "WARNING"), business.get("message", "Surveillance active"), "business"

    return "OK", "Systeme stable", None


def current_status_from_row(row) -> str:
    for key in ["local_status", "LOCAL_STATUS", "risk_level"]:
        if key in row and pd.notna(row.get(key)):
            return str(row.get(key)).upper()
    return "NORMAL"


def alert_explanation_from_row(row, results: dict) -> tuple[str, dict | None]:
    for key in ["alert_explanation", "ALERT_EXPLANATION"]:
        if key in row and pd.notna(row.get(key)) and str(row.get(key)).strip():
            explanation = str(row.get(key)).strip()
            break
    else:
        explanation = (results.get("business_alert") or {}).get("message", "Etat urgent ou critique detecte")

    for key in ["alert_explainer_json", "ALERT_EXPLAINER_JSON"]:
        if key in row and pd.notna(row.get(key)) and str(row.get(key)).strip():
            if isinstance(row.get(key), dict):
                return explanation, row.get(key)
            try:
                return explanation, json.loads(str(row.get(key)))
            except Exception:
                return explanation, None

    return explanation, None


def build_explainer_payload(results: dict, latest_row, horizon: str | None) -> dict | None:
    if latest_row is not None:
        business_explanation, business_payload = alert_explanation_from_row(latest_row, results)
    else:
        business_explanation = (results.get("business_alert") or {}).get("message", "")
        business_payload = None

    prediction = (results.get("predictions") or {}).get(horizon or "", {})
    shap_payload = prediction.get("shap")
    payload = {
        "type": "hgw_xai",
        "horizon": horizon,
        "business_explanation": business_explanation,
        "business": business_payload,
        "shap": shap_payload,
    }
    if shap_payload:
        payload["summary"] = shap_payload.get("summary")
    return payload


def shap_hint_from_results(results: dict, preferred_horizon: str | None = None) -> tuple[str | None, dict | None]:
    predictions = results.get("predictions") or {}
    if preferred_horizon in predictions and predictions[preferred_horizon].get("shap"):
        shap_payload = predictions[preferred_horizon]["shap"]
        return shap_payload.get("summary"), shap_payload

    ranked = sorted(
        predictions.items(),
        key=lambda item: float(item[1].get("probability") or 0),
        reverse=True,
    )
    for _horizon, prediction in ranked:
        if prediction.get("shap"):
            shap_payload = prediction["shap"]
            return shap_payload.get("summary"), shap_payload
    return None, None


def save_predictions(results: dict, gateway_id: str, latest_row=None) -> None:
    decision_level, decision_message, _source = decision_from_results(results)
    if latest_row is not None:
        decision_explanation, explainer_json = alert_explanation_from_row(latest_row, results)
    else:
        decision_explanation, explainer_json = (results.get("business_alert") or {}).get("message", ""), None
    predictions_json = json.dumps(results, ensure_ascii=True, default=str)
    timestamp = pd.to_datetime(results["timestamp"]).to_pydatetime()
    predictions = dict(results.get("predictions", {}))

    if "3 jours" not in predictions:
        predictions["3 jours"] = {
            "horizon_min": 3 * 24 * 60,
            "probability": None,
            "threshold": None,
            "alert": False,
            "status": "unavailable",
            "error": results.get("dl_error", "LSTM 3 jours non disponible pour ce cycle"),
        }

    with engine.begin() as conn:
        for horizon, pred in predictions.items():
            probability = pred.get("probability")
            threshold = pred.get("threshold")
            horizon_explainer = build_explainer_payload(results, latest_row, horizon)
            explainer_json_text = (
                json.dumps(horizon_explainer, ensure_ascii=True, default=str)
                if horizon_explainer
                else None
            )
            conn.execute(
                text(
                    """
                    INSERT INTO predictions_log (
                        timestamp, gateway_id, horizon, horizon_min, probability,
                        threshold, alert, decision_level, decision_message,
                        decision_explanation, explainer_json, model_version, predictions_json
                    )
                    VALUES (
                        :timestamp, :gateway_id, :horizon, :horizon_min, :probability,
                        :threshold, :alert, :decision_level, :decision_message,
                        :decision_explanation, CAST(:explainer_json AS JSONB),
                        :model_version, CAST(:predictions_json AS JSONB)
                    )
                    """
                ),
                {
                    "timestamp": timestamp,
                    "gateway_id": gateway_id,
                    "horizon": horizon,
                    "horizon_min": int(pred.get("horizon_min", 0)),
                    "probability": None if probability is None else float(probability),
                    "threshold": None if threshold is None else float(threshold),
                    "alert": bool(pred.get("alert", False)),
                    "decision_level": decision_level,
                    "decision_message": decision_message,
                    "decision_explanation": decision_explanation,
                    "explainer_json": explainer_json_text,
                    "model_version": MODEL_VERSION,
                    "predictions_json": predictions_json,
                },
            )


def should_notify(results: dict, gateway_id: str, latest_row) -> bool:
    current_status = current_status_from_row(latest_row)
    if current_status not in NOTIFY_STATUSES:
        return False

    decision_level, _message, source = decision_from_results(results)
    if decision_level == "OK":
        return False

    state = load_state(ALERT_STATE_FILE)
    key = f"{gateway_id}:{decision_level}:{source or 'business'}"
    last_sent_raw = state.get(key)
    if last_sent_raw:
        try:
            last_sent = datetime.fromisoformat(last_sent_raw)
            if utc_now() - last_sent < timedelta(minutes=ALERT_COOLDOWN_MINUTES):
                return False
        except ValueError:
            pass

    state[key] = utc_now().isoformat()
    save_state(ALERT_STATE_FILE, state)
    return True


def send_email_alert(subject: str, body: str) -> None:
    if not ALERT_EMAIL_TO or not SMTP_HOST:
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = ", ".join(ALERT_EMAIL_TO)
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
            smtp.starttls()
            if SMTP_USER and SMTP_PASSWORD:
                smtp.login(SMTP_USER, SMTP_PASSWORD)
            smtp.send_message(msg)
        print(f"[ALERT] Email sent to {', '.join(ALERT_EMAIL_TO)}")
    except Exception as exc:
        print(f"[WARN] Email alert failed: {exc}")


def notify_if_needed(results: dict, gateway_id: str, latest_row) -> None:
    if not should_notify(results, gateway_id, latest_row):
        return

    decision_level, decision_message, source = decision_from_results(results)
    current_status = current_status_from_row(latest_row)
    explanation, explainer_json = alert_explanation_from_row(latest_row, results)
    payload = {
        "gateway_id": gateway_id,
        "level": current_status,
        "message": decision_message,
        "explanation": explanation,
        "horizon": source,
        "timestamp": results.get("timestamp"),
    }
    if explainer_json:
        payload["explainer"] = explainer_json

    if source and source in results.get("predictions", {}):
        payload["probability"] = results["predictions"][source].get("probability")

    shap_summary, shap_payload = shap_hint_from_results(results, source)
    if shap_summary:
        payload["xai"] = shap_summary
    if shap_payload:
        payload["shap"] = {
            "horizon": shap_payload.get("horizon"),
            "top_features": shap_payload.get("top_features", [])[:3],
        }

    subject = f"[HGW] {current_status} - {gateway_id}"
    body = (
        f"Gateway: {gateway_id}\n"
        f"Etat: {current_status}\n"
        f"Message: {decision_message}\n"
        f"Explication: {explanation}\n"
        f"XAI: {shap_summary or 'N/A'}\n"
        f"Source: {source or 'business'}\n"
        f"Timestamp: {results.get('timestamp')}\n"
    )
    send_email_alert(subject, body)


def retrain_due() -> bool:
    if not RETRAIN_ENABLED:
        return False

    state = load_state(RETRAIN_STATE_FILE)
    last_run_raw = state.get("last_retrain_at")
    if not last_run_raw:
        if RETRAIN_RUN_ON_START:
            return True
        state["last_retrain_at"] = utc_now().isoformat()
        save_state(RETRAIN_STATE_FILE, state)
        return False

    try:
        last_run = datetime.fromisoformat(last_run_raw)
    except ValueError:
        return True

    return utc_now() - last_run >= timedelta(days=RETRAIN_INTERVAL_DAYS)


def run_retraining() -> None:
    print(f"[TRAIN] Weekly retraining started: {RETRAIN_COMMAND}")
    started = utc_now()
    state = load_state(RETRAIN_STATE_FILE)
    state["last_attempt_at"] = started.isoformat()
    save_state(RETRAIN_STATE_FILE, state)

    try:
        completed = subprocess.run(
            RETRAIN_COMMAND,
            shell=True,
            cwd=Path(__file__).resolve().parent,
            check=False,
            text=True,
        )
        state = load_state(RETRAIN_STATE_FILE)
        state["last_attempt_at"] = started.isoformat()
        state["last_exit_code"] = completed.returncode
        if completed.returncode == 0:
            state["last_retrain_at"] = utc_now().isoformat()
            print("[TRAIN] Weekly retraining completed successfully")
        else:
            print(f"[WARN] Weekly retraining failed with code {completed.returncode}")
        save_state(RETRAIN_STATE_FILE, state)
    except Exception as exc:
        state = load_state(RETRAIN_STATE_FILE)
        state["last_error"] = str(exc)
        save_state(RETRAIN_STATE_FILE, state)
        print(f"[WARN] Weekly retraining error: {exc}")


def run_prediction_cycle(models: dict) -> None:
    df_db = fetch_recent_snapshots()
    if len(df_db) < MIN_ROWS_FOR_PREDICTION:
        print(f"[PREDICT] Waiting data: {len(df_db)}/{MIN_ROWS_FOR_PREDICTION} rows")
        return

    df_raw = normalize_input_dataframe(df_db)
    target_ts = df_raw["timestamp"].max()
    gateway_id = get_gateway_id(df_db)
    latest_row = df_db.sort_values("timestamp").iloc[-1]
    results = predict_at_timestamp(df_raw, target_ts, models)

    if "error" in results:
        print(f"[WARN] Prediction skipped: {results['error']}")
        return

    save_predictions(results, gateway_id, latest_row)
    notify_if_needed(results, gateway_id, latest_row)

    decision_level, decision_message, _source = decision_from_results(results)
    print(
        f"[PREDICT] {target_ts} gateway={gateway_id} "
        f"decision={decision_level} message={decision_message}"
    )


def main() -> None:
    print("[START] HGW continuous prediction service")
    print(f"[CONFIG] prediction_interval={PREDICTION_INTERVAL_SECONDS}s")
    print(f"[CONFIG] retrain_enabled={RETRAIN_ENABLED} interval={RETRAIN_INTERVAL_DAYS}d")

    ensure_predictions_schema()
    models = load_all_models()

    while True:
        cycle_started = time.time()
        try:
            run_prediction_cycle(models)
            if retrain_due():
                run_retraining()
                models = load_all_models()
        except Exception as exc:
            print(f"[ERROR] Service cycle failed: {exc}")

        elapsed = time.time() - cycle_started
        sleep_for = max(5, PREDICTION_INTERVAL_SECONDS - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
