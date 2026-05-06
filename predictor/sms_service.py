import os
from typing import List

from flask import Flask, jsonify, request
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client


app = Flask(__name__)


def _read_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def _parse_recipients(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


TWILIO_ACCOUNT_SID = _read_env("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = _read_env("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = _read_env("TWILIO_NUMBER")
DEFAULT_RECIPIENTS = _parse_recipients(_read_env("ALERT_PHONE_TO"))
SMS_HOST = _read_env("SMS_SERVICE_HOST", "0.0.0.0")
SMS_PORT = int(_read_env("SMS_SERVICE_PORT", "5000"))


def _has_twilio_config() -> bool:
    return all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_NUMBER])


def _build_client() -> Client | None:
    if not _has_twilio_config():
        return None
    return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def _format_message(payload: dict) -> str:
    gateway_id = payload.get("gateway_id", "HGW")
    level = str(payload.get("level", "ALERTE")).upper()
    horizon = payload.get("horizon")
    probability = payload.get("probability")
    message = payload.get("message") or "Incident detecte ou predit"
    explanation = payload.get("explanation")
    timestamp = payload.get("timestamp")

    parts = [f"[HGW] {level}", f"Gateway: {gateway_id}", f"Message: {message}"]

    if explanation:
        parts.append(f"Pourquoi: {explanation}")
    if horizon:
        parts.append(f"Horizon: {horizon}")
    if probability is not None:
        try:
            parts.append(f"Probabilite: {float(probability):.1%}")
        except (TypeError, ValueError):
            parts.append(f"Probabilite: {probability}")
    if timestamp:
        parts.append(f"Date: {timestamp}")

    return " | ".join(parts)


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "twilio_configured": _has_twilio_config(),
            "default_recipients": DEFAULT_RECIPIENTS,
            "sender": TWILIO_NUMBER or None,
        }
    )


@app.post("/sms-alert")
def sms_alert():
    payload = request.get_json(silent=True) or {}

    recipients = payload.get("to")
    if isinstance(recipients, str):
        recipients = [recipients]
    elif not isinstance(recipients, list):
        recipients = DEFAULT_RECIPIENTS

    recipients = [str(item).strip() for item in recipients if str(item).strip()]
    if not recipients:
        return jsonify({"status": "error", "message": "No destination phone numbers configured"}), 400

    client = _build_client()
    if client is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Twilio is not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_NUMBER.",
                }
            ),
            500,
        )

    body = payload.get("body")
    if not body:
        body = _format_message(payload)

    sent = []
    failed = []

    for recipient in recipients:
        try:
            result = client.messages.create(body=body, from_=TWILIO_NUMBER, to=recipient)
            sent.append({"to": recipient, "sid": result.sid, "status": result.status})
        except TwilioRestException as exc:
            failed.append({"to": recipient, "error": str(exc)})
        except Exception as exc:
            failed.append({"to": recipient, "error": str(exc)})

    status_code = 200 if sent else 502
    return (
        jsonify(
            {
                "status": "sent" if sent else "error",
                "message": body,
                "sent_count": len(sent),
                "failed_count": len(failed),
                "sent": sent,
                "failed": failed,
            }
        ),
        status_code,
    )


if __name__ == "__main__":
    app.run(host=SMS_HOST, port=SMS_PORT)
