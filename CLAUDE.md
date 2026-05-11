# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HGW Predictive Maintenance — a full ML/DL pipeline that collects telemetry from a residential gateway (via Telnet), stores it in TimescaleDB, runs multi-horizon risk predictions (CatBoost + LSTM), and surfaces results via Grafana dashboards and email alerts.

## Common Commands

### Local development (PowerShell or WSL)

```powershell
# Setup
conda create -n hgw_project python=3.10 -y
conda activate hgw_project
pip install -r requirements.txt

# Test models on CSV (primary development workflow)
cd predictor
python test_models.py "..\data\test_full_scenarios.csv" --last
python test_models.py "..\data\test_full_scenarios.csv" --random 10
python test_models.py "..\data\test_full_scenarios.csv" --urgents 5
python test_models.py "..\data\test_full_scenarios.csv" --time "2026-05-01 08:20:00"

# Export predictions for notebook analysis
python test_models.py "..\data\test_full_scenarios.csv" --random 10 --export "..\data\predictions_now.csv"

# Retrain CatBoost models
python train_multi_horizon.py
```

### Docker (production stack)

```bash
# Start everything
docker compose up --build -d

# Follow logs per service
docker logs -f hgw_collector
docker logs -f hgw_predictor

# Apply additional SQL schemas manually
docker exec -i hgw_timescaledb psql -U hgw_user -d hgw_monitoring < sql/schema.sql

# Inspect predictions table
docker exec -it hgw_timescaledb psql -U hgw_user -d hgw_monitoring \
  -c "SELECT timestamp, horizon, probability, alert FROM predictions_log ORDER BY timestamp DESC LIMIT 10;"

# Teardown (add -v to also wipe TimescaleDB data)
docker compose down
```

### Services and ports

| Container | Port | Purpose |
|-----------|------|---------|
| `hgw_timescaledb` | 5432 | TimescaleDB |
| `hgw_predictor` | — | Prediction daemon (5-min cycles) |
| `hgw_collector` | — | Telnet collector |
| `hgw_backend_api` | 8000 | FastAPI REST |
| `hgw_grafana` | 3000 | Dashboards (admin/admin) |
| `hgw_frontend` | 8080 | React UI |

## Architecture

```
HGW (Telnet) → collector/ → TimescaleDB → predictor/ → Grafana / Email / Frontend
```

### Critical data flow in `predictor/`

`predict_service.py` is the production daemon. It **imports directly from `test_models.py`** — the offline test script doubles as the prediction engine:

```python
from test_models import load_all_models, normalize_input_dataframe, predict_at_timestamp
```

Every 5 minutes it calls `predict_at_timestamp()`, stores results in `predictions_log`, and triggers email alerts for `URGENT`/`CRITICAL` states only (60-min cooldown).

### Feature engineering pipeline

All incoming data (CSV or DB snapshots) passes through this chain:

1. `normalize_input_dataframe()` — flexible schema normalization, accepts many column naming conventions
2. `map_real_to_standard()` — maps normalized columns to the canonical 13-column ML format
3. `build_ml_features()` — computes 43 CatBoost features (rolling stats, lags, slopes, cyclical hour, interactions)
4. `build_dl_features()` — computes 13 LSTM features resampled to 30-min granularity

### Two-tier model system

**CatBoost (short-term)** — 4 independent models, loaded from `predictor/multi_horizon/`:
- Horizons: 15min, 30min, 60min, 360min
- Thresholds and feature lists are defined in `multi_horizon_bundle.json` (source of truth)
- SHAP explanations computed per prediction via `explain_catboost_prediction()`

**LSTM (long-term)** — single Bi-LSTM for 3-day ahead prediction:
- Requires **minimum 24 rows at 30-min resolution (12h of real data)** before running — below this threshold, `dl_error` is set and LSTM is skipped (no padding)
- With 24–47 rows: uses last 24 rows directly → `(1, 24, 13)`
- With 48+ rows: takes last 48 rows, subsamples by 2 → `(1, 24, 13)`
- Trained on synthetic 5-year data; fine-tuning with real data planned when 30+ days are collected

### Alert logic

`predict_service.py → should_notify()` only fires when `LOCAL_STATUS ∈ {URGENT, CRITICAL}`. Email is sent via SMTP; the SMS service was removed. State between alerts is persisted in `ALERT_STATE_FILE` (JSON).

### Business rules layer

`critical_state_detector.py` applies rule-based checks **independently** of ML predictions:
- CPU ≥ 90% or MEM ≥ 95% → CRITICAL
- CPU ≥ 85% + MEM ≥ 90% → CRITICAL
- Status field = CRITICAL/pre_crash → CRITICAL

These run alongside ML and the most severe result wins.

## Configuration

All runtime config lives in `.env` (copied from `.env.example`). Key variables:

```env
# HGW connection
HGW_HOST=192.168.1.1
HGW_PASSWORD=...

# Prediction daemon
PREDICTION_INTERVAL_SECONDS=300
PREDICTION_LOOKBACK_HOURS=24
MIN_ROWS_FOR_PREDICTION=60

# Retraining
RETRAIN_ENABLED=true
RETRAIN_INTERVAL_DAYS=7

# Email alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=...
SMTP_PASSWORD=...       # Gmail: use App Password (16 chars)
ALERT_EMAIL_TO=...
ALERT_COOLDOWN_MINUTES=60
```

## Key Files

| File | Role |
|------|------|
| `predictor/test_models.py` | Core prediction engine + offline CLI tester |
| `predictor/predict_service.py` | Production daemon (imports from test_models.py) |
| `predictor/train_multi_horizon.py` | CatBoost training with 5-fold CV |
| `predictor/multi_horizon/multi_horizon_bundle.json` | Model thresholds and feature lists |
| `predictor/long_horizon_dl/lstm_metdata.json` | LSTM threshold and metadata |
| `collector/data_collection.py` | Telnet collection loop |
| `sql/init.sql` | Primary schema (auto-applied on first TimescaleDB start) |
| `notebooks/Viz.ipynb` | Visualization of exported predictions CSV |

## Known Limitations

- `cwmp_rss_mb`, `dhcp_rss_mb`, `nemo_rss_mb` are stubbed to 0.0 — Telnet collector does not extract them yet
- LSTM was trained on synthetic data; predictions are directionally valid but less calibrated than CatBoost until 30+ days of real data are accumulated
- `corbeille/` and `scripts/archive/` contain obsolete files — ignore them
- On corporate/university networks, Docker image pulls may fail with TLS certificate errors; fix by exporting the corporate CA from Windows and adding it to WSL2 trust store
