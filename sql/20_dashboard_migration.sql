-- Migration idempotente pour Grafana, predictions et alertes.
-- A executer sur une base deja creee si le volume TimescaleDB existe avant les derniers changements.

CREATE EXTENSION IF NOT EXISTS timescaledb;

ALTER TABLE monitor_snapshots ADD COLUMN IF NOT EXISTS alert_eligible BOOLEAN DEFAULT FALSE;
ALTER TABLE monitor_snapshots ADD COLUMN IF NOT EXISTS alert_explanation TEXT;
ALTER TABLE monitor_snapshots ADD COLUMN IF NOT EXISTS alert_explainer_json JSONB;

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
ALTER TABLE predictions_log ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

SELECT create_hypertable('predictions_log', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_predictions_log_time ON predictions_log (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_log_alert ON predictions_log (alert, timestamp DESC);

CREATE TABLE IF NOT EXISTS predictions (
    timestamp TIMESTAMPTZ,
    gateway_id TEXT,
    model_type TEXT,
    horizon TEXT,
    prediction_score FLOAT,
    prediction_label TEXT
);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hgw_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO hgw_user;
