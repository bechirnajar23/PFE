-- init.sql - Initialisation de la base TimescaleDB pour HGW monitoring

-- Extension TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Table principale : snapshots de monitoring
CREATE TABLE IF NOT EXISTS monitor_snapshots (
    timestamp TIMESTAMPTZ NOT NULL,
    serial_number TEXT,
    LOCAL_STATUS TEXT,
    STATUS_REASON TEXT,
    CPU_USAGE_PERCENT INTEGER,
    MEM_USAGE_PERCENT INTEGER,
    WAN_STATE TEXT,
    WAN_TX_RATE_KBPS INTEGER,
    WAN_RX_RATE_KBPS INTEGER,
    NET_LATENCY_MS NUMERIC,
    NET_PING_STATUS TEXT,
    NET_LATENCY_AVG_5 INTEGER,
    CWMP_PROCESS_STATUS TEXT,
    DHCP_PROCESS_STATUS TEXT,
    DHCP_DATA_STATE TEXT,
    DHCP_V6_STATE TEXT,
    NEMO_PROCESS_STATUS TEXT,
    SYSTEM_UPTIME_SECONDS INTEGER,
    alert_eligible BOOLEAN DEFAULT FALSE,
    alert_explanation TEXT,
    alert_explainer_json JSONB
);

-- Convertir en hypertable (optimisée pour séries temporelles)
SELECT create_hypertable('monitor_snapshots', 'timestamp', if_not_exists => TRUE);

-- Index pour performance
CREATE INDEX IF NOT EXISTS idx_status ON monitor_snapshots (LOCAL_STATUS);
CREATE INDEX IF NOT EXISTS idx_timestamp_status ON monitor_snapshots (timestamp DESC, LOCAL_STATUS);

-- Table des prédictions
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

SELECT create_hypertable('predictions_log', 'timestamp', if_not_exists => TRUE);

-- Index pour Grafana
CREATE INDEX IF NOT EXISTS idx_predictions_alert ON predictions_log (timestamp DESC, alert);

-- Politiques de rétention (garder 90 jours max)
SELECT add_retention_policy('monitor_snapshots', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('predictions_log', INTERVAL '90 days', if_not_exists => TRUE);

-- Vue agrégée pour dashboard
CREATE OR REPLACE VIEW hourly_stats AS
SELECT
    time_bucket('1 hour', timestamp) AS hour,
    AVG(CPU_USAGE_PERCENT) AS avg_cpu,
    MAX(CPU_USAGE_PERCENT) AS max_cpu,
    AVG(MEM_USAGE_PERCENT) AS avg_mem,
    MAX(MEM_USAGE_PERCENT) AS max_mem,
    COUNT(CASE WHEN LOCAL_STATUS = 'URGENT' THEN 1 END) AS urgent_count,
    COUNT(CASE WHEN LOCAL_STATUS = 'WARNING' THEN 1 END) AS warning_count
FROM monitor_snapshots
GROUP BY hour
ORDER BY hour DESC;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO hgw_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO hgw_user;

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    gateway_id TEXT,
    alert_type TEXT,
    severity TEXT,
    message TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
  timestamp TIMESTAMPTZ,
  gateway_id TEXT,

  model_type TEXT,          -- 'LSTM' ou 'CatBoost' ou autre
  horizon TEXT,             -- '1h', '24h', '72h'

  prediction_score FLOAT,
  prediction_label TEXT
);
