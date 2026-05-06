-- HGW Airflow pipeline tables for TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Clean, normalized data used by training and prediction
CREATE TABLE IF NOT EXISTS monitor_snapshots_clean (
    timestamp TIMESTAMPTZ NOT NULL,
    gateway_id TEXT NOT NULL DEFAULT 'HGW_001',
    local_status TEXT,
    status_reason TEXT,
    cpu_load DOUBLE PRECISION,
    cpu_user_pct DOUBLE PRECISION,
    cpu_system_pct DOUBLE PRECISION,
    cpu_idle_pct DOUBLE PRECISION,
    mem_total_mb DOUBLE PRECISION,
    mem_free_mb DOUBLE PRECISION,
    mem_used_mb DOUBLE PRECISION,
    buffers_mb DOUBLE PRECISION,
    cached_mb DOUBLE PRECISION,
    mem_used_pct DOUBLE PRECISION,
    dhcp_process_status TEXT,
    dhcp_data_state TEXT,
    dhcp_v6_state TEXT,
    wan_state TEXT,
    wan_status SMALLINT,
    wan_ipv4_enable SMALLINT,
    wan_ipv6_enable SMALLINT,
    wan_rx_rate_kbps DOUBLE PRECISION,
    wan_tx_rate_kbps DOUBLE PRECISION,
    ping_latency DOUBLE PRECISION,
    ping_avg_5 DOUBLE PRECISION,
    packet_loss DOUBLE PRECISION,
    net_ping_status TEXT,
    health_score DOUBLE PRECISION,
    cpu_slope_30min DOUBLE PRECISION,
    ram_slope_30min DOUBLE PRECISION,
    cpu_mean_30min DOUBLE PRECISION,
    mem_mean_30min DOUBLE PRECISION,
    cpu_max_30min DOUBLE PRECISION,
    mem_max_30min DOUBLE PRECISION,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(timestamp, gateway_id)
);
SELECT create_hypertable('monitor_snapshots_clean', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_clean_gateway_time ON monitor_snapshots_clean(gateway_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_clean_status_time ON monitor_snapshots_clean(local_status, timestamp DESC);

-- Training-ready rows/materialized export history
CREATE TABLE IF NOT EXISTS model_training_dataset (
    timestamp TIMESTAMPTZ NOT NULL,
    gateway_id TEXT NOT NULL DEFAULT 'HGW_001',
    cpu_load DOUBLE PRECISION,
    mem_used_pct DOUBLE PRECISION,
    ping_latency DOUBLE PRECISION,
    packet_loss DOUBLE PRECISION,
    wan_status SMALLINT,
    health_score DOUBLE PRECISION,
    label_24h SMALLINT,
    label_72h SMALLINT,
    label_7d SMALLINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(timestamp, gateway_id)
);
SELECT create_hypertable('model_training_dataset', 'timestamp', if_not_exists => TRUE);

-- Multi-horizon predictions for Grafana and alerting
CREATE TABLE IF NOT EXISTS predictions_log (
    timestamp TIMESTAMPTZ NOT NULL,
    gateway_id TEXT NOT NULL DEFAULT 'HGW_001',
    horizon TEXT NOT NULL,
    horizon_min INTEGER,
    probability DOUBLE PRECISION,
    threshold DOUBLE PRECISION,
    alert BOOLEAN,
    decision_level TEXT,
    decision_message TEXT,
    model_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(timestamp, gateway_id, horizon)
);
SELECT create_hypertable('predictions_log', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_predictions_gateway_time ON predictions_log(gateway_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_alert ON predictions_log(alert, timestamp DESC);

-- Pipeline execution logs
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    task_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    rows_processed INTEGER DEFAULT 0,
    message TEXT
);

-- Grafana helper views
CREATE OR REPLACE VIEW v_latest_clean_status AS
SELECT DISTINCT ON (gateway_id)
    gateway_id,
    timestamp AS last_seen,
    local_status,
    status_reason,
    cpu_load,
    mem_used_pct,
    ping_latency,
    packet_loss,
    wan_status,
    health_score,
    EXTRACT(EPOCH FROM (NOW() - timestamp))/60 AS minutes_since_last
FROM monitor_snapshots_clean
ORDER BY gateway_id, timestamp DESC;

CREATE OR REPLACE VIEW v_latest_predictions AS
SELECT DISTINCT ON (gateway_id, horizon)
    gateway_id,
    timestamp,
    horizon,
    horizon_min,
    probability,
    threshold,
    alert,
    decision_level,
    decision_message
FROM predictions_log
ORDER BY gateway_id, horizon, timestamp DESC;
