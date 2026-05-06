-- =============================================================
-- HGW Predictive Maintenance — TimescaleDB Schema
-- =============================================================
-- Design choices:
--   * hgw_telemetry: hypertable, 30-second to 1-minute granularity
--   * 24h hot partition for fast Grafana queries
--   * Compression after 7 days (10x reduction)
--   * Retention: 1 year of compressed history, 24h uncompressed for batch
--   * Continuous aggregates for Grafana dashboard performance
--   * Predictions table for ML model outputs (CatBoost + LSTM)
-- =============================================================

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- -------------------------------------------------------------
-- 1. RAW TELEMETRY (collected via telnet_client.py)
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hgw_telemetry (
    timestamp           TIMESTAMPTZ       NOT NULL,
    gateway_id          TEXT              NOT NULL,
    firmware            TEXT,
    -- System metrics
    cpu_load            DOUBLE PRECISION,
    cpu_user_pct        DOUBLE PRECISION,
    cpu_system_pct      DOUBLE PRECISION,
    cpu_idle_pct        DOUBLE PRECISION,
    mem_total_mb        INTEGER,
    mem_used_mb         INTEGER,
    mem_free_mb         INTEGER,
    mem_used_pct        DOUBLE PRECISION,
    buffers_mb          INTEGER,
    cached_mb           INTEGER,
    -- Process-level (from /proc/<pid>/status when available)
    cwmp_rss_mb         DOUBLE PRECISION,
    dhcp_rss_mb         DOUBLE PRECISION,
    nemo_rss_mb         DOUBLE PRECISION,
    -- Network
    ping_latency        DOUBLE PRECISION,
    packet_loss         DOUBLE PRECISION,
    wan_status          SMALLINT,
    wan_rx_kbps         DOUBLE PRECISION,
    wan_tx_kbps         DOUBLE PRECISION,
    -- DHCP / WAN state
    dhcp_process_status TEXT,
    dhcp_data_state     TEXT,
    wan_state           TEXT,
    -- Status (computed by collector, not ML)
    local_status        TEXT,
    status_reason       TEXT,
    -- Raw command output (for audit/debug)
    raw_output          JSONB,
    PRIMARY KEY (timestamp, gateway_id)
);

-- Convert to TimescaleDB hypertable (1-day chunks)
SELECT create_hypertable('hgw_telemetry', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Index for gateway filtering
CREATE INDEX IF NOT EXISTS idx_telemetry_gateway_time
    ON hgw_telemetry (gateway_id, timestamp DESC);

-- -------------------------------------------------------------
-- 2. ML PREDICTIONS (output of 05_predict_service.py)
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hgw_predictions (
    timestamp           TIMESTAMPTZ       NOT NULL,
    gateway_id          TEXT              NOT NULL,
    model_version       TEXT              NOT NULL,
    -- Short-term (CatBoost 24h)
    prob_incident_24h   DOUBLE PRECISION,
    alert_24h           SMALLINT,
    -- Long-term (LSTM 72h)
    prob_incident_72h   DOUBLE PRECISION,
    alert_72h           SMALLINT,
    -- TTF + health
    ttf_hours_pred      DOUBLE PRECISION,
    health_score        DOUBLE PRECISION,
    risk_level          TEXT,
    -- Explainability
    top_reasons         JSONB,
    mc_dropout_std      DOUBLE PRECISION,
    PRIMARY KEY (timestamp, gateway_id)
);

SELECT create_hypertable('hgw_predictions', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_predictions_gateway_time
    ON hgw_predictions (gateway_id, timestamp DESC);

-- -------------------------------------------------------------
-- 3. CRASH/INCIDENT EVENTS (ground truth log)
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hgw_incidents (
    incident_id         BIGSERIAL         PRIMARY KEY,
    gateway_id          TEXT              NOT NULL,
    detected_at         TIMESTAMPTZ       NOT NULL,
    resolved_at         TIMESTAMPTZ,
    incident_type       TEXT,           -- 'crash', 'wan_outage', 'reboot', 'mem_leak'
    severity            TEXT,           -- 'low', 'medium', 'high', 'critical'
    -- Was this incident predicted in advance?
    was_predicted       BOOLEAN,
    prediction_lead_h   DOUBLE PRECISION,
    -- Free-form notes from on-call engineer
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_incidents_gateway_time
    ON hgw_incidents (gateway_id, detected_at DESC);

-- -------------------------------------------------------------
-- 4. DRIFT MONITORING LOG
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hgw_drift_log (
    check_id            BIGSERIAL         PRIMARY KEY,
    checked_at          TIMESTAMPTZ       NOT NULL DEFAULT NOW(),
    feature_name        TEXT              NOT NULL,
    psi                 DOUBLE PRECISION,
    ks_stat             DOUBLE PRECISION,
    ks_pval             DOUBLE PRECISION,
    status              TEXT,           -- 'OK', 'WARN', 'DRIFT'
    reference_period    TSTZRANGE,
    new_period          TSTZRANGE
);

-- =============================================================
-- 5. CONTINUOUS AGGREGATES (Grafana performance)
-- =============================================================

-- 5-minute rollup (for last-24h dashboard)
CREATE MATERIALIZED VIEW IF NOT EXISTS hgw_telemetry_5min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', timestamp) AS bucket,
    gateway_id,
    AVG(cpu_load)        AS cpu_load_avg,
    MAX(cpu_load)        AS cpu_load_max,
    AVG(mem_used_pct)    AS mem_used_avg,
    MAX(mem_used_pct)    AS mem_used_max,
    AVG(ping_latency)    AS ping_avg,
    MAX(ping_latency)    AS ping_max,
    AVG(packet_loss)     AS loss_avg,
    MIN(wan_status)      AS wan_status_min,
    MAX(cwmp_rss_mb)     AS cwmp_rss_max,
    COUNT(*)             AS samples
FROM hgw_telemetry
GROUP BY bucket, gateway_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('hgw_telemetry_5min',
    start_offset      => INTERVAL '7 days',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists     => TRUE
);

-- 1-hour rollup (for last-7-days dashboard)
CREATE MATERIALIZED VIEW IF NOT EXISTS hgw_telemetry_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    gateway_id,
    AVG(cpu_load)        AS cpu_load_avg,
    MAX(cpu_load)        AS cpu_load_max,
    AVG(mem_used_pct)    AS mem_used_avg,
    MAX(mem_used_pct)    AS mem_used_max,
    AVG(ping_latency)    AS ping_avg,
    AVG(packet_loss)     AS loss_avg,
    AVG(cwmp_rss_mb)     AS cwmp_rss_avg,
    MAX(cwmp_rss_mb)     AS cwmp_rss_max,
    COUNT(*) FILTER (WHERE wan_status = 0) AS wan_outage_count,
    COUNT(*)             AS samples
FROM hgw_telemetry
GROUP BY bucket, gateway_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('hgw_telemetry_1h',
    start_offset      => INTERVAL '90 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists     => TRUE
);

-- =============================================================
-- 6. RETENTION POLICIES
-- =============================================================
-- Compress raw telemetry after 7 days (~10x size reduction)
ALTER TABLE hgw_telemetry SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'gateway_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);
SELECT add_compression_policy('hgw_telemetry', INTERVAL '7 days',
    if_not_exists => TRUE);

-- Drop raw telemetry older than 1 year (aggregates remain for 5+ years)
SELECT add_retention_policy('hgw_telemetry', INTERVAL '365 days',
    if_not_exists => TRUE);

-- Drop predictions older than 6 months
SELECT add_retention_policy('hgw_predictions', INTERVAL '180 days',
    if_not_exists => TRUE);

-- =============================================================
-- 7. UTILITY VIEWS (used by Grafana)
-- =============================================================

-- Latest snapshot per gateway (for status panel)
CREATE OR REPLACE VIEW v_latest_status AS
SELECT DISTINCT ON (t.gateway_id)
    t.gateway_id,
    t.timestamp                AS last_seen,
    t.cpu_load,
    t.mem_used_pct,
    t.ping_latency,
    t.packet_loss,
    t.wan_status,
    t.cwmp_rss_mb,
    p.prob_incident_24h,
    p.prob_incident_72h,
    p.ttf_hours_pred,
    p.health_score,
    p.risk_level,
    p.top_reasons,
    EXTRACT(EPOCH FROM (NOW() - t.timestamp))/60 AS minutes_since_last
FROM hgw_telemetry t
LEFT JOIN hgw_predictions p
    ON t.gateway_id = p.gateway_id AND t.timestamp = p.timestamp
ORDER BY t.gateway_id, t.timestamp DESC;

-- 24h batch view (for dashboard "last day" panels)
CREATE OR REPLACE VIEW v_last_24h AS
SELECT
    timestamp,
    gateway_id,
    cpu_load,
    mem_used_pct,
    ping_latency,
    packet_loss,
    wan_status,
    cwmp_rss_mb,
    local_status
FROM hgw_telemetry
WHERE timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY gateway_id, timestamp;

-- Active alerts
CREATE OR REPLACE VIEW v_active_alerts AS
SELECT
    p.gateway_id,
    p.timestamp,
    p.prob_incident_24h,
    p.prob_incident_72h,
    p.health_score,
    p.risk_level,
    p.top_reasons
FROM hgw_predictions p
WHERE p.timestamp >= NOW() - INTERVAL '1 hour'
  AND (p.alert_24h = 1 OR p.alert_72h = 1 OR p.risk_level IN ('HIGH','CRITICAL'))
ORDER BY p.health_score ASC, p.timestamp DESC;

-- =============================================================
-- 8. HELPER FUNCTIONS
-- =============================================================

-- Insert a telemetry row (called by Python collector)
CREATE OR REPLACE FUNCTION insert_telemetry(payload JSONB)
RETURNS VOID AS $$
BEGIN
    INSERT INTO hgw_telemetry (
        timestamp, gateway_id, firmware,
        cpu_load, cpu_user_pct, cpu_system_pct, cpu_idle_pct,
        mem_total_mb, mem_used_mb, mem_free_mb, mem_used_pct,
        buffers_mb, cached_mb,
        cwmp_rss_mb, dhcp_rss_mb, nemo_rss_mb,
        ping_latency, packet_loss, wan_status, wan_rx_kbps, wan_tx_kbps,
        dhcp_process_status, dhcp_data_state, wan_state,
        local_status, status_reason, raw_output
    )
    VALUES (
        (payload->>'timestamp')::TIMESTAMPTZ,
        payload->>'gateway_id',
        payload->>'firmware',
        (payload->>'cpu_load')::DOUBLE PRECISION,
        (payload->>'cpu_user_pct')::DOUBLE PRECISION,
        (payload->>'cpu_system_pct')::DOUBLE PRECISION,
        (payload->>'cpu_idle_pct')::DOUBLE PRECISION,
        (payload->>'mem_total_mb')::INTEGER,
        (payload->>'mem_used_mb')::INTEGER,
        (payload->>'mem_free_mb')::INTEGER,
        (payload->>'mem_used_pct')::DOUBLE PRECISION,
        (payload->>'buffers_mb')::INTEGER,
        (payload->>'cached_mb')::INTEGER,
        (payload->>'cwmp_rss_mb')::DOUBLE PRECISION,
        (payload->>'dhcp_rss_mb')::DOUBLE PRECISION,
        (payload->>'nemo_rss_mb')::DOUBLE PRECISION,
        (payload->>'ping_latency')::DOUBLE PRECISION,
        (payload->>'packet_loss')::DOUBLE PRECISION,
        (payload->>'wan_status')::SMALLINT,
        (payload->>'wan_rx_kbps')::DOUBLE PRECISION,
        (payload->>'wan_tx_kbps')::DOUBLE PRECISION,
        payload->>'dhcp_process_status',
        payload->>'dhcp_data_state',
        payload->>'wan_state',
        payload->>'local_status',
        payload->>'status_reason',
        payload
    )
    ON CONFLICT (timestamp, gateway_id) DO NOTHING;
END;
$$ LANGUAGE plpgsql;

-- Print schema summary
DO $$
BEGIN
    RAISE NOTICE '================================';
    RAISE NOTICE 'HGW schema deployed successfully';
    RAISE NOTICE '================================';
END $$;
