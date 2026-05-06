# data_logger.py - Adaptation pour TimescaleDB
import os
from datetime import datetime
from sqlalchemy import create_engine, text

# Configuration DB depuis variable d'environnement ou défaut
DB_DSN = os.getenv(
    "DB_DSN",
    "postgresql://hgw_user:hgw_password@timescaledb:5432/hgw_monitoring"
)
print("DB_DSN =", DB_DSN)#pour debug
engine = create_engine(DB_DSN, pool_pre_ping=True)


def ensure_schema():
    if engine is None:
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE monitor_snapshots ADD COLUMN IF NOT EXISTS alert_eligible BOOLEAN"))
            conn.execute(text("ALTER TABLE monitor_snapshots ADD COLUMN IF NOT EXISTS alert_explanation TEXT"))
            conn.execute(text("ALTER TABLE monitor_snapshots ADD COLUMN IF NOT EXISTS alert_explainer_json JSONB"))
    except Exception as e:
        print(f"[DB WARN] Schema alert columns not ready: {e}")


def init_db():
    """Initialise la connexion à la base de données"""
    global engine
    try:
        engine = create_engine(DB_DSN, pool_pre_ping=True)
        print(f"[DB] Connexion établie ✅")
    except Exception as e:
        print(f"[DB ERROR] Initialisation échouée : {e}")
        engine = None

def save_snapshot(snap):
    global engine
    
    if engine is None:
        init_db()
    
    if engine is None:
        print("[DB ERROR] Engine non initialisé")
        return

    def clean(v):
        if v in ("NA", "N/A", "", None):
            return None
        try:
            return float(v)
        except:
            return v

    try:
        ensure_schema()
        data = {
            'timestamp': datetime.now(),  # 🔥 FIX IMPORTANT
            'LOCAL_STATUS': snap.get('LOCAL_STATUS'),
            'STATUS_REASON': snap.get('STATUS_REASON'),
            'CPU_USAGE_PERCENT': clean(snap.get('CPU_USAGE_PERCENT')),
            'MEM_USAGE_PERCENT': clean(snap.get('MEM_USAGE_PERCENT')),
            'WAN_STATE': snap.get('WAN_STATE'),
            'NET_LATENCY_MS': clean(snap.get('NET_LATENCY_MS')),
            'NET_PING_STATUS': snap.get('NET_PING_STATUS'),
            'DHCP_PROCESS_STATUS': snap.get('DHCP_PROCESS_STATUS'),
            'WAN_RX_RATE_KBPS': clean(snap.get('WAN_RX_RATE_KBPS')),
            'WAN_TX_RATE_KBPS': clean(snap.get('WAN_TX_RATE_KBPS')),
            'NET_LATENCY_AVG_5': clean(snap.get('NET_LATENCY_AVG_5')),
            'DHCP_DATA_STATE': snap.get('DHCP_DATA_STATE'),
            'DHCP_V6_STATE': snap.get('DHCP_V6_STATE'),
            'ALERT_ELIGIBLE': bool(snap.get('ALERT_ELIGIBLE', False)),
            'ALERT_EXPLANATION': snap.get('ALERT_EXPLANATION'),
            'ALERT_EXPLAINER_JSON': snap.get('ALERT_EXPLAINER_JSON'),
        }

        query = text("""
            INSERT INTO monitor_snapshots (
                timestamp,
                local_status,
                status_reason,
                cpu_usage_percent,
                mem_usage_percent,
                wan_state,
                net_latency_ms,
                net_ping_status,
                dhcp_process_status,
                wan_rx_rate_kbps,
                wan_tx_rate_kbps,
                net_latency_avg_5,
                dhcp_data_state,
                dhcp_v6_state,
                alert_eligible,
                alert_explanation,
                alert_explainer_json
            )
            VALUES (
                :timestamp,
                :LOCAL_STATUS,
                :STATUS_REASON,
                :CPU_USAGE_PERCENT,
                :MEM_USAGE_PERCENT,
                :WAN_STATE,
                :NET_LATENCY_MS,
                :NET_PING_STATUS,
                :DHCP_PROCESS_STATUS,
                :WAN_RX_RATE_KBPS,
                :WAN_TX_RATE_KBPS,
                :NET_LATENCY_AVG_5,
                :DHCP_DATA_STATE,
                :DHCP_V6_STATE,
                :ALERT_ELIGIBLE,
                :ALERT_EXPLANATION,
                CAST(:ALERT_EXPLAINER_JSON AS JSONB)
            )
        """)

        with engine.begin() as conn:  # 🔥 FIX transaction
            conn.execute(query, data)

        print(f"[DB] ✓ {data['timestamp']} | {data['LOCAL_STATUS']}")

    except Exception as e:
        print(f"[DB ERROR] Insert failed: {e}")

# Initialiser au chargement du module
init_db()
