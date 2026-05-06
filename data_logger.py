# data_logger.py - Adaptation pour TimescaleDB
import os
from datetime import datetime
from sqlalchemy import create_engine, text

# Configuration DB depuis variable d'environnement ou défaut
DB_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@timescaledb:5432/{os.getenv('POSTGRES_DB')}"

engine = create_engine(DB_URL)

def init_db():
    """Initialise la connexion à la base de données"""
    global engine
    try:
        engine = create_engine(DB_URL, pool_pre_ping=True)
        print(f"[DB] Connexion établie ✅")
    except Exception as e:
        print(f"[DB ERROR] Initialisation échouée : {e}")
        engine = None

def save_snapshot(snap):
    """
    Enregistre un snapshot dans TimescaleDB.
    Format compatible avec la table monitor_snapshots.
    """
    global engine
    
    if engine is None:
        init_db()
    
    if engine is None:
        print("[DB ERROR] Engine non initialisé, skip save")
        return
    
    try:
        # Convertir les valeurs "NA" en NULL
        def clean_value(v):
            if v in ("NA", "N/A", "", None):
                return None
            try:
                # Essayer de convertir en nombre si possible
                if isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit():
                    return float(v) if '.' in v else int(v)
                return v
            except:
                return v
        
        # Préparer les données
        data = {
            'timestamp': snap.get('timestamp', datetime.now().isoformat()),
            'LOCAL_STATUS': snap.get('LOCAL_STATUS'),
            'STATUS_REASON': snap.get('STATUS_REASON'),
            'CPU_USAGE_PERCENT': clean_value(snap.get('CPU_USAGE_PERCENT')),
            'CPU_USER_PERCENT': clean_value(snap.get('CPU_USER_PERCENT')),
            'CPU_SYSTEM_PERCENT': clean_value(snap.get('CPU_SYSTEM_PERCENT')),
            'CPU_IDLE_PERCENT': clean_value(snap.get('CPU_IDLE_PERCENT')),
            'MEM_TOTAL_MB': clean_value(snap.get('MEM_TOTAL_MB')),
            'MEM_FREE_MB': clean_value(snap.get('MEM_FREE_MB')),
            'MEM_USED_MB': clean_value(snap.get('MEM_USED_MB')),
            'BUFFERS_MB': clean_value(snap.get('BUFFERS_MB')),
            'CACHED_MB': clean_value(snap.get('CACHED_MB')),
            'MEM_USAGE_PERCENT': clean_value(snap.get('MEM_USAGE_PERCENT')),
            'DHCP_PROCESS_STATUS': snap.get('DHCP_PROCESS_STATUS'),
            'DHCP_DATA_STATE': snap.get('DHCP_DATA_STATE'),
            'DHCP_V6_STATE': snap.get('DHCP_V6_STATE'),
            'WAN_STATE': snap.get('WAN_STATE'),
            'WAN_IPV4_ENABLE': clean_value(snap.get('WAN_IPV4_ENABLE')),
            'WAN_IPV6_ENABLE': clean_value(snap.get('WAN_IPV6_ENABLE')),
            'WAN_RX_RATE_KBPS': clean_value(snap.get('WAN_RX_RATE_KBPS')),
            'WAN_TX_RATE_KBPS': clean_value(snap.get('WAN_TX_RATE_KBPS')),
            'NET_LATENCY_MS': clean_value(snap.get('NET_LATENCY_MS')),
            'NET_LATENCY_AVG_5': clean_value(snap.get('NET_LATENCY_AVG_5')),
            'NET_PING_STATUS': snap.get('NET_PING_STATUS'),
        }
        
        # Requête SQL INSERT
        query = text("""
            INSERT INTO monitor_snapshots (
                timestamp, LOCAL_STATUS, STATUS_REASON,
                CPU_USAGE_PERCENT, CPU_USER_PERCENT, CPU_SYSTEM_PERCENT, CPU_IDLE_PERCENT,
                MEM_TOTAL_MB, MEM_FREE_MB, MEM_USED_MB, BUFFERS_MB, CACHED_MB, MEM_USAGE_PERCENT,
                DHCP_PROCESS_STATUS, DHCP_DATA_STATE, DHCP_V6_STATE,
                WAN_STATE, WAN_IPV4_ENABLE, WAN_IPV6_ENABLE,
                WAN_RX_RATE_KBPS, WAN_TX_RATE_KBPS,
                NET_LATENCY_MS, NET_LATENCY_AVG_5, NET_PING_STATUS
            ) VALUES (
                :timestamp, :LOCAL_STATUS, :STATUS_REASON,
                :CPU_USAGE_PERCENT, :CPU_USER_PERCENT, :CPU_SYSTEM_PERCENT, :CPU_IDLE_PERCENT,
                :MEM_TOTAL_MB, :MEM_FREE_MB, :MEM_USED_MB, :BUFFERS_MB, :CACHED_MB, :MEM_USAGE_PERCENT,
                :DHCP_PROCESS_STATUS, :DHCP_DATA_STATE, :DHCP_V6_STATE,
                :WAN_STATE, :WAN_IPV4_ENABLE, :WAN_IPV6_ENABLE,
                :WAN_RX_RATE_KBPS, :WAN_TX_RATE_KBPS,
                :NET_LATENCY_MS, :NET_LATENCY_AVG_5, :NET_PING_STATUS
            )
        """)
        
        with engine.connect() as conn:
            conn.execute(query, data)
            conn.commit()
        
        # Log succinct (pas tous les champs)
        print(f"[DB] ✓ {data['timestamp']} | {data['LOCAL_STATUS']} | "
              f"CPU={data['CPU_USAGE_PERCENT']}% MEM={data['MEM_USAGE_PERCENT']}%")
    
    except Exception as e:
        print(f"[DB ERROR] Insert failed: {e}")

# Initialiser au chargement du module
init_db()
