"""
HGW Models — Interactive Testing Script
========================================

Teste tous les modèles entraînés sur CSV réelle.

USAGE :
    python test_models.py monitor_snapshots.csv

    # Ou pour tester sur une fenêtre temporelle spécifique :
    python test_models.py monitor_snapshots.csv --time "2026-04-22 11:00:00"

    # Ou pour tester sur N points aléatoires de la CSV :
    python test_models.py monitor_snapshots.csv --random 5

    #Pour tester et exporter les résultats dans un CSV pour visualisation en notebook :
    python test_models.py monitor_snapshots.csv --export predictions_now.csv

    # Pour tester et exporter en ajoutant à un CSV existant :
    python test_models.py monitor_snapshots.csv --export predictions_now.csv --append-export

LE SCRIPT FAIT :
  1. Charge tous les modèles ML (CatBoost 15min/30min/1h/6h)
  2. Charge le modèle DL (LSTM 3 jours)
  3. Applique le pipeline de preprocessing complet
  4. Affiche les prédictions pour chaque horizon
  5. Compare avec la vérité terrain (LOCAL_STATUS) si disponible
  6. Exporte les résultats dans un CSV pour visualisation (optionnel)
  7. Permet de tester sur des points spécifiques ou aléatoires de la CSV
"""
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

import joblib
from catboost import CatBoostClassifier, Pool
import tensorflow as tf

# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR / 'multi_horizon'
DL_DIR = SCRIPT_DIR / 'long_horizon_dl'

ML_FEATURES = [
    'cpu_load', 'mem_used_pct', 'ping_latency', 'packet_loss',
    'wan_status', 'reboot_event', 'recovery_phase',
    'cwmp_rss_mb', 'dhcp_rss_mb', 'nemo_rss_mb',
    'cpu_slope_5min', 'cpu_slope_30min', 'ram_slope_5min', 'ram_slope_30min',
    'cpu_mean_5min', 'cpu_mean_30min', 'cpu_std_30min', 'cpu_max_30min',
    'mem_mean_5min', 'mem_mean_30min', 'mem_std_30min', 'mem_max_30min',
    'ping_mean_5min', 'ping_mean_30min', 'ping_max_5min', 'loss_mean_5min',
    'wan_instability_5min',
    'cpu_lag1m', 'cpu_lag3m', 'cpu_lag5m', 'cpu_lag10m', 'cpu_lag15m',
    'mem_lag1m', 'mem_lag3m', 'mem_lag5m', 'mem_lag10m', 'mem_lag15m',
    'sin_hour', 'cos_hour', 'cpu_x_mem', 'saturation_idx', 'mem_headroom',
    'health_score',
]

FEATURE_LABELS = {
    'cpu_load': 'Charge CPU',
    'mem_used_pct': 'Mémoire utilisée',
    'ping_latency': 'Latence réseau',
    'packet_loss': 'Perte de paquets',
    'wan_status': 'État connexion WAN',
    'reboot_event': 'Redémarrage récent',
    'recovery_phase': 'Phase de récupération',
    'cwmp_rss_mb': 'Mémoire processus CWMP',
    'dhcp_rss_mb': 'Mémoire processus DHCP',
    'nemo_rss_mb': 'Mémoire processus NEMO',
    'cpu_slope_5min': 'Tendance CPU (5 min)',
    'cpu_slope_30min': 'Tendance CPU (30 min)',
    'ram_slope_5min': 'Tendance mémoire (5 min)',
    'ram_slope_30min': 'Tendance mémoire (30 min)',
    'cpu_mean_5min': 'CPU moyen 5 min',
    'cpu_mean_30min': 'CPU moyen 30 min',
    'cpu_std_30min': 'Instabilité CPU 30 min',
    'cpu_max_30min': 'Pic CPU 30 min',
    'mem_mean_5min': 'Mémoire moyenne 5 min',
    'mem_mean_30min': 'Mémoire moyenne 30 min',
    'mem_std_30min': 'Instabilité mémoire 30 min',
    'mem_max_30min': 'Pic mémoire 30 min',
    'ping_mean_5min': 'Latence moyenne 5 min',
    'ping_mean_30min': 'Latence moyenne 30 min',
    'ping_max_5min': 'Latence max 5 min',
    'loss_mean_5min': 'Perte de paquets moy. 5 min',
    'wan_instability_5min': 'Instabilité WAN 5 min',
    'cpu_lag1m': 'CPU il y a 1 min',
    'cpu_lag3m': 'CPU il y a 3 min',
    'cpu_lag5m': 'CPU il y a 5 min',
    'cpu_lag10m': 'CPU il y a 10 min',
    'cpu_lag15m': 'CPU il y a 15 min',
    'mem_lag1m': 'Mémoire il y a 1 min',
    'mem_lag3m': 'Mémoire il y a 3 min',
    'mem_lag5m': 'Mémoire il y a 5 min',
    'mem_lag10m': 'Mémoire il y a 10 min',
    'mem_lag15m': 'Mémoire il y a 15 min',
    'sin_hour': 'Heure de la journée',
    'cos_hour': 'Heure de la journée',
    'cpu_x_mem': 'Pression CPU + Mémoire combinée',
    'saturation_idx': 'Taux de saturation système',
    'mem_headroom': 'Marge mémoire disponible',
    'health_score': 'Score de santé global',
}

DL_FEATURES = [
    'cpu_load', 'mem_used_pct', 'ping_latency', 'packet_loss', 'wan_status',
    'cpu_mean_24h', 'ram_mean_24h', 'cpu_std_24h', 'ram_std_24h',
    'cpu_slope_6h', 'ram_slope_6h', 'wan_instability_6h', 'health_score',
]

INCIDENT_STATUSES = {'CRITICAL', 'URGENT'}


def get_lstm_threshold(metadata):
    """Support both metadata formats: bilstm_3d_metadata.json and lstm_metdata.json."""
    if 'metrics_synthetic_test' in metadata:
        return float(metadata['metrics_synthetic_test'].get('threshold', 0.5))
    return float(metadata.get('threshold', 0.5))


# =============================================================================
# Colors for terminal
# =============================================================================
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def colored(text, color):
    return f"{color}{text}{Colors.END}"


def format_probability(prob):
    """Keep tiny LSTM probabilities visible instead of printing them as 0.0000."""
    if prob == 0:
        return "0.000000"
    if abs(prob) < 0.01:
        return f"{prob:.6f}"
    return f"{prob:.4f}"


def describe_lstm_training(metadata):
    data = metadata.get('data', {})
    input_csv = Path(data.get('input_csv', '')).name if data.get('input_csv') else ''
    source_mode = metadata.get('source_mode', 'unknown')
    model_name = metadata.get('model')
    if input_csv:
        return f"entraîné sur {input_csv}"
    if model_name:
        return f"modÃ¨le {model_name}"
    if source_mode == 'engineered':
        return "entraîné sur dataset long terme préparé"
    if source_mode == 'raw_rebuilt':
        return "entraîné sur CSV réel reconstruit"
    return "pré-entraînement long terme"


# =============================================================================
# Flexible input loading / schema normalization
# =============================================================================
def find_similar_files(requested_path):
    requested = Path(requested_path)
    roots = [Path.cwd(), Path.cwd().parent, SCRIPT_DIR.parent, SCRIPT_DIR.parent / 'data', SCRIPT_DIR]
    suffixes = {'.csv', '.txt', '.tsv', '.json', '.jsonl', '.xlsx', '.xls', '.parquet', '.feather'}
    stem = requested.stem.lower()
    candidates, seen = [], set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob('*'):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            resolved = str(path.resolve()).lower()
            if resolved in seen:
                continue
            seen.add(resolved)
            name = path.stem.lower()
            if stem in name or name in stem or any(part and part in name for part in stem.split('_')):
                candidates.append(path)
    return candidates[:8]


def resolve_input_path(path_text):
    path = Path(path_text)
    if path.exists():
        return path
    root_relative = SCRIPT_DIR.parent / path_text
    if root_relative.exists():
        return root_relative
    similar = find_similar_files(path_text)
    msg = f"Fichier introuvable: {path_text}"
    if similar:
        msg += "\nFichiers proches trouvÃ©s:\n" + "\n".join(f"  - {p}" for p in similar)
    raise FileNotFoundError(msg)


def read_any_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in ['.csv', '.txt']:
        return pd.read_csv(path)
    if suffix == '.tsv':
        return pd.read_csv(path, sep='\t')
    if suffix == '.json':
        return pd.read_json(path)
    if suffix == '.jsonl':
        return pd.read_json(path, lines=True)
    if suffix in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    if suffix == '.parquet':
        return pd.read_parquet(path)
    if suffix == '.feather':
        return pd.read_feather(path)
    return pd.read_csv(path)


def find_column(df, aliases):
    normalized = {str(col).strip().lower().replace(' ', '_'): col for col in df.columns}
    for alias in aliases:
        key = alias.strip().lower().replace(' ', '_')
        if key in normalized:
            return normalized[key]
    return None


def numeric_series(df, aliases, default=0.0):
    col = find_column(df, aliases)
    if col is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors='coerce').ffill().bfill().fillna(default).astype(float)


def text_series(df, aliases, default=''):
    col = find_column(df, aliases)
    if col is None:
        return pd.Series(default, index=df.index, dtype=object)
    return df[col].astype(str).replace({'nan': default, 'None': default}).ffill().bfill().fillna(default)


def infer_status(cpu, mem, wan_state, ping_status, explicit=None):
    if explicit is not None:
        status = explicit.astype(str).str.upper()
        status = status.replace({'1': 'CRITICAL', 'TRUE': 'CRITICAL', '0': 'NORMAL', 'FALSE': 'NORMAL'})
        valid = status.isin(['NORMAL', 'WARNING', 'CRITICAL', 'URGENT'])
        inferred = pd.Series('NORMAL', index=cpu.index, dtype=object)
        inferred.loc[valid] = status.loc[valid]
        missing = ~valid
    else:
        inferred = pd.Series('NORMAL', index=cpu.index, dtype=object)
        missing = pd.Series(True, index=cpu.index)

    wan_down = wan_state.astype(str).str.upper().ne('UP')
    ping_fail = ping_status.astype(str).str.upper().eq('FAIL')
    inferred.loc[missing & ((cpu >= 90) | (mem >= 90) | wan_down)] = 'CRITICAL'
    inferred.loc[missing & inferred.eq('NORMAL') & ((cpu >= 75) | (mem >= 80) | ping_fail)] = 'WARNING'
    return inferred


def normalize_input_dataframe(df_raw):
    """Accept raw HGW snapshots, engineered long-horizon data, or minimal CPU/MEM tables."""
    if df_raw.empty:
        raise ValueError("Le fichier est vide.")

    df = df_raw.copy()
    ts_col = find_column(df, ['timestamp', 'time', 'datetime', 'date', 'logged_at', 'created_at'])
    if ts_col is not None:
        timestamps = pd.to_datetime(df[ts_col], errors='coerce')
    else:
        timestamps = pd.Series(pd.date_range('2026-01-01', periods=len(df), freq='1min'), index=df.index)
    if timestamps.isna().all():
        timestamps = pd.Series(pd.date_range('2026-01-01', periods=len(df), freq='1min'), index=df.index)
    else:
        timestamps = timestamps.ffill().bfill()

    cpu = numeric_series(df, ['CPU_USAGE_PERCENT', 'cpu_load', 'cpu', 'cpu_usage', 'cpu_percent', 'cpu_usage_percent'], 0.0)
    mem = numeric_series(df, ['MEM_USAGE_PERCENT', 'mem_used_pct', 'memory', 'mem', 'ram', 'ram_used_pct', 'memory_usage_percent'], 0.0)
    latency = numeric_series(df, ['NET_LATENCY_MS', 'ping_latency', 'latency', 'latency_ms', 'ping', 'NET_LATENCY_AVG_5'], 50.0)
    packet_loss = numeric_series(df, ['packet_loss', 'loss', 'packet_loss_pct'], 0.0)

    ping_status = text_series(df, ['NET_PING_STATUS', 'ping_status'], '')
    ping_missing = ping_status.eq('')
    ping_status.loc[ping_missing] = np.where((packet_loss.loc[ping_missing] >= 50) | (latency.loc[ping_missing] >= 300), 'FAIL', 'OK')

    wan_numeric = find_column(df, ['wan_status'])
    wan_text = text_series(df, ['WAN_STATE', 'wan_state'], '')
    if wan_numeric is not None and wan_text.eq('').all():
        wan_values = pd.to_numeric(df[wan_numeric], errors='coerce').fillna(1)
        wan_state = pd.Series(np.where(wan_values > 0, 'UP', 'DOWN'), index=df.index)
    else:
        wan_state = wan_text.str.upper().replace({'': 'UP', '1': 'UP', '0': 'DOWN', 'TRUE': 'UP', 'FALSE': 'DOWN'})

    status_col = find_column(df, ['LOCAL_STATUS', 'status', 'state', 'risk_level', 'is_crash'])
    explicit_status = df[status_col] if status_col is not None else None
    status = infer_status(cpu, mem, wan_state, ping_status, explicit_status)

    reason = text_series(df, ['STATUS_REASON', 'reason', 'episode_type'], '')
    reason = reason.mask(reason.eq('') & status.eq('CRITICAL'), 'pre_crash')
    reason = reason.mask(reason.eq('') & status.eq('WARNING'), 'resource_pressure')
    reason = reason.mask(reason.eq(''), 'healthy')

    out = pd.DataFrame({
        'timestamp': timestamps,
        'LOCAL_STATUS': status,
        'STATUS_REASON': reason,
        'CPU_USAGE_PERCENT': cpu,
        'CPU_USER_PERCENT': numeric_series(df, ['CPU_USER_PERCENT', 'cpu_user_percent'], cpu * 0.7),
        'CPU_SYSTEM_PERCENT': numeric_series(df, ['CPU_SYSTEM_PERCENT', 'cpu_system_percent'], cpu * 0.3),
        'CPU_IDLE_PERCENT': numeric_series(df, ['CPU_IDLE_PERCENT', 'cpu_idle_percent'], 100.0 - cpu),
        'MEM_TOTAL_MB': numeric_series(df, ['MEM_TOTAL_MB', 'mem_total_mb'], 936.0),
        'MEM_USAGE_PERCENT': mem,
        'DHCP_PROCESS_STATUS': text_series(df, ['DHCP_PROCESS_STATUS', 'dhcp_process_status'], 'RUNNING'),
        'WAN_STATE': wan_state,
        'NET_LATENCY_MS': latency,
        'NET_PING_STATUS': ping_status,
    })
    out['MEM_USED_MB'] = numeric_series(df, ['MEM_USED_MB', 'mem_used_mb'], out['MEM_TOTAL_MB'] * out['MEM_USAGE_PERCENT'] / 100)
    out['MEM_FREE_MB'] = numeric_series(df, ['MEM_FREE_MB', 'mem_free_mb'], out['MEM_TOTAL_MB'] - out['MEM_USED_MB'])
    out['BUFFERS_MB'] = numeric_series(df, ['BUFFERS_MB', 'buffers_mb'], 25.0)
    out['CACHED_MB'] = numeric_series(df, ['CACHED_MB', 'cached_mb'], 120.0)
    out['DHCP_DATA_STATE'] = text_series(df, ['DHCP_DATA_STATE', 'dhcp_data_state'], 'Bound')
    out['DHCP_V6_STATE'] = text_series(df, ['DHCP_V6_STATE', 'dhcp_v6_state'], 'Bound')
    out['WAN_IPV4_ENABLE'] = numeric_series(df, ['WAN_IPV4_ENABLE', 'wan_ipv4_enable'], 1)
    out['WAN_IPV6_ENABLE'] = numeric_series(df, ['WAN_IPV6_ENABLE', 'wan_ipv6_enable'], 1)
    out['WAN_RX_RATE_KBPS'] = numeric_series(df, ['WAN_RX_RATE_KBPS', 'wan_rx_rate_kbps', 'wan_rx_rate'], 0)
    out['WAN_TX_RATE_KBPS'] = numeric_series(df, ['WAN_TX_RATE_KBPS', 'wan_tx_rate_kbps', 'wan_tx_rate'], 0)
    out['NET_LATENCY_AVG_5'] = numeric_series(df, ['NET_LATENCY_AVG_5', 'latency_mean_5min'], out['NET_LATENCY_MS'])
    out['logged_at'] = timestamps

    return out.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)


# =============================================================================
# Preprocessing functions
# =============================================================================
def compute_health_score(cpu, mem, ping, loss):
    cpu, mem, ping, loss = [np.asarray(x, dtype=float) for x in [cpu, mem, ping, loss]]
    n_cpu = np.clip((cpu - 20) / 70, 0, 1)
    n_mem = np.clip((mem - 35) / 55, 0, 1)
    n_ping = np.clip((ping - 20) / 200, 0, 1)
    n_loss = np.clip(loss / 15, 0, 1)
    composite = 0.35 * n_mem + 0.30 * n_cpu + 0.20 * n_ping + 0.15 * n_loss
    return np.round((1.0 - np.clip(composite, 0, 1)) * 100, 1)


def map_real_to_standard(df_raw):
    df = pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    df['cpu_load'] = df_raw['CPU_USAGE_PERCENT'].astype(float)
    df['mem_used_pct'] = df_raw['MEM_USAGE_PERCENT'].astype(float)
    df['ping_latency'] = pd.to_numeric(
        df_raw['NET_LATENCY_MS'], errors='coerce'
    ).ffill().fillna(50.0)
    df['wan_status'] = (df_raw['WAN_STATE'] == 'UP').astype(int)
    df['packet_loss'] = (df_raw['NET_PING_STATUS'] == 'FAIL').astype(int) * 100.0
    df['cwmp_rss_mb'] = 0.0
    df['dhcp_rss_mb'] = 0.0
    df['nemo_rss_mb'] = 0.0
    dhcp_run = (df_raw['DHCP_PROCESS_STATUS'] == 'RUNNING').astype(int).values
    df['reboot_event'] = np.concatenate(
        [[0], (dhcp_run[1:] == 1) & (dhcp_run[:-1] == 0)]
    ).astype(int)
    df['recovery_phase'] = 0
    df['LOCAL_STATUS'] = df_raw['LOCAL_STATUS'].values
    df['STATUS_REASON'] = df_raw['STATUS_REASON'].values
    return df.sort_values('timestamp').reset_index(drop=True)


def build_ml_features(df_1min):
    """Compute the 43 ML features at 1-min granularity."""
    g = df_1min.copy().sort_values('timestamp').reset_index(drop=True)
    g['cpu_slope_30min'] = g['cpu_load'].diff(30).fillna(0) / 30
    g['ram_slope_30min'] = g['mem_used_pct'].diff(30).fillna(0) / 30
    g['cpu_slope_5min'] = g['cpu_load'].diff(5).fillna(0) / 5
    g['ram_slope_5min'] = g['mem_used_pct'].diff(5).fillna(0) / 5
    g['cpu_mean_5min'] = g['cpu_load'].rolling(5, min_periods=1).mean()
    g['cpu_mean_30min'] = g['cpu_load'].rolling(30, min_periods=1).mean()
    g['cpu_std_30min'] = g['cpu_load'].rolling(30, min_periods=1).std().fillna(0)
    g['cpu_max_30min'] = g['cpu_load'].rolling(30, min_periods=1).max()
    g['mem_mean_5min'] = g['mem_used_pct'].rolling(5, min_periods=1).mean()
    g['mem_mean_30min'] = g['mem_used_pct'].rolling(30, min_periods=1).mean()
    g['mem_std_30min'] = g['mem_used_pct'].rolling(30, min_periods=1).std().fillna(0)
    g['mem_max_30min'] = g['mem_used_pct'].rolling(30, min_periods=1).max()
    g['ping_mean_5min'] = g['ping_latency'].rolling(5, min_periods=1).mean()
    g['ping_mean_30min'] = g['ping_latency'].rolling(30, min_periods=1).mean()
    g['ping_max_5min'] = g['ping_latency'].rolling(5, min_periods=1).max()
    g['loss_mean_5min'] = g['packet_loss'].rolling(5, min_periods=1).mean()
    g['wan_instability_5min'] = g['wan_status'].eq(0).rolling(5, min_periods=1).mean()
    for lag in [1, 3, 5, 10, 15]:
        g[f'cpu_lag{lag}m'] = g['cpu_load'].shift(lag).bfill()
        g[f'mem_lag{lag}m'] = g['mem_used_pct'].shift(lag).bfill()
    g['hour'] = g['timestamp'].dt.hour
    g['sin_hour'] = np.sin(2 * np.pi * g['hour'] / 24)
    g['cos_hour'] = np.cos(2 * np.pi * g['hour'] / 24)
    g['cpu_x_mem'] = g['cpu_load'] * g['mem_used_pct'] / 10000
    g['saturation_idx'] = (g['cpu_load'] / 88 + g['mem_used_pct'] / 90) / 2
    g['mem_headroom'] = np.clip(90.0 - g['mem_used_pct'], 0, 90)
    g['health_score'] = compute_health_score(
        g['cpu_load'].fillna(0), g['mem_used_pct'].fillna(0),
        g['ping_latency'].fillna(50), g['packet_loss'].fillna(0)
    )
    return g


def build_dl_features(df_1min):
    """Compute the 13 DL features at 30-min granularity."""
    g = df_1min.copy().sort_values('timestamp').reset_index(drop=True)
    if len(g) < 2:
        return None

    win_24h = min(len(g), 1440)
    g['cpu_mean_24h'] = g['cpu_load'].rolling(win_24h, min_periods=1).mean()
    g['ram_mean_24h'] = g['mem_used_pct'].rolling(win_24h, min_periods=1).mean()
    g['cpu_std_24h'] = g['cpu_load'].rolling(win_24h, min_periods=1).std().fillna(0)
    g['ram_std_24h'] = g['mem_used_pct'].rolling(win_24h, min_periods=1).std().fillna(0)
    win_6h = min(len(g), 360)
    g['cpu_slope_6h'] = (g['cpu_load'] - g['cpu_load'].shift(win_6h)).fillna(0) / 6
    g['ram_slope_6h'] = (g['mem_used_pct'] - g['mem_used_pct'].shift(win_6h)).fillna(0) / 6
    g['wan_instability_6h'] = g['wan_status'].eq(0).rolling(win_6h, min_periods=1).mean()
    g['health_score'] = compute_health_score(
        g['cpu_load'].fillna(0), g['mem_used_pct'].fillna(0),
        g['ping_latency'].fillna(50), g['packet_loss'].fillna(0)
    )

    # Resample to 30-min
    g30 = g.set_index('timestamp')[DL_FEATURES].resample('30min').mean().dropna()
    return g30


def _feature_value_context(feature, value):
    """Return a short plain-language description of a feature's current value."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None

    if feature in ('cos_hour', 'sin_hour'):
        # Decode approximate hour from the cyclical encoding
        try:
            import math
            if feature == 'cos_hour':
                h = round(math.acos(max(-1.0, min(1.0, v))) * 24 / (2 * math.pi))
                return f"Vers {h}h ou {24 - h}h — plage associée à plus d'incidents dans l'historique"
            else:
                h = round(math.asin(max(-1.0, min(1.0, v))) * 24 / (2 * math.pi))
                h = h % 24
                return f"L'heure actuelle ({h}h) est une plage à risque selon l'historique"
        except Exception:
            return "L'heure actuelle correspond à une plage à risque selon l'historique"
    if feature == 'cpu_load':
        if v >= 90: return f"CPU critique ({v:.0f}%)"
        if v >= 70: return f"CPU élevé ({v:.0f}%)"
        if v >= 40: return f"CPU modéré ({v:.0f}%)"
        return f"CPU faible ({v:.0f}%)"
    if feature == 'mem_used_pct':
        if v >= 95: return f"Mémoire saturée ({v:.0f}%)"
        if v >= 85: return f"Mémoire sous pression ({v:.0f}%)"
        if v >= 70: return f"Mémoire modérée ({v:.0f}%)"
        return f"Mémoire normale ({v:.0f}%)"
    if feature == 'cpu_mean_30min':
        if v >= 70: return "Moyenne CPU élevée sur 30 min"
        if v >= 40: return "Moyenne CPU modérée sur 30 min"
        return "Moyenne CPU basse sur 30 min"
    if feature == 'cpu_mean_5min':
        if v >= 70: return "Moyenne CPU élevée sur 5 min"
        return "Moyenne CPU normale sur 5 min"
    if feature == 'cpu_max_30min':
        if v >= 85: return "Pic CPU dangereux sur 30 min"
        if v >= 60: return "Pic CPU modéré sur 30 min"
        return "Pic CPU normal sur 30 min"
    if feature == 'cpu_std_30min':
        if v >= 10: return "CPU très instable sur 30 min"
        if v >= 5: return "CPU fluctuant sur 30 min"
        return "CPU stable sur 30 min"
    if feature in ('mem_mean_5min', 'mem_mean_30min'):
        suffix = "5 min" if '5min' in feature else "30 min"
        if v >= 85: return f"Mémoire élevée sur {suffix}"
        if v >= 70: return f"Mémoire modérée sur {suffix}"
        return f"Mémoire normale sur {suffix}"
    if feature == 'mem_max_30min':
        if v >= 90: return "Pic mémoire critique sur 30 min"
        if v >= 75: return "Pic mémoire élevé sur 30 min"
        return "Pic mémoire normal sur 30 min"
    if feature == 'mem_headroom':
        if v <= 10: return "Mémoire presque pleine"
        if v <= 25: return "Peu de mémoire disponible"
        return "Mémoire disponible suffisante"
    if feature == 'ping_latency':
        if v >= 200: return "Latence réseau très élevée"
        if v >= 100: return "Latence réseau élevée"
        if v >= 50: return "Latence réseau acceptable"
        return "Latence réseau bonne"
    if feature == 'ping_max_5min':
        if v >= 300: return "Pics de latence extrêmes"
        if v >= 150: return "Pics de latence importants"
        return "Latence stable"
    if feature in ('ping_mean_5min', 'ping_mean_30min'):
        suffix = "5 min" if '5min' in feature else "30 min"
        if v >= 150: return f"Latence moyenne élevée sur {suffix}"
        return f"Latence moyenne correcte sur {suffix}"
    if feature == 'packet_loss':
        if v >= 10: return "Perte de paquets significative"
        if v >= 3: return "Légère perte de paquets"
        return "Réseau sans perte"
    if feature == 'wan_instability_5min':
        if v > 0.3: return "WAN instable"
        return "WAN stable"
    if 'cpu_slope' in feature:
        d = "5 min" if '5min' in feature else "30 min"
        if v > 2: return f"CPU en forte hausse sur {d}"
        if v > 0.5: return f"CPU en légère hausse sur {d}"
        if v < -1: return f"CPU en baisse sur {d}"
        return f"CPU stable sur {d}"
    if 'ram_slope' in feature:
        d = "5 min" if '5min' in feature else "30 min"
        if v > 2: return f"Mémoire en forte hausse sur {d}"
        if v > 0.5: return f"Mémoire en légère hausse sur {d}"
        if v < -1: return f"Mémoire en baisse sur {d}"
        return f"Mémoire stable sur {d}"
    if feature == 'cpu_x_mem':
        return "Pression simultanée CPU et mémoire" if v > 50 else None
    if feature == 'saturation_idx':
        if v > 0.7: return "Système proche de la saturation"
        if v > 0.4: return "Saturation modérée"
        return "Charge système normale"
    if feature == 'health_score':
        if v > 0.7: return "État système sain"
        if v > 0.4: return "État système modéré"
        return "État système dégradé"
    return None


def _impact_level(share):
    """Convert relative SHAP share to human-readable level."""
    if share >= 0.20:
        return 'fort'
    if share >= 0.10:
        return 'modéré'
    return 'faible'


_HORIZON_LABELS = {
    '15min': '15 min', '30min': '30 min', '60min': '1 h', '360min': '6 h',
    '3 jours': '3 jours', '3j': '3 jours', '3day': '3 jours', 'bilstm_3d': '3 jours',
}


def _shap_business_explanation(top_features, probability, horizon):
    """Generate a plain-language diagnostic sentence from SHAP results."""
    pct = round(probability * 100)
    h_label = _HORIZON_LABELS.get(str(horizon), str(horizon))

    if probability >= 0.75:
        intro = f"Risque élevé ({pct}%) dans les {h_label} à venir."
    elif probability >= 0.45:
        intro = f"Risque modéré ({pct}%) dans les {h_label} à venir."
    elif probability >= 0.25:
        intro = f"Risque faible ({pct}%) dans les {h_label} à venir."
    else:
        intro = f"Système en bon état ({pct}% de risque sur {h_label})."

    strong_up = [f for f in top_features if f['impact'] == 'increase' and f['impact_level'] == 'fort']
    mod_up = [f for f in top_features if f['impact'] == 'increase' and f['impact_level'] == 'modéré']
    strong_down = [f for f in top_features if f['impact'] == 'decrease' and f['impact_level'] == 'fort']

    parts = [intro]

    if strong_up:
        ctx = strong_up[0].get('context') or strong_up[0]['label']
        extra = f", {strong_up[1]['label']}" if len(strong_up) > 1 else ""
        parts.append(f"{ctx}{extra} augmente le risque.")
    elif mod_up:
        ctx = mod_up[0].get('context') or mod_up[0]['label']
        parts.append(f"{ctx} contribue légèrement au risque.")

    if strong_down:
        ctx = strong_down[0].get('context') or strong_down[0]['label']
        parts.append(f"{ctx} stabilise la situation.")

    return ' '.join(parts)


def explain_catboost_prediction(model, x_row, features, horizon, probability, top_n=6):
    """Return local SHAP explanation for one CatBoost prediction."""
    pool = Pool(x_row[features], feature_names=features)
    shap_values = np.asarray(model.get_feature_importance(pool, type='ShapValues'))

    if shap_values.ndim == 3:
        # Multiclass safety: keep the positive class when available.
        class_idx = 1 if shap_values.shape[2] > 1 else 0
        feature_shap = shap_values[0, :-1, class_idx]
        base_value = shap_values[0, -1, class_idx]
    else:
        feature_shap = shap_values[0, :-1]
        base_value = shap_values[0, -1]

    abs_total = float(np.sum(np.abs(feature_shap))) or 1.0
    x_values = x_row.iloc[0]
    top_indexes = np.argsort(np.abs(feature_shap))[::-1][:top_n]
    top_features = []

    for idx in top_indexes:
        feature = features[int(idx)]
        shap_value = float(feature_shap[int(idx)])
        value = x_values.get(feature)
        if pd.isna(value):
            clean_value = None
        else:
            clean_value = float(value) if isinstance(value, (int, float, np.number)) else str(value)

        share = round(abs(shap_value) / abs_total, 4)
        top_features.append({
            'feature': feature,
            'label': FEATURE_LABELS.get(feature, feature),
            'value': clean_value,
            'context': _feature_value_context(feature, clean_value),
            'shap_value': round(shap_value, 6),
            'abs_shap': round(abs(shap_value), 6),
            'share': share,
            'impact': 'increase' if shap_value >= 0 else 'decrease',
            'impact_level': _impact_level(share),
        })

    increasing = [item['label'] for item in top_features if item['impact'] == 'increase']
    decreasing = [item['label'] for item in top_features if item['impact'] == 'decrease']
    if increasing:
        summary = 'Risque augmente surtout par: ' + ', '.join(increasing[:3])
    elif decreasing:
        summary = 'Risque attenue surtout par: ' + ', '.join(decreasing[:3])
    else:
        summary = 'Aucun facteur dominant detecte'

    return {
        'type': 'catboost_shap',
        'horizon': horizon,
        'probability': round(float(probability), 6),
        'base_value': round(float(base_value), 6),
        'summary': summary,
        'business_explanation': _shap_business_explanation(top_features, float(probability), horizon),
        'top_features': top_features,
    }


# =============================================================================
# Load all models
# =============================================================================
def load_all_models():
    """Load CatBoost multi-horizon models + optional LSTM.
    Compatible with both bundle formats:
      A) {"horizons": {"15min": {...}}}
      B) {"15min": {...}, "30min": {...}}
    """
    print(colored("Chargement des modèles...", Colors.BLUE))
    models = {}

    bundle_path = ML_DIR / 'multi_horizon_bundle.json'
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle introuvable: {bundle_path}")

    with open(bundle_path, encoding='utf-8') as f:
        bundle_raw = json.load(f)

    # Support old/new JSON formats
    horizons = bundle_raw.get('horizons', bundle_raw)
    models['bundle'] = bundle_raw

    for h_min in [15, 30, 60, 360]:
        h_key = f'{h_min}min'
        info = horizons.get(h_key)
        if info is None:
            print(f"  ⚠️ Horizon {h_key} absent du bundle")
            continue

        if info.get('status', 'OK') != 'OK':
            print(f"  ⚠️ CatBoost {h_key} ignoré: status={info.get('status')}")
            continue

        model_file = info.get('model_file')
        if not model_file:
            print(f"  ⚠️ CatBoost {h_key} ignoré: model_file manquant")
            continue

        model_path = ML_DIR / model_file
        if not model_path.exists():
            print(f"  ⚠️ CatBoost {h_key} introuvable: {model_path}")
            continue

        # Support both threshold formats
        if 'thresholds' in info:
            threshold = info['thresholds'].get('balanced_F1', info['thresholds'].get('high_recall_F2', 0.5))
        else:
            threshold = info.get('threshold', 0.5)

        cb = CatBoostClassifier()
        cb.load_model(str(model_path))
        models[h_key] = {
            'model': cb,
            'threshold': float(threshold),
            'horizon_min': h_min,
        }
        print(f"  ✓ CatBoost {h_min:>4}min  (seuil={float(threshold):.3f})")

    # DL Bi-LSTM / LSTM 3-day (optional)
    lstm_path = DL_DIR / 'bilstm_3d_synthetic.keras'
    scaler_path = DL_DIR / 'transfer_scaler.pkl'
    metadata_path = DL_DIR / 'bilstm_3d_metadata.json'
    weights_path = DL_DIR / 'bilstm_3d.weights.h5'

    # Preferred LSTM artifacts requested for the demo.
    lstm_keras_path = DL_DIR / 'lstm_3day.keras'
    lstm_scaler_path = DL_DIR / 'lstm_scaler.pkl'
    lstm_metadata_path = DL_DIR / 'lstm_metdata.json'
    if lstm_keras_path.exists() and lstm_scaler_path.exists():
        try:
            metadata = json.load(open(lstm_metadata_path, encoding='utf-8')) if lstm_metadata_path.exists() else {}
            models['bilstm_3d'] = {
                'model': tf.keras.models.load_model(str(lstm_keras_path), compile=False),
                'scaler': joblib.load(lstm_scaler_path),
                'metadata': metadata,
                'threshold': get_lstm_threshold(metadata),
                'artifact': 'lstm_3day.keras + lstm_scaler.pkl',
            }
            th = models['bilstm_3d']['threshold']
            print(f"  ✓ LSTM 3 jours      (seuil={th:.3f}, lstm_scaler.pkl)")
        except Exception as e:
            print(f"  ⚠️ LSTM loading skipped: {e}")

    

    return models

# =============================================================================
# Predict at a specific timestamp
# =============================================================================
def predict_at_timestamp(df_raw, target_ts, models):
    """
    Predict at a specific timestamp using all models.
    Uses the data BEFORE target_ts for inference (no future leakage).
    """
    df_before = df_raw[df_raw['timestamp'] <= target_ts].copy()

    if len(df_before) < 60:
        return {
            'error': f'Pas assez de données avant {target_ts} (besoin de 60+ samples)',
            'target_ts': str(target_ts),
        }

    # Prep
    df_std = map_real_to_standard(df_before)

    # Resample to 1-min
    df_1m = df_std.set_index('timestamp').resample('1min').mean(numeric_only=True).ffill().dropna()
    df_1m = df_1m.reset_index()

    if len(df_1m) < 30:
        return {
            'error': f'Pas assez de données après resampling: {len(df_1m)} min',
            'target_ts': str(target_ts),
        }

    results = {
        'timestamp': str(target_ts),
        'predictions': {},
        'ground_truth': None,
        'business_alert': None,
        'xai': {
            'type': 'catboost_shap',
            'description': 'Explication locale des predictions CatBoost par valeurs SHAP.',
            'horizons': {},
        },
    }

    # Current raw state = latest row available before target_ts
    current_raw = df_before.iloc[-1]
    cpu_now = float(current_raw.get('CPU_USAGE_PERCENT', 0))
    mem_now = float(current_raw.get('MEM_USAGE_PERCENT', 0))
    latency_now = pd.to_numeric(current_raw.get('NET_LATENCY_MS', 0), errors='coerce')
    latency_now = 0.0 if pd.isna(latency_now) else float(latency_now)
    wan_down = str(current_raw.get('WAN_STATE', 'UP')).upper() != 'UP'
    ping_fail = str(current_raw.get('NET_PING_STATUS', 'OK')).upper() == 'FAIL'
    status_now = str(current_raw.get('LOCAL_STATUS', 'NORMAL')).upper()

    # Rule-based QA guardrail: catches immediate critical states that ML may miss
    business = {'alert': False, 'level': 'OK', 'message': 'État courant normal'}
    if status_now in INCIDENT_STATUSES:
        business = {'alert': True, 'level': 'CRITICAL', 'message': f'État système {status_now}'}
    elif wan_down:
        business = {'alert': True, 'level': 'CRITICAL', 'message': 'WAN down'}
    elif cpu_now >= 90:
        business = {'alert': True, 'level': 'CRITICAL', 'message': f'CPU critique ({cpu_now:.0f}%)'}
    elif mem_now >= 90:
        business = {'alert': True, 'level': 'CRITICAL', 'message': f'RAM critique ({mem_now:.0f}%)'}
    elif latency_now >= 300 or ping_fail:
        business = {'alert': True, 'level': 'WARNING', 'message': f'Réseau dégradé (latence={latency_now:.0f}ms)'}
    elif cpu_now >= 80 or mem_now >= 80:
        business = {'alert': True, 'level': 'WARNING', 'message': f'Ressources élevées CPU={cpu_now:.0f}% MEM={mem_now:.0f}%'}
    results['business_alert'] = business

    # Get ground truth at target_ts
    actual = df_raw[df_raw['timestamp'] == target_ts]
    if not actual.empty:
        results['ground_truth'] = {
            'LOCAL_STATUS': actual.iloc[0]['LOCAL_STATUS'],
            'STATUS_REASON': actual.iloc[0]['STATUS_REASON'],
            'CPU': int(actual.iloc[0]['CPU_USAGE_PERCENT']),
            'MEM': int(actual.iloc[0]['MEM_USAGE_PERCENT']),
        }

    # ============ ML predictions ============
    df_ml = build_ml_features(df_1m)
    last_row = df_ml.iloc[[-1]]

    # Verify all features present
    missing = set(ML_FEATURES) - set(last_row.columns)
    if missing:
        results['ml_error'] = f'Features manquantes: {missing}'
    elif last_row[ML_FEATURES].isna().any().any():
        results['ml_error'] = 'NaN dans les features ML'
    else:
        X_ml = last_row[ML_FEATURES]
        for h_key in ['15min', '30min', '60min', '360min']:
            if h_key not in models:
                continue
            m = models[h_key]
            prob = float(m['model'].predict_proba(X_ml)[0, 1])
            alert = prob >= m['threshold']
            shap_explanation = None
            try:
                shap_explanation = explain_catboost_prediction(
                    m['model'], X_ml, ML_FEATURES, h_key, prob
                )
                results['xai']['horizons'][h_key] = shap_explanation
            except Exception as exc:
                results['xai_error'] = f'SHAP indisponible pour {h_key}: {exc}'
            results['predictions'][h_key] = {
                'horizon_min': m['horizon_min'],
                'probability': round(prob, 4),
                'threshold': round(m['threshold'], 4),
                'alert': alert,
                'shap': shap_explanation,
            }

    # ============ DL prediction ============
    if 'bilstm_3d' in models:
        df_30m = build_dl_features(df_1m)
        LSTM_MIN_ROWS = 24   # 24 × 30-min = 12h minimum de données réelles
        if df_30m is None or len(df_30m) < LSTM_MIN_ROWS:
            results['dl_error'] = (
                f'Pas assez de données 30-min: {len(df_30m) if df_30m is not None else 0} '
                f'(besoin de {LSTM_MIN_ROWS} = 12h)'
            )
        else:
            TARGET_STEPS = 24
            available = df_30m[DL_FEATURES]
            n_avail = len(available)

            if n_avail >= 48:
                # Full 24h window: 48 × 30-min → subsample by 2 → 24 timesteps
                X_seq = available.tail(48).values[::2]
                coverage = '24h'
            else:
                # 12–24h window: take last 24 rows directly (no padding)
                X_seq = available.tail(TARGET_STEPS).values
                coverage = f'{n_avail * 30 // 60}h'

            if np.isnan(X_seq).any():
                results['dl_error'] = 'NaN dans les features DL'
            else:
                X_scaled = models['bilstm_3d']['scaler'].transform(X_seq)
                seq = X_scaled[np.newaxis, ...].astype(np.float32)  # (1, 24, 13)

                if seq.shape != (1, TARGET_STEPS, 13):
                    results['dl_error'] = f'Shape attendue (1,{TARGET_STEPS},13), obtenu {seq.shape}'
                else:
                    prob = float(models['bilstm_3d']['model'].predict(seq, verbose=0)[0, 0])
                    th = models['bilstm_3d'].get(
                        'threshold',
                        get_lstm_threshold(models['bilstm_3d'].get('metadata', {}))
                    )
                    results['predictions']['3 jours'] = {
                        'horizon_min': 3 * 24 * 60,
                        'probability': round(prob, 6),
                        'threshold': round(th, 4),
                        'alert': prob >= th,
                        'coverage': coverage,
                    }

    return results


# =============================================================================
# Pretty print results
# =============================================================================
def print_results(results):
    print("\n" + "=" * 75)
    print(colored(f"  PRÉDICTION À T = {results.get('timestamp', results.get('ts'))}", Colors.BOLD))
    print("=" * 75)

    if 'error' in results:
        print(colored(f"\n  ❌ ERREUR : {results['error']}", Colors.RED))
        return

    # Ground truth
    if results.get('ground_truth'):
        gt = results['ground_truth']
        status = gt['LOCAL_STATUS']
        if status == 'NORMAL':
            color = Colors.GREEN
        elif status == 'WARNING':
            color = Colors.YELLOW
        else:
            color = Colors.RED
        print(colored(
            f"\n  État réel à cet instant : {status} ({gt['STATUS_REASON']})",
            color
        ))
        print(f"  CPU={gt['CPU']}%  MEM={gt['MEM']}%")

    if results.get('business_alert') and results['business_alert']['alert']:
        ba = results['business_alert']
        icon = '🔴' if ba['level'] == 'CRITICAL' else '🟡'
        print(colored(f"\n  {icon} ALERTE MÉTIER : {ba['level']} — {ba['message']}",
                      Colors.RED if ba['level'] == 'CRITICAL' else Colors.YELLOW))
    else:
        ba = {'alert': False}

    # Predictions
    if not results.get('predictions'):
        print(colored("\n  ⚠️  Aucune prédiction (vérifier les erreurs)", Colors.YELLOW))
    else:
        print(colored("\n  PRÉDICTIONS :", Colors.BOLD))
        print(f"  {'Horizon':<12} {'Probabilité':>12} {'Seuil':>8} {'Alerte':>10}")
        print(f"  {'-' * 50}")

        order = ['15min', '30min', '60min', '360min', '3 jours']
        for h_key in order:
            if h_key not in results['predictions']:
                continue
            p = results['predictions'][h_key]
            prob = p['probability']
            th = p['threshold']
            alert = p['alert']

            # Color the alert
            if ba.get('alert') and ba.get('level') == 'CRITICAL':
                badge = colored("🚨 CRASH ACTUEL", Colors.RED + Colors.BOLD)
            elif alert:
                if prob >= 0.85:
                    badge = colored("⚠️ ALERTE", Colors.RED + Colors.BOLD)
                else:
                    badge = colored("⚠️ ALERTE", Colors.YELLOW)
            elif ba.get('alert'):
                badge = colored("◯ surveillé", Colors.YELLOW)
            else:
                if prob < 0.30:
                    badge = colored("✓ OK", Colors.GREEN)
                else:
                    badge = colored("◯ surveillé", Colors.BLUE)

            label = h_key
            if h_key == '15min': label = '15 min'
            elif h_key == '30min': label = '30 min'
            elif h_key == '60min': label = '1 heure'
            elif h_key == '360min': label = '6 heures'

            print(f"  {label:<12} {format_probability(prob):>12} {th:>8.4f}    {badge}")

        

    # Synthesis
    if results.get('predictions'):
        ml_alerts = [h for h, p in results['predictions'].items() if p.get('alert')]
        ba = results.get('business_alert') or {'alert': False}
        print("\n  " + "=" * 50)
        if ba.get('alert') and ba.get('level') == 'CRITICAL':
            print(colored(f"  🚨 SYNTHÈSE : CRITIQUE — {ba['message']}", Colors.RED + Colors.BOLD))
        elif ml_alerts:
            print(colored(f"  ⚠️ SYNTHÈSE : incident prédit ({ml_alerts[0]})", Colors.YELLOW + Colors.BOLD))
        elif ba.get('alert'):
            print(colored(f"  🟡 SYNTHÈSE : surveillance — {ba['message']}", Colors.YELLOW))
        else:
            print(colored("  ✅ SYNTHÈSE : système stable", Colors.GREEN))
        print("  " + "=" * 50)

    # Errors
    for k in ['ml_error', 'dl_error']:
        if k in results:
            print(colored(f"\n  ⚠️  {k}: {results[k]}", Colors.YELLOW))


def summarize_decision(results):
    predictions = results.get('predictions', {})
    ba = results.get('business_alert') or {'alert': False}
    ml_alerts = [h for h, p in predictions.items() if p.get('alert')]
    if ba.get('alert') and ba.get('level') == 'CRITICAL':
        return 'CRITICAL', f"CRITIQUE - {ba.get('message', '')}", 'business'
    if ml_alerts:
        return 'PREDICTED_INCIDENT', f"incident prédit ({ml_alerts[0]})", ml_alerts[0]
    if ba.get('alert'):
        return 'WATCH', f"surveillance - {ba.get('message', '')}", 'business'
    return 'STABLE', 'système stable', ''


def results_to_export_rows(results, source_csv, run_id):
    gt = results.get('ground_truth') or {}
    ba = results.get('business_alert') or {}
    decision_level, decision_message, decision_source = summarize_decision(results)
    rows = []
    for horizon, pred in results.get('predictions', {}).items():
        rows.append({
            'run_id': run_id,
            'source_csv': str(source_csv),
            'timestamp': results.get('timestamp'),
            'horizon': horizon,
            'horizon_min': pred.get('horizon_min'),
            'probability': pred.get('probability'),
            'threshold': pred.get('threshold'),
            'predicted_alert': bool(pred.get('alert')),
            'actual_status': gt.get('LOCAL_STATUS'),
            'actual_reason': gt.get('STATUS_REASON'),
            'actual_cpu': gt.get('CPU'),
            'actual_mem': gt.get('MEM'),
            'business_alert': bool(ba.get('alert')),
            'business_level': ba.get('level'),
            'business_message': ba.get('message'),
            'decision_level': decision_level,
            'decision_message': decision_message,
            'decision_source': decision_source,
            'model_note': pred.get('note', ''),
        })
    return rows


def export_results(export_rows, export_path, append=False):
    if not export_rows:
        return
    path = Path(export_path)
    if not path.is_absolute():
        path = SCRIPT_DIR.parent / path
    path.parent.mkdir(parents=True, exist_ok=True)
    df_export = pd.DataFrame(export_rows)
    write_header = not append or not path.exists()
    df_export.to_csv(path, mode='a' if append else 'w', index=False, header=write_header, encoding='utf-8')
    print(colored(f"\n  Export CSV: {path} ({len(df_export)} lignes)", Colors.BLUE))


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Test interactif des modèles HGW')
    parser.add_argument('csv', help='Path to CSV file (e.g. monitor_snapshots.csv)')
    parser.add_argument('--time', help='Timestamp to test (e.g. "2026-04-22 11:00:00")')
    parser.add_argument('--random', type=int, default=0,
                        help='Test on N random points')
    parser.add_argument('--last', action='store_true',
                        help='Test on the last available timestamp')
    parser.add_argument('--urgents', type=int, default=0,
                        help='Test on N timestamps just BEFORE URGENT episodes')
    parser.add_argument('--incidents', type=int, default=0,
                        help='Test on N timestamps just BEFORE CRITICAL/URGENT episodes')
    parser.add_argument('--export', default=str(SCRIPT_DIR.parent / 'data' / 'predictions_now.csv'),
                        help='CSV export path for notebook visualization')
    parser.add_argument('--append-export', action='store_true',
                        help='Append to export CSV instead of replacing it')
    args = parser.parse_args()

    # Load data
    print(colored(f"Chargement de {args.csv}...", Colors.BLUE))
    input_path = resolve_input_path(args.csv)
    df_original = read_any_table(input_path)
    df = normalize_input_dataframe(df_original)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"  Statuts: {dict(df['LOCAL_STATUS'].value_counts())}")
    print(f"  {len(df)} lignes, période {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Load models
    models = load_all_models()

    # Determine test timestamps
    timestamps_to_test = []

    if args.time:
        timestamps_to_test.append(pd.Timestamp(args.time))

    if args.last:
        timestamps_to_test.append(df['timestamp'].max())

    if args.random > 0:
        # Pick N random timestamps from the second half (need history before)
        valid_idx = df.index[df.index > 100]
        if len(valid_idx) > 0:
            picked = np.random.RandomState(42).choice(
                valid_idx, size=min(args.random, len(valid_idx)), replace=False
            )
            for idx in picked:
                timestamps_to_test.append(df.loc[idx, 'timestamp'])

    if args.urgents > 0:
        # Pick N timestamps just BEFORE URGENT episodes (the interesting ones!)
        df['urgent_change'] = (
            (df['LOCAL_STATUS'].astype(str).str.upper() == 'URGENT') &
            (df['LOCAL_STATUS'].astype(str).str.upper().shift(1) != 'URGENT')
        )
        urgent_starts = df[df['urgent_change']]['timestamp'].tolist()
        # Take 5 minutes BEFORE each URGENT start (to test prediction)
        for ts in urgent_starts[:args.urgents]:
            ts_before = ts - pd.Timedelta(minutes=5)
            timestamps_to_test.append(ts_before)

    if args.incidents > 0:
        # Pick N timestamps just BEFORE incident episodes.
        incident_mask = df['LOCAL_STATUS'].astype(str).str.upper().isin(INCIDENT_STATUSES)
        df['incident_change'] = incident_mask & ~incident_mask.shift(1, fill_value=False)
        incident_starts = df[df['incident_change']]['timestamp'].tolist()
        for ts in incident_starts[:args.incidents]:
            timestamps_to_test.append(ts - pd.Timedelta(minutes=5))

    if not timestamps_to_test:
        print(colored("\nAucun timestamp choisi. Test par défaut sur le dernier point.",
                       Colors.YELLOW))
        timestamps_to_test.append(df['timestamp'].max())

    # Run predictions
    run_id = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    export_rows = []
    for ts in timestamps_to_test:
        results = predict_at_timestamp(df, ts, models)
        print_results(results)
        export_rows.extend(results_to_export_rows(results, input_path, run_id))

    export_results(export_rows, args.export, append=args.append_export)

    print("\n" + "=" * 75)
    print(colored("  Test terminé.", Colors.BOLD))
    print("=" * 75)
    print(f"\nUsage avancé :")
    print(f"  python test_models.py {args.csv} --random 10")
    print(f"  python test_models.py {args.csv} --urgents 5")
    print(f"  python test_models.py {args.csv} --incidents 5")
    print(f"  python test_models.py {args.csv} --time '2026-04-22 11:30:00'")


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(colored(f"\n  ❌ {e}", Colors.RED))
        sys.exit(1)
