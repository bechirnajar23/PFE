"""
HGW Big Data Generator — 5 Years · 30-min step
================================================
Generates 87,600 rows of realistic HGW telemetry with:
  - Crashes distributed across ALL 5 years (not clustered)
  - Multi-scale degradation: drift (10-15 days) -> critical -> crash -> recovery
  - Realistic circadian + weekly + seasonal patterns
  - Network-CPU-RAM correlations matching real HGW behaviour
  - Health score column (Grafana-ready: 0-100%)
  - Multi-horizon target labels: 24h, 72h, 7d + continuous TTF

Output: data/hgw_5yr_bigdata.csv  (~11 MB)
"""

import numpy as np
import pandas as pd
import os
import json

np.random.seed(42)

STEP_MINUTES   = 30
STEPS_PER_HOUR = 60 // STEP_MINUTES
YEARS          = 5
TOTAL_STEPS    = YEARS * 365 * 24 * STEPS_PER_HOUR
START_TS       = pd.Timestamp("2020-01-01 00:00:00")

CPU_BASE, CPU_STD = 30.0, 4.5
MEM_BASE, MEM_STD = 43.0, 3.0
PING_BASE         = 30.0
LOSS_BASE, LOSS_STD = 0.55, 0.3

CPU_CIRCADIAN = {
    0:-1.4,1:-3.9,2:-4.7,3:-4.8,4:-4.8,5:-4.7,6:-4.8,
    7:-1.9,8:-0.9,9:-0.7,10:-0.6,11:-0.5,12:-0.4,13:-0.4,
    14:-0.5,15:-0.6,16:-0.5,17:-0.6,
    18:4.3,19:5.9,20:6.3,21:6.4,22:6.6,23:6.6
}
MEM_CIRCADIAN = {h: v * 0.45 for h, v in CPU_CIRCADIAN.items()}
DOW_CPU = {0:-5.5,1:-3.8,2:-0.6,3:2.1,4:1.4,5:4.5,6:1.6}
DOW_MEM = {k: v * 0.35 for k, v in DOW_CPU.items()}

CRASH_MEM_THRESH = 90.0
CRASH_CPU_THRESH = 88.0

N_EPISODES_PER_YEAR = 10
MIN_GAP_HOURS  = 250
DRIFT_DAYS_MIN = 10
DRIFT_DAYS_MAX = 16
EP_DUR_MIN     = 18
EP_DUR_MAX     = 48


def schedule_episodes(total_steps):
    episodes_hours = []
    for year_idx in range(YEARS):
        year_start = year_idx * 365 * 24 + 500
        year_end   = (year_idx + 1) * 365 * 24 - 200
        n = N_EPISODES_PER_YEAR + np.random.randint(-1, 2)
        candidates = sorted(np.random.choice(
            range(year_start, year_end), size=n * 4, replace=False
        ))
        last = -MIN_GAP_HOURS * 2
        added = 0
        for c in candidates:
            if c - last >= MIN_GAP_HOURS and added < n:
                episodes_hours.append(c)
                last = c
                added += 1

    out = []
    for crash_h in sorted(episodes_hours):
        drift_d = int(np.random.randint(DRIFT_DAYS_MIN, DRIFT_DAYS_MAX + 1))
        dur_h   = int(np.random.randint(EP_DUR_MIN, EP_DUR_MAX + 1))
        out.append({
            'drift_start': max(0, crash_h - drift_d * 24),
            'crash_start': crash_h,
            'crash_end':   crash_h + dur_h,
            'drift_days':  drift_d,
            'dur_hours':   dur_h,
            'severity':    float(np.random.uniform(0.85, 1.0)),
        })
    return out


def degradation_curve(progress, severity=1.0):
    x = np.clip(progress, 0, 1)
    return severity * (1.0 / (1.0 + np.exp(-8.0 * (x - 0.45))))


def generate():
    print(f"Generating {TOTAL_STEPS:,} rows ({YEARS} years x {24*STEPS_PER_HOUR} steps/day)...")
    episodes = schedule_episodes(TOTAL_STEPS)
    print(f"Scheduled {len(episodes)} degradation episodes across {YEARS} years")

    n = TOTAL_STEPS
    timestamps = pd.date_range(START_TS, periods=n, freq=f'{STEP_MINUTES}min')
    phase       = np.full(n, 'normal', dtype='U10')
    degradation = np.zeros(n)
    episode_id  = np.full(n, -1, dtype=int)

    for ei, ep in enumerate(episodes):
        d_start = ep['drift_start'] * STEPS_PER_HOUR
        c_start = ep['crash_start'] * STEPS_PER_HOUR
        c_end   = min(ep['crash_end'] * STEPS_PER_HOUR, n)
        drift_len = c_start - d_start

        for s in range(max(0, d_start), min(c_start, n)):
            progress = (s - d_start) / max(1, drift_len)
            phase[s] = 'drift'
            degradation[s] = degradation_curve(progress, ep['severity'])
            episode_id[s] = ei

        for s in range(max(0, c_start), min(c_end, n)):
            phase[s] = 'crash'
            degradation[s] = ep['severity']
            episode_id[s] = ei

    ar_cpu  = np.zeros(n); ar_mem = np.zeros(n); ar_ping = np.zeros(n)
    for i in range(1, n):
        ar_cpu[i]  = 0.65 * ar_cpu[i-1]  + np.random.normal(0, CPU_STD * 0.55)
        ar_mem[i]  = 0.72 * ar_mem[i-1]  + np.random.normal(0, MEM_STD * 0.50)
        ar_ping[i] = 0.50 * ar_ping[i-1] + np.random.normal(0, 8.0 * 0.60)

    hours = timestamps.hour
    dows  = timestamps.dayofweek
    doy   = timestamps.dayofyear

    cpu_circ = np.array([CPU_CIRCADIAN[h] for h in hours])
    mem_circ = np.array([MEM_CIRCADIAN[h] for h in hours])
    cpu_dow  = np.array([DOW_CPU[d] for d in dows])
    mem_dow  = np.array([DOW_MEM[d] for d in dows])
    seasonal_cpu = 3.0 * np.sin(2 * np.pi * doy / 365 - np.pi/2)
    seasonal_mem = 2.0 * np.sin(2 * np.pi * doy / 365 - np.pi/2)

    cpu_healthy = CPU_BASE + cpu_circ + cpu_dow + seasonal_cpu + ar_cpu
    cpu = cpu_healthy + degradation * (78.0 - CPU_BASE)
    cpu = np.clip(cpu, 5, 100)

    mem_healthy = MEM_BASE + mem_circ + mem_dow + seasonal_mem + ar_mem
    mem = mem_healthy + degradation * (92.0 - MEM_BASE)
    mem = np.clip(mem, 10, 100)

    ping = PING_BASE + ar_ping + degradation * 80.0
    spike_mask = (degradation > 0.5) & (np.random.random(n) < 0.05)
    ping[spike_mask] += np.random.uniform(100, 400, spike_mask.sum())
    ping = np.clip(ping, 5, 2000)

    loss = np.abs(np.random.normal(LOSS_BASE, LOSS_STD, n)) + degradation * 8.0
    pre_wan = degradation > 0.7
    loss[pre_wan] += np.random.exponential(3.0, pre_wan.sum())
    loss = np.clip(loss, 0, 100)

    wan_fail_prob = np.minimum(0.5, 0.001 + (loss / 100) * 0.4)
    wan = (np.random.random(n) >= wan_fail_prob).astype(int)
    crash_mask = (phase == 'crash')
    wan[crash_mask] = (np.random.random(crash_mask.sum()) >= 0.35).astype(int)

    is_crash = ((mem >= CRASH_MEM_THRESH) | (cpu >= CRASH_CPU_THRESH) | crash_mask).astype(int)

    df = pd.DataFrame({
        'timestamp':    timestamps,
        'gateway_id':   'HGW_001',
        'cpu_load':     np.round(cpu, 3),
        'mem_used_pct': np.round(mem, 3),
        'ping_latency': np.round(ping, 3),
        'packet_loss':  np.round(loss, 4),
        'wan_status':   wan,
        'hour':         hours,
        'dow':          dows,
        'is_crash':     is_crash,
    })
    print(f"Raw signals: {df.shape}")

    W6  = 6  * STEPS_PER_HOUR
    W24 = 24 * STEPS_PER_HOUR

    df['cpu_mean_24h'] = df['cpu_load'].rolling(W24, min_periods=1).mean().round(3)
    df['ram_mean_24h'] = df['mem_used_pct'].rolling(W24, min_periods=1).mean().round(3)
    df['cpu_std_24h']  = df['cpu_load'].rolling(W24, min_periods=1).std().round(3)
    df['ram_std_24h']  = df['mem_used_pct'].rolling(W24, min_periods=1).std().round(3)
    df['cpu_slope_6h'] = (df['cpu_load'].diff(W6) / 6).round(4)
    df['ram_slope_6h'] = (df['mem_used_pct'].diff(W6) / 6).round(4)
    df['wan_instability_6h'] = (
        df['wan_status'].eq(0).rolling(W6, min_periods=1).mean().round(4)
    )

    crash_indices = df.index[df['is_crash'] == 1].tolist()
    ttf = np.full(len(df), np.nan)
    ptr = 0
    for i in range(len(df)):
        while ptr < len(crash_indices) and crash_indices[ptr] <= i:
            ptr += 1
        if ptr < len(crash_indices):
            ttf[i] = (crash_indices[ptr] - i) / STEPS_PER_HOUR
    df['ttf_hours'] = np.where(np.isnan(ttf), 720, np.round(ttf, 2))

    df['incident_in_24h'] = (df['ttf_hours'] <= 24).astype(int)
    df['incident_in_72h'] = (df['ttf_hours'] <= 72).astype(int)
    df['incident_in_7d']  = (df['ttf_hours'] <= 168).astype(int)

    n_cpu  = np.clip((df['cpu_load']     - 20) / 70, 0, 1)
    n_mem  = np.clip((df['mem_used_pct'] - 35) / 55, 0, 1)
    n_ping = np.clip((df['ping_latency'] - 20) / 200, 0, 1)
    n_loss = np.clip(df['packet_loss'] / 15, 0, 1)
    composite = 0.35*n_mem + 0.30*n_cpu + 0.20*n_ping + 0.15*n_loss
    df['health_score'] = ((1.0 - composite.clip(0, 1)) * 100).round(1)

    df = df[[
        'timestamp','gateway_id',
        'cpu_load','mem_used_pct','ping_latency','packet_loss','wan_status',
        'hour','dow','is_crash',
        'cpu_mean_24h','ram_mean_24h','cpu_std_24h','ram_std_24h',
        'cpu_slope_6h','ram_slope_6h','wan_instability_6h',
        'incident_in_24h','incident_in_72h','incident_in_7d',
        'ttf_hours','health_score'
    ]]
    return df, episodes


def diagnostics(df, episodes):
    print("\n" + "="*60)
    print("CALIBRATION DIAGNOSTICS")
    print("="*60)
    normal = df[df['is_crash'] == 0]
    crash  = df[df['is_crash'] == 1]

    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    print(f"\n-- Baseline (normal rows) --")
    print(f"  CPU:  mean={normal['cpu_load'].mean():.1f}  std={normal['cpu_load'].std():.1f}")
    print(f"  MEM:  mean={normal['mem_used_pct'].mean():.1f}  std={normal['mem_used_pct'].std():.1f}")
    print(f"  PING: mean={normal['ping_latency'].mean():.1f}  std={normal['ping_latency'].std():.1f}")

    print(f"\n-- Crash distribution --")
    print(f"  Total crashes: {crash.shape[0]:,} ({crash.shape[0]/df.shape[0]*100:.2f}%)")
    print(f"  Episodes: {len(episodes)}")
    by_year = df.groupby(df['timestamp'].dt.year)['is_crash'].sum()
    print(f"  Crashes per year:")
    for y, c in by_year.items():
        print(f"    {y}: {c}")

    print(f"\n-- Label distribution --")
    for col in ['incident_in_24h', 'incident_in_72h', 'incident_in_7d']:
        pos = df[col].sum()
        print(f"  {col}: {pos:,} ({pos/len(df)*100:.2f}%)")

    print(f"\n-- Health score --")
    print(f"  Normal: mean={normal['health_score'].mean():.1f}%  min={normal['health_score'].min():.1f}%")
    print(f"  Crash:  mean={crash['health_score'].mean():.1f}%   min={crash['health_score'].min():.1f}%")


if __name__ == "__main__":
    df, episodes = generate()
    diagnostics(df, episodes)

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "hgw_5yr_bigdata.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv} ({os.path.getsize(out_csv) / 1e6:.1f} MB)")

    out_json = os.path.join(out_dir, "episodes.json")
    def _to_native(obj):
        if isinstance(obj, dict):  return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_to_native(v) for v in obj]
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj
    with open(out_json, 'w') as f:
        json.dump(_to_native(episodes), f, indent=2)
    print(f"Saved -> {out_json}")
