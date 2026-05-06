"""
HGW Dataset Generator V2 — Production-Grade
=============================================
Generates 5 years of multi-gateway HGW telemetry with:
  - 5 gateway profiles (different firmware/ISP/region baselines)
  - 3 episode types: slow degradation, rapid failure, recovered drift
  - Process-level RSS columns (cwmp, dhcp, nemo)
  - Controlled noise injection: sensor errors + missing data gaps
  - Reboot and recovery markers
  - Multi-horizon labels (24h, 72h, 7d) + continuous TTF
  - Health score column (Grafana-ready)

Output:
    data/hgw_short_term.csv  — for 24h CatBoost model (1h step, 5 gateways)
    data/hgw_long_term.csv   — for 72h+ LSTM model (30min step, 5 gateways)

Usage:
    python 01_generate_datasets.py
    python 01_generate_datasets.py --gateways 5 --years 5
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)


# =============================================================
# GATEWAY PROFILES (5 distinct baselines)
# =============================================================
GATEWAY_PROFILES = {
    "HGW_001": {  # Standard urban
        "cpu_base": 30.0, "mem_base": 43.0, "ping_base": 30.0,
        "firmware": "v15.10.20", "region": "EU-W", "isp": "ISP_A",
    },
    "HGW_002": {  # Rural, slower link
        "cpu_base": 25.0, "mem_base": 40.0, "ping_base": 55.0,
        "firmware": "v15.10.22", "region": "EU-W", "isp": "ISP_A",
    },
    "HGW_003": {  # Heavy-use urban (gaming/streaming household)
        "cpu_base": 38.0, "mem_base": 50.0, "ping_base": 28.0,
        "firmware": "v15.10.20", "region": "EU-S", "isp": "ISP_B",
    },
    "HGW_004": {  # Low-use, older firmware
        "cpu_base": 22.0, "mem_base": 38.0, "ping_base": 35.0,
        "firmware": "v15.9.18", "region": "EU-N", "isp": "ISP_C",
    },
    "HGW_005": {  # New firmware, optimized
        "cpu_base": 27.0, "mem_base": 41.0, "ping_base": 22.0,
        "firmware": "v15.10.30", "region": "EU-W", "isp": "ISP_A",
    },
}

CPU_CIRCADIAN = {
    0:-1.4, 1:-3.9, 2:-4.7, 3:-4.8, 4:-4.8, 5:-4.7, 6:-4.8,
    7:-1.9, 8:-0.9, 9:-0.7, 10:-0.6, 11:-0.5, 12:-0.4, 13:-0.4,
    14:-0.5, 15:-0.6, 16:-0.5, 17:-0.6,
    18:4.3, 19:5.9, 20:6.3, 21:6.4, 22:6.6, 23:6.6,
}
MEM_CIRCADIAN = {h: v * 0.45 for h, v in CPU_CIRCADIAN.items()}
DOW_CPU = {0:-5.5, 1:-3.8, 2:-0.6, 3:2.1, 4:1.4, 5:4.5, 6:1.6}
DOW_MEM = {k: v * 0.35 for k, v in DOW_CPU.items()}

CRASH_MEM_THRESH = 90.0
CRASH_CPU_THRESH = 88.0


def schedule_episodes(n_hours, n_episodes, episode_types_dist):
    """Schedule episodes spread evenly across the full timeline with type variety."""
    # Distribute episodes evenly across the timeline using time bins
    n_bins = max(n_episodes, 4)
    bin_size = (n_hours - 700) // n_bins
    selected = []
    for i in range(n_episodes):
        bin_start = 500 + i * bin_size
        bin_end   = min(500 + (i + 1) * bin_size, n_hours - 200)
        if bin_end <= bin_start:
            continue
        candidate = int(np.random.randint(bin_start, bin_end))
        selected.append(candidate)

    types_list = []
    for k, p in episode_types_dist.items():
        types_list.extend([k] * int(p * n_episodes))
    while len(types_list) < n_episodes:
        types_list.append("slow")
    np.random.shuffle(types_list)

    episodes = []
    for crash_h, etype in zip(selected, types_list[:n_episodes]):
        if etype == "slow":
            drift_d = int(np.random.randint(10, 16))
            dur_h   = int(np.random.randint(20, 50))
            severity = float(np.random.uniform(0.85, 1.0))
            crashes  = True
        elif etype == "rapid":
            drift_d = int(np.random.randint(0, 2))  # 12-24h
            dur_h   = int(np.random.randint(8, 20))
            severity = float(np.random.uniform(0.90, 1.0))
            crashes  = True
        elif etype == "recovered":
            drift_d = int(np.random.randint(7, 12))
            dur_h   = 0
            severity = float(np.random.uniform(0.50, 0.75))
            crashes  = False
        else:
            drift_d = 1
            dur_h   = 0
            severity = float(np.random.uniform(0.35, 0.55))
            crashes  = False

        rapid_offset = drift_d * 24 if etype != "rapid" else int(np.random.randint(12, 24))
        episodes.append({
            "drift_start": max(0, crash_h - rapid_offset),
            "crash_start": crash_h if crashes else -1,
            "crash_end":   crash_h + dur_h if crashes else -1,
            "peak_at":     crash_h,
            "etype":       etype,
            "severity":    severity,
            "crashes":     bool(crashes),
        })
    return sorted(episodes, key=lambda e: e["peak_at"])


def degradation_curve(progress, severity=1.0):
    x = np.clip(progress, 0, 1)
    return severity * (1.0 / (1.0 + np.exp(-8.0 * (x - 0.45))))


def generate_one_gateway(gw_id, profile, n_steps, sph, start_ts, step_minutes,
                          n_episodes, episode_dist):
    """Generate telemetry for a single gateway."""
    n_hours = n_steps // sph
    episodes = schedule_episodes(n_hours, n_episodes, episode_dist)

    timestamps = pd.date_range(start_ts, periods=n_steps, freq=f"{step_minutes}min")
    phase       = np.full(n_steps, "normal", dtype="U10")
    degradation = np.zeros(n_steps)
    episode_id  = np.full(n_steps, -1, dtype=int)
    reboot_event = np.zeros(n_steps, dtype=int)
    recovery_phase = np.zeros(n_steps, dtype=int)

    for ei, ep in enumerate(episodes):
        d_start = ep["drift_start"] * sph
        peak    = ep["peak_at"] * sph
        c_end   = ep["crash_end"] * sph if ep["crashes"] else peak

        for s in range(max(0, d_start), min(peak, n_steps)):
            progress = (s - d_start) / max(1, peak - d_start)
            phase[s] = "drift"
            degradation[s] = degradation_curve(progress, ep["severity"])
            episode_id[s] = ei

        if ep["crashes"]:
            for s in range(max(0, peak), min(c_end, n_steps)):
                phase[s] = "crash"
                degradation[s] = ep["severity"]
                episode_id[s] = ei

            # Reboot at end of crash
            if c_end < n_steps:
                reboot_event[c_end] = 1
                # 24-hour recovery phase
                recovery_end = min(c_end + 24 * sph, n_steps)
                recovery_phase[c_end:recovery_end] = 1

    # AR(1) noise
    ar_cpu  = np.zeros(n_steps); ar_mem = np.zeros(n_steps); ar_ping = np.zeros(n_steps)
    for i in range(1, n_steps):
        ar_cpu[i]  = 0.65 * ar_cpu[i-1]  + np.random.normal(0, 4.5 * 0.55)
        ar_mem[i]  = 0.72 * ar_mem[i-1]  + np.random.normal(0, 3.0 * 0.50)
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

    cpu_healthy = profile["cpu_base"] + cpu_circ + cpu_dow + seasonal_cpu + ar_cpu
    cpu = cpu_healthy + degradation * (78.0 - profile["cpu_base"])
    cpu = np.clip(cpu, 5, 100)

    mem_healthy = profile["mem_base"] + mem_circ + mem_dow + seasonal_mem + ar_mem
    mem = mem_healthy + degradation * (92.0 - profile["mem_base"])
    mem = np.clip(mem, 10, 100)

    ping = profile["ping_base"] + ar_ping + degradation * 80.0
    spike_mask = (degradation > 0.5) & (np.random.random(n_steps) < 0.05)
    ping[spike_mask] += np.random.uniform(100, 400, spike_mask.sum())
    ping = np.clip(ping, 5, 2000)

    loss = np.abs(np.random.normal(0.55, 0.3, n_steps)) + degradation * 8.0
    pre_wan = degradation > 0.7
    loss[pre_wan] += np.random.exponential(3.0, pre_wan.sum())
    loss = np.clip(loss, 0, 100)

    wan_fail_prob = np.minimum(0.5, 0.001 + (loss / 100) * 0.4)
    wan = (np.random.random(n_steps) >= wan_fail_prob).astype(int)
    crash_mask = (phase == "crash")
    wan[crash_mask] = (np.random.random(crash_mask.sum()) >= 0.35).astype(int)

    # Process-level RSS (MB) — cwmp leaks during memory degradation
    cwmp_rss = 12.0 + degradation * 850.0 + np.random.normal(0, 1.5, n_steps)
    cwmp_rss = np.clip(cwmp_rss, 10, 920)
    dhcp_rss = 8.5 + np.random.normal(0, 0.4, n_steps)
    dhcp_rss = np.clip(dhcp_rss, 6, 14)
    nemo_rss = 22.0 + np.random.normal(0, 1.0, n_steps) + degradation * 10.0
    nemo_rss = np.clip(nemo_rss, 18, 60)

    # Apply post-reboot reset
    reboot_steps = np.where(reboot_event == 1)[0]
    for r in reboot_steps:
        end = min(r + 24 * sph, n_steps)
        cwmp_rss[r:end] = 12.0 + np.random.normal(0, 0.5, end - r)
        cwmp_rss[r:end] = np.clip(cwmp_rss[r:end], 10, 16)

    is_crash = ((mem >= CRASH_MEM_THRESH) | (cpu >= CRASH_CPU_THRESH) | crash_mask).astype(int)

    df = pd.DataFrame({
        "timestamp":     timestamps,
        "gateway_id":    gw_id,
        "firmware":      profile["firmware"],
        "region":        profile["region"],
        "isp":           profile["isp"],
        "cpu_load":      np.round(cpu, 3),
        "mem_used_pct":  np.round(mem, 3),
        "ping_latency":  np.round(ping, 3),
        "packet_loss":   np.round(loss, 4),
        "wan_status":    wan,
        "cwmp_rss_mb":   np.round(cwmp_rss, 2),
        "dhcp_rss_mb":   np.round(dhcp_rss, 2),
        "nemo_rss_mb":   np.round(nemo_rss, 2),
        "hour":          hours,
        "dow":           dows,
        "is_crash":      is_crash,
        "reboot_event":  reboot_event,
        "recovery_phase":recovery_phase,
        "episode_type":  np.array([episodes[ei]["etype"] if ei >= 0 else "normal"
                                    for ei in episode_id], dtype="U10"),
    })

    return df


def inject_noise_and_gaps(df, noise_pct=0.10, gap_pct=0.05):
    """Add sensor noise + missing data gaps (post-generation)."""
    n = len(df)
    metric_cols = ["cpu_load", "mem_used_pct", "ping_latency", "packet_loss"]

    # Noise injection on 10% of rows
    noise_idx = np.random.choice(n, int(n * noise_pct), replace=False)
    for col in metric_cols:
        std = df[col].std()
        df.loc[noise_idx, col] = df.loc[noise_idx, col] + np.random.normal(0, std * 0.02, len(noise_idx))

    # NaN gaps (5% of rows, in 2-6 step windows)
    n_gaps = int(n * gap_pct / 4)
    for _ in range(n_gaps):
        start = np.random.randint(0, n - 12)
        length = np.random.randint(2, 12)
        col = np.random.choice(metric_cols)
        df.loc[start:start+length, col] = np.nan

    # Forward-fill to mimic real interpolation
    df[metric_cols] = df[metric_cols].ffill().bfill()
    df[metric_cols] = df[metric_cols].clip(lower=0)
    return df


def add_engineered_columns(df, sph):
    """Add derived columns (slopes, rolling stats, TTF, labels, health score)."""
    df = df.sort_values(["gateway_id", "timestamp"]).reset_index(drop=True)
    W6 = 6 * sph
    W24 = 24 * sph

    parts = []
    for gw, group in df.groupby("gateway_id"):
        g = group.copy()
        g["cpu_mean_24h"] = g["cpu_load"].rolling(W24, min_periods=1).mean().round(3)
        g["ram_mean_24h"] = g["mem_used_pct"].rolling(W24, min_periods=1).mean().round(3)
        g["cpu_std_24h"]  = g["cpu_load"].rolling(W24, min_periods=1).std().round(3)
        g["ram_std_24h"]  = g["mem_used_pct"].rolling(W24, min_periods=1).std().round(3)
        g["cpu_slope_6h"] = (g["cpu_load"].diff(W6) / 6).round(4)
        g["ram_slope_6h"] = (g["mem_used_pct"].diff(W6) / 6).round(4)
        g["wan_instability_6h"] = g["wan_status"].eq(0).rolling(W6, min_periods=1).mean().round(4)

        # TTF computed per-gateway
        crash_idx_local = g.index[g["is_crash"] == 1].tolist()
        ttf = np.full(len(g), np.nan)
        ptr = 0
        for i, idx in enumerate(g.index):
            while ptr < len(crash_idx_local) and crash_idx_local[ptr] <= idx:
                ptr += 1
            if ptr < len(crash_idx_local):
                ttf[i] = (crash_idx_local[ptr] - idx) / sph
        g["ttf_hours"] = np.where(np.isnan(ttf), 720, np.round(ttf, 2))
        parts.append(g)

    df = pd.concat(parts).sort_index().reset_index(drop=True)

    df["incident_in_24h"] = (df["ttf_hours"] <= 24).astype(int)
    df["incident_in_72h"] = (df["ttf_hours"] <= 72).astype(int)
    df["incident_in_7d"]  = (df["ttf_hours"] <= 168).astype(int)

    n_cpu  = np.clip((df["cpu_load"] - 20) / 70, 0, 1)
    n_mem  = np.clip((df["mem_used_pct"] - 35) / 55, 0, 1)
    n_ping = np.clip((df["ping_latency"] - 20) / 200, 0, 1)
    n_loss = np.clip(df["packet_loss"] / 15, 0, 1)
    composite = 0.35*n_mem + 0.30*n_cpu + 0.20*n_ping + 0.15*n_loss
    df["health_score"] = ((1.0 - composite.clip(0, 1)) * 100).round(1)
    return df


def build_dataset(years, n_gateways, step_minutes, episodes_per_year_per_gw, episode_dist):
    sph = 60 // step_minutes
    n_steps = years * 365 * 24 * sph
    start_ts = pd.Timestamp("2020-01-01 00:00:00")

    parts = []
    profiles = list(GATEWAY_PROFILES.items())[:n_gateways]
    for gw_id, profile in profiles:
        print(f"  Generating {gw_id} ({profile['firmware']} / {profile['region']})...")
        n_episodes = years * episodes_per_year_per_gw
        df_gw = generate_one_gateway(
            gw_id, profile, n_steps, sph, start_ts, step_minutes,
            n_episodes, episode_dist
        )
        df_gw = inject_noise_and_gaps(df_gw)
        parts.append(df_gw)

    df = pd.concat(parts).reset_index(drop=True)
    df = add_engineered_columns(df, sph)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years",     type=int, default=5)
    parser.add_argument("--gateways",  type=int, default=5)
    parser.add_argument("--out-dir",   default="data")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    EPISODE_DIST = {"slow": 0.55, "rapid": 0.20, "recovered": 0.15, "transient_spike": 0.10}

    print(f"\n{'='*60}\nSHORT-TERM DATASET (1h step, for CatBoost 24h)\n{'='*60}")
    df_short = build_dataset(args.years, args.gateways, step_minutes=60,
                              episodes_per_year_per_gw=10, episode_dist=EPISODE_DIST)
    print(f"\n  Total rows: {len(df_short):,}  Cols: {df_short.shape[1]}")
    print(f"  Crashes: {df_short['is_crash'].sum():,} ({df_short['is_crash'].mean()*100:.2f}%)")
    print(f"  incident_in_24h: {df_short['incident_in_24h'].sum():,} positives")

    out_short = out / "hgw_short_term.csv"
    df_short.to_csv(out_short, index=False)
    print(f"  Saved -> {out_short}  ({os.path.getsize(out_short)/1e6:.1f} MB)")

    print(f"\n{'='*60}\nLONG-TERM DATASET (30min step, for LSTM 72h+)\n{'='*60}")
    df_long = build_dataset(args.years, args.gateways, step_minutes=30,
                              episodes_per_year_per_gw=10, episode_dist=EPISODE_DIST)
    print(f"\n  Total rows: {len(df_long):,}  Cols: {df_long.shape[1]}")
    print(f"  Crashes: {df_long['is_crash'].sum():,} ({df_long['is_crash'].mean()*100:.2f}%)")
    print(f"  incident_in_72h: {df_long['incident_in_72h'].sum():,} positives")
    print(f"  incident_in_7d: {df_long['incident_in_7d'].sum():,} positives")

    out_long = out / "hgw_long_term.csv"
    df_long.to_csv(out_long, index=False)
    print(f"  Saved -> {out_long}  ({os.path.getsize(out_long)/1e6:.1f} MB)")

    # Save metadata
    metadata = {
        "years": args.years,
        "n_gateways": args.gateways,
        "gateway_profiles": {k: v for k, v in profiles_dict_serializable().items()},
        "episode_distribution": EPISODE_DIST,
        "short_term": {
            "step_minutes": 60,
            "rows": len(df_short),
            "crashes": int(df_short["is_crash"].sum()),
        },
        "long_term": {
            "step_minutes": 30,
            "rows": len(df_long),
            "crashes": int(df_long["is_crash"].sum()),
        },
    }
    with open(out / "datasets_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata -> {out / 'datasets_metadata.json'}")


def profiles_dict_serializable():
    return {k: v for k, v in GATEWAY_PROFILES.items()}


if __name__ == "__main__":
    main()
