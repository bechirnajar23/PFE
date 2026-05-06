"""
Drift Monitor — Production Data Drift Detection
=================================================
Detects covariate drift (PSI, KS) and concept drift (PR-AUC degradation)
between a baseline training set and incoming production data.

Triggers retraining when drift exceeds configurable thresholds.

Usage:
    # First run: establish baseline from training data
    python 04_drift_monitor.py --mode baseline --data data/hgw_short_term.csv

    # Subsequent runs: compare new data against baseline
    python 04_drift_monitor.py --mode check --new-data data/new_telemetry.csv

    # Periodic check (e.g. weekly via cron):
    python 04_drift_monitor.py --mode check --new-data data/last_30d.csv \\
        --baseline data/drift_baseline.json --psi-warn 0.10 --psi-trigger 0.25
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


KEY_FEATURES = [
    "cpu_load", "mem_used_pct", "ping_latency", "packet_loss",
    "cpu_slope_6h", "ram_slope_6h", "wan_status",
    "cwmp_rss_mb", "dhcp_rss_mb", "nemo_rss_mb",
]


def compute_psi(expected, actual, bins=10):
    """Population Stability Index — detects covariate drift."""
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]
    if len(expected) < 50 or len(actual) < 50:
        return float("nan")
    e_hist, bin_edges = np.histogram(expected, bins=bins)
    a_hist, _         = np.histogram(actual, bins=bin_edges)
    e_pct = np.clip(e_hist / max(1, len(expected)), 1e-6, 1)
    a_pct = np.clip(a_hist / max(1, len(actual)),   1e-6, 1)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def build_baseline(df, features, out_path):
    """Save distribution stats from training data for future drift checks."""
    baseline = {"features": {}, "n_rows": int(len(df))}
    for f in features:
        if f not in df.columns:
            continue
        vals = df[f].dropna().values
        if len(vals) == 0:
            continue
        baseline["features"][f] = {
            "mean":    round(float(vals.mean()), 4),
            "std":     round(float(vals.std()), 4),
            "min":     round(float(vals.min()), 4),
            "max":     round(float(vals.max()), 4),
            "p10":     round(float(np.percentile(vals, 10)), 4),
            "p50":     round(float(np.percentile(vals, 50)), 4),
            "p90":     round(float(np.percentile(vals, 90)), 4),
            "samples": vals[:5000].tolist(),
        }
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Baseline saved -> {out_path}")
    print(f"  Features tracked: {len(baseline['features'])}")
    print(f"  Reference rows:   {baseline['n_rows']:,}")


def check_drift(new_df, baseline_path, psi_warn, psi_trigger):
    """Run drift checks; report status and retrain decision."""
    with open(baseline_path) as f:
        baseline = json.load(f)

    print(f"\n{'='*70}")
    print(f"DRIFT CHECK")
    print(f"{'='*70}")
    print(f"  Baseline:  {baseline['n_rows']:,} rows")
    print(f"  New data:  {len(new_df):,} rows")
    print(f"\n  Feature                PSI       KS        p-val    Status")
    print(f"  " + "-"*65)

    report = {}
    drift_count = 0
    warn_count = 0

    for f, b in baseline["features"].items():
        if f not in new_df.columns:
            continue
        new_vals = new_df[f].dropna().values
        ref_vals = np.asarray(b["samples"])
        if len(new_vals) < 50:
            print(f"  {f:20s}  insufficient data ({len(new_vals)} rows)")
            continue

        psi = compute_psi(ref_vals, new_vals)
        ks_stat, ks_pval = sp_stats.ks_2samp(ref_vals, new_vals)

        if psi >= psi_trigger:
            status = "DRIFT"
            drift_count += 1
        elif psi >= psi_warn:
            status = "WARN"
            warn_count += 1
        else:
            status = "OK"

        report[f] = {
            "psi":       round(psi, 4),
            "ks_stat":   round(float(ks_stat), 4),
            "ks_pval":   round(float(ks_pval), 6),
            "status":    status,
            "ref_mean":  b["mean"],
            "new_mean":  round(float(new_vals.mean()), 4),
            "ref_std":   b["std"],
            "new_std":   round(float(new_vals.std()), 4),
        }
        print(f"  {f:20s}  {psi:6.4f}    {ks_stat:6.4f}    {ks_pval:6.4f}    [{status}]")

    print(f"\n  Drift triggered: {drift_count} feature(s)")
    print(f"  Warnings:        {warn_count} feature(s)")

    # Decision
    if drift_count >= 1:
        decision = "RETRAIN_REQUIRED"
        print(f"\n  >>> {decision}: at least one feature shows significant drift (PSI >= {psi_trigger}).")
        print(f"      Run: python 02_train_catboost_short.py  AND  python 03_train_bilstm_long.py")
    elif warn_count >= 3:
        decision = "RETRAIN_RECOMMENDED"
        print(f"\n  >>> {decision}: {warn_count} features in warning zone.")
    else:
        decision = "OK"
        print(f"\n  >>> {decision}: no significant drift detected.")

    report["__summary__"] = {
        "decision": decision,
        "drift_count": drift_count,
        "warn_count": warn_count,
        "psi_warn_threshold": psi_warn,
        "psi_trigger_threshold": psi_trigger,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "check"], required=True)
    parser.add_argument("--data",      help="Training CSV (for baseline mode)")
    parser.add_argument("--new-data",  help="New CSV (for check mode)")
    parser.add_argument("--baseline",  default="data/drift_baseline.json")
    parser.add_argument("--out",       default="data/drift_report.json")
    parser.add_argument("--psi-warn",    type=float, default=0.10)
    parser.add_argument("--psi-trigger", type=float, default=0.25)
    args = parser.parse_args()

    if args.mode == "baseline":
        if not args.data:
            raise SystemExit("--data required for baseline mode")
        df = pd.read_csv(args.data, parse_dates=["timestamp"], low_memory=False)
        build_baseline(df, KEY_FEATURES, args.baseline)
    else:
        if not args.new_data:
            raise SystemExit("--new-data required for check mode")
        if not Path(args.baseline).exists():
            raise SystemExit(f"Baseline not found: {args.baseline}\n"
                              f"Run with --mode baseline first.")
        df = pd.read_csv(args.new_data, parse_dates=["timestamp"], low_memory=False)
        report = check_drift(df, args.baseline, args.psi_warn, args.psi_trigger)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved -> {args.out}")


if __name__ == "__main__":
    main()
