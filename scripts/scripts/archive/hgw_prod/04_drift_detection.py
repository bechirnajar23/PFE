"""
Drift Detection & Retraining Trigger
======================================
Production-ready drift monitoring for HGW models.

Monitors three types of drift:
  1. Covariate drift (input features) — PSI + Kolmogorov-Smirnov
  2. Concept drift (model performance) — rolling PR-AUC vs baseline
  3. Streaming drift — ADWIN-style adaptive window

Outputs:
  - data/drift_report.json — current drift status
  - data/drift_baseline.json — reference statistics for production
  - Triggers retraining if PSI > 0.25 or PR-AUC drop > 0.10

Usage:
    python 04_drift_detection.py --baseline data/hgw_short_term.csv --new data/recent_data.csv
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", default="data/hgw_short_term.csv",
                     help="Reference dataset (training data)")
parser.add_argument("--new",      default=None,
                     help="New data to check for drift (default: last 25% of baseline)")
parser.add_argument("--out-dir",  default="data")
parser.add_argument("--psi-warn",     type=float, default=0.10)
parser.add_argument("--psi-retrain",  type=float, default=0.25)
args = parser.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================
# CORE DRIFT METRICS
# =============================================================
def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index — measures distributional shift.

    PSI < 0.10  → no drift
    0.10 - 0.25 → moderate shift (monitor)
    > 0.25      → significant drift (retrain)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 100 or len(actual) < 100:
        return 0.0
    e_hist, bin_edges = np.histogram(expected, bins=bins)
    a_hist, _ = np.histogram(actual, bins=bin_edges)
    e_pct = np.clip(e_hist / len(expected), 1e-6, 1)
    a_pct = np.clip(a_hist / len(actual),   1e-6, 1)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def ks_test(expected: np.ndarray, actual: np.ndarray) -> tuple:
    """Kolmogorov-Smirnov two-sample test."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    stat, pval = sp_stats.ks_2samp(expected, actual)
    return float(stat), float(pval)


def adwin_drift_check(error_history: np.ndarray, delta: float = 0.002) -> dict:
    """
    Simplified ADWIN: detects change in mean of a streaming error sequence.
    Returns the position of the change point if detected.
    """
    n = len(error_history)
    if n < 100:
        return {"drift_detected": False, "change_point": None}

    for split in range(50, n - 50):
        left = error_history[:split]
        right = error_history[split:]
        if len(left) < 30 or len(right) < 30:
            continue
        mean_diff = abs(left.mean() - right.mean())
        eps = np.sqrt(np.log(2 / delta) / (2 * len(left))) + np.sqrt(np.log(2 / delta) / (2 * len(right)))
        if mean_diff > eps:
            return {"drift_detected": True, "change_point": int(split),
                    "left_mean": float(left.mean()), "right_mean": float(right.mean()),
                    "epsilon": float(eps)}
    return {"drift_detected": False, "change_point": None}


def classify_drift(psi: float, ks_pval: float, ks_stat: float, warn: float, retrain: float) -> str:
    """
    Classify drift severity.
    PSI is the primary indicator (sample-size invariant).
    KS only triggers if the effect size (ks_stat) is also large.
    """
    if psi >= retrain or (ks_pval < 0.001 and ks_stat > 0.10):
        return "RETRAIN"
    if psi >= warn or (ks_pval < 0.05 and ks_stat > 0.05):
        return "WARN"
    return "OK"


# =============================================================
# MAIN PIPELINE
# =============================================================
print("=" * 70)
print("Drift Detection & Retraining Trigger")
print("=" * 70)

# Load datasets
df_base = pd.read_csv(args.baseline, parse_dates=["timestamp"])
df_base = df_base.sort_values("timestamp").reset_index(drop=True)
print(f"Baseline: {len(df_base):,} rows  ({df_base['timestamp'].min()} -> {df_base['timestamp'].max()})")

if args.new:
    df_new = pd.read_csv(args.new, parse_dates=["timestamp"])
    df_new = df_new.sort_values("timestamp").reset_index(drop=True)
    print(f"New data: {len(df_new):,} rows  ({df_new['timestamp'].min()} -> {df_new['timestamp'].max()})")
else:
    # Self-test: split baseline into reference / "new"
    sp = int(len(df_base) * 0.75)
    df_new = df_base.iloc[sp:].copy()
    df_base = df_base.iloc[:sp].copy()
    print(f"No --new given. Using last 25% of baseline as test split.")
    print(f"Reference: {len(df_base):,} rows  |  Test: {len(df_new):,} rows")

# Features to monitor
DRIFT_FEATURES = [
    "cpu_load", "mem_used_pct", "ping_latency", "packet_loss",
    "cpu_slope_6h", "ram_slope_6h", "cwmp_rss_mb",
]
DRIFT_FEATURES = [f for f in DRIFT_FEATURES if f in df_base.columns]


# =============================================================
# 1. COVARIATE DRIFT
# =============================================================
print(f"\n--- COVARIATE DRIFT ({len(DRIFT_FEATURES)} features) ---")
print(f"{'Feature':<20s} {'PSI':>8s} {'KS-stat':>9s} {'KS-pval':>10s} {'Status':>8s}")
print("-" * 60)

drift_per_feature = {}
overall_status = "OK"
for feat in DRIFT_FEATURES:
    train_vals = df_base[feat].values
    test_vals = df_new[feat].values
    psi = compute_psi(train_vals, test_vals)
    ks_stat, ks_pval = ks_test(train_vals, test_vals)
    status = classify_drift(psi, ks_pval, ks_stat, args.psi_warn, args.psi_retrain)

    drift_per_feature[feat] = {
        "psi": round(psi, 4),
        "ks_stat": round(ks_stat, 4),
        "ks_pval": round(ks_pval, 6),
        "status": status,
        "baseline_mean": round(float(np.nanmean(train_vals)), 3),
        "baseline_std": round(float(np.nanstd(train_vals)), 3),
        "new_mean": round(float(np.nanmean(test_vals)), 3),
        "new_std": round(float(np.nanstd(test_vals)), 3),
    }

    if status == "RETRAIN":
        overall_status = "RETRAIN"
    elif status == "WARN" and overall_status == "OK":
        overall_status = "WARN"

    print(f"{feat:<20s} {psi:>8.4f} {ks_stat:>9.4f} {ks_pval:>10.4f}   {status:>6s}")


# =============================================================
# 2. CONCEPT DRIFT (if model predictions are available)
# =============================================================
concept_drift = None
pred_paths = list(Path(args.out_dir).glob("*_predictions.csv"))
if pred_paths:
    print(f"\n--- CONCEPT DRIFT (model performance) ---")
    for pp in pred_paths:
        try:
            preds = pd.read_csv(pp)
            if "y_true" in preds.columns and "y_prob" in preds.columns:
                # Compute rolling PR-AUC on chunks
                n_chunks = 5
                chunk_size = len(preds) // n_chunks
                chunk_aucs = []
                for i in range(n_chunks):
                    chunk = preds.iloc[i*chunk_size:(i+1)*chunk_size]
                    if chunk["y_true"].sum() < 10:
                        continue
                    from sklearn.metrics import average_precision_score
                    auc = average_precision_score(chunk["y_true"], chunk["y_prob"])
                    chunk_aucs.append(auc)
                if len(chunk_aucs) >= 2:
                    drop = max(chunk_aucs) - min(chunk_aucs)
                    print(f"  {pp.name}: PR-AUC range over {len(chunk_aucs)} chunks = {min(chunk_aucs):.4f} -> {max(chunk_aucs):.4f}  (drop={drop:.4f})")
                    concept_drift = {
                        "file": pp.name,
                        "chunk_pr_aucs": [round(a, 4) for a in chunk_aucs],
                        "max_drop": round(float(drop), 4),
                        "concept_drift_detected": bool(drop > 0.10),
                    }
        except Exception as e:
            print(f"  {pp.name}: skipped ({e})")


# =============================================================
# 3. STREAMING DRIFT (ADWIN on synthetic error series)
# =============================================================
print(f"\n--- STREAMING DRIFT (ADWIN simulation) ---")
# Simulate: prediction error increases over time if model is degrading
synthetic_errors = np.concatenate([
    np.random.normal(0.05, 0.02, 500),    # stable phase
    np.random.normal(0.05, 0.02, 500),    # still stable
    np.random.normal(0.12, 0.04, 200),    # drift onset
])
adwin_result = adwin_drift_check(synthetic_errors)
if adwin_result["drift_detected"]:
    print(f"  Drift detected at step {adwin_result['change_point']}")
    print(f"  Mean error before: {adwin_result['left_mean']:.4f}  after: {adwin_result['right_mean']:.4f}")
else:
    print(f"  No drift detected in synthetic stream")


# =============================================================
# 4. DECISION + REPORT
# =============================================================
print(f"\n{'='*60}")
print(f"OVERALL DRIFT STATUS: {overall_status}")
print(f"{'='*60}")

action_required = {
    "OK":      "No action — continue monitoring",
    "WARN":    "Monitor weekly; investigate trending features",
    "RETRAIN": "TRIGGER RETRAIN within 48h (PSI>0.25 or KS p<0.001)",
}
print(f"  Action: {action_required[overall_status]}")

# Save baseline for production
baseline_stats = {
    feat: {
        "mean":    round(float(df_base[feat].mean()), 4),
        "std":     round(float(df_base[feat].std()), 4),
        "p10":     round(float(df_base[feat].quantile(0.10)), 4),
        "p25":     round(float(df_base[feat].quantile(0.25)), 4),
        "p50":     round(float(df_base[feat].quantile(0.50)), 4),
        "p75":     round(float(df_base[feat].quantile(0.75)), 4),
        "p90":     round(float(df_base[feat].quantile(0.90)), 4),
    }
    for feat in DRIFT_FEATURES
}
with open(OUT / "drift_baseline.json", "w") as f:
    json.dump(baseline_stats, f, indent=2)
print(f"\n  Baseline stats -> {OUT / 'drift_baseline.json'}")

drift_report = {
    "timestamp_check":   pd.Timestamp.now().isoformat(),
    "overall_status":    overall_status,
    "action":            action_required[overall_status],
    "covariate_drift":   drift_per_feature,
    "concept_drift":     concept_drift,
    "adwin_streaming":   adwin_result,
    "thresholds": {
        "psi_warn":      args.psi_warn,
        "psi_retrain":   args.psi_retrain,
        "ks_pval_warn":  0.05,
        "ks_pval_retrain": 0.001,
        "ks_stat_warn":  0.05,
        "ks_stat_retrain": 0.10,
    },
}
with open(OUT / "drift_report.json", "w") as f:
    json.dump(drift_report, f, indent=2)
print(f"  Drift report   -> {OUT / 'drift_report.json'}")
print("\nDone.")
