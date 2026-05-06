"""
HGW Predictive Maintenance — Advanced Pipeline v2
====================================================
Pillars:
  1. Advanced Feature Engineering (entropy, saturation ratios, wavelet features)
  2. Concept/Data Drift Detection (PSI, KS test, ADWIN)
  3. Expanded Benchmark (LightGBM, CatBoost, IsolationForest pre-filter)
  4. SHAP Explainability + Health Score

Requirements:
    pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna shap scipy

Usage:
    python advanced_pipeline.py                         # full run
    python advanced_pipeline.py --skip-catboost         # if catboost install fails
    python advanced_pipeline.py --skip-shap             # faster run without SHAP
    python advanced_pipeline.py --trials 50             # more Optuna trials
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data',           default='data/hgw_5yr_bigdata.csv')
parser.add_argument('--out-dir',        default='data')
parser.add_argument('--trials',         type=int, default=30)
parser.add_argument('--skip-catboost',  action='store_true')
parser.add_argument('--skip-shap',      action='store_true')
parser.add_argument('--skip-optuna',    action='store_true')
parser.add_argument('--skip-iforest',   action='store_true')
args = parser.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================
# 1. LOAD DATA
# =============================================================
print("=" * 70)
print("PILLAR 1: Advanced Feature Engineering")
print("=" * 70)

df = pd.read_csv(args.data, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
SPH = 2  # steps per hour (30-min intervals)
print(f"Loaded: {len(df):,} rows  x  {df.shape[1]} cols")


# =============================================================
# 1A. STANDARD FEATURES (lags, rolling, slopes)
# =============================================================
print("\n[1a] Standard temporal features...")

parts = []
for col, sh in [('cpu_load','cpu'), ('mem_used_pct','mem'),
                ('ping_latency','ping'), ('packet_loss','loss')]:
    d = {}
    # Multi-horizon lags
    for lag_h in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        d[f'{sh}_lag{lag_h}h'] = df[col].shift(lag_h * SPH)

    # Rolling stats at multiple windows
    for win_h in [6, 24, 72, 168]:
        w = win_h * SPH
        d[f'{sh}_ma{win_h}h']  = df[col].rolling(w, min_periods=1).mean()
        d[f'{sh}_std{win_h}h'] = df[col].rolling(w, min_periods=1).std()
        d[f'{sh}_max{win_h}h'] = df[col].rolling(w, min_periods=1).max()
        d[f'{sh}_min{win_h}h'] = df[col].rolling(w, min_periods=1).min()

    # EWM (exponentially-weighted)
    d[f'{sh}_ewm12'] = df[col].ewm(span=12*SPH, adjust=False).mean()
    d[f'{sh}_ewm48'] = df[col].ewm(span=48*SPH, adjust=False).mean()

    # Slopes (rate of change per hour)
    d[f'{sh}_slope6h']  = df[col].diff(6*SPH)  / 6
    d[f'{sh}_slope24h'] = df[col].diff(24*SPH) / 24
    d[f'{sh}_slope72h'] = df[col].diff(72*SPH) / 72

    parts.append(pd.DataFrame(d, index=df.index))

print(f"  Standard features: {sum(p.shape[1] for p in parts)} columns")


# =============================================================
# 1B. ADVANCED FEATURES (new — entropy, ratios, quantiles)
# =============================================================
print("[1b] Advanced features (entropy, saturation, quantiles)...")

adv = {}

# --- Acceleration (second derivative) ---
adv['mem_accel6h'] = (df['mem_used_pct'].diff(6*SPH)/6).diff(6*SPH)/6
adv['cpu_accel6h'] = (df['cpu_load'].diff(6*SPH)/6).diff(6*SPH)/6

# --- Cross-metric interactions ---
adv['cpu_x_mem']      = df['cpu_load'] * df['mem_used_pct'] / 10000
adv['mem_cpu_ratio']  = df['mem_used_pct'] / df['cpu_load'].replace(0, 1)
adv['ping_x_loss']    = df['ping_latency'] * df['packet_loss'] / 100
adv['ping_delta_1h']  = df['ping_latency'].diff(SPH)

# --- Resource Saturation Ratios ---
# How close to critical threshold (90% mem, 88% cpu)
adv['mem_headroom']   = np.clip(90.0 - df['mem_used_pct'], 0, 90)
adv['cpu_headroom']   = np.clip(88.0 - df['cpu_load'], 0, 88)
adv['saturation_idx'] = (df['cpu_load']/88 + df['mem_used_pct']/90) / 2

# --- Rolling Entropy (captures irregularity in signal) ---
def rolling_entropy(series, window):
    """Shannon entropy over binned rolling window — detects chaos before crash."""
    out = np.full(len(series), np.nan)
    vals = series.values
    for i in range(window, len(vals)):
        chunk = vals[i-window:i]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) < 5:
            continue
        hist, _ = np.histogram(chunk, bins=10, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        out[i] = -np.sum(hist * np.log2(hist + 1e-12))
    return out

for col, sh in [('cpu_load','cpu'), ('mem_used_pct','mem')]:
    adv[f'{sh}_entropy_24h'] = rolling_entropy(df[col], 24 * SPH)

# --- Rolling Quantile Spread (P90-P10 range — detects volatility spikes) ---
for col, sh in [('cpu_load','cpu'), ('mem_used_pct','mem'), ('ping_latency','ping')]:
    w = 24 * SPH
    adv[f'{sh}_iqr_24h'] = (
        df[col].rolling(w, min_periods=1).quantile(0.9) -
        df[col].rolling(w, min_periods=1).quantile(0.1)
    )

# --- Consecutive Anomaly Counter ---
# Counts consecutive steps where metric exceeds P95 (from training baseline)
for col, sh in [('cpu_load','cpu'), ('mem_used_pct','mem')]:
    p95 = df[col].quantile(0.95)
    above = (df[col] > p95).astype(int)
    adv[f'{sh}_streak_above_p95'] = above.groupby(
        (above != above.shift()).cumsum()
    ).cumcount() * above

# --- Time-of-Day / Calendar ---
adv['sin_hour']  = np.sin(2*np.pi*df['hour']/24)
adv['cos_hour']  = np.cos(2*np.pi*df['hour']/24)
adv['sin_dow']   = np.sin(2*np.pi*df['dow']/7)
adv['cos_dow']   = np.cos(2*np.pi*df['dow']/7)
adv['sin_month'] = np.sin(2*np.pi*df['timestamp'].dt.month/12)
adv['cos_month'] = np.cos(2*np.pi*df['timestamp'].dt.month/12)
adv['is_weekend']  = (df['dow'] >= 5).astype(int)
adv['is_biz_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)

# --- WAN features ---
adv['wan_status'] = df['wan_status']
adv['wan_outage_streak'] = (
    df['wan_status'].eq(0)
    .groupby((df['wan_status'] != df['wan_status'].shift()).cumsum())
    .cumcount()
) * df['wan_status'].eq(0).astype(int)

adv_df = pd.DataFrame(adv, index=df.index)
print(f"  Advanced features: {adv_df.shape[1]} columns")

# --- Pre-computed features from dataset ---
orig_cols = ['cpu_mean_24h','ram_mean_24h','cpu_std_24h','ram_std_24h',
             'cpu_slope_6h','ram_slope_6h','wan_instability_6h']
orig = df[orig_cols]

# --- Assemble full feature matrix ---
X_all = pd.concat(
    [df[['cpu_load','mem_used_pct','ping_latency','packet_loss']]]
    + parts + [adv_df, orig],
    axis=1
).fillna(0)

FEATURE_COLS = X_all.columns.tolist()
print(f"\n  TOTAL features: {len(FEATURE_COLS)}")


# =============================================================
# 2. TEMPORAL SPLIT (with crash guarantee)
# =============================================================
crash_idx_arr = df.index[df['is_crash'] == 1].to_numpy()

def find_split(min_ep=5, target=0.75):
    pct = target
    while pct >= 0.50:
        sp = int(len(df) * pct)
        if (crash_idx_arr >= sp).sum() >= min_ep * 15:
            return sp, pct
        pct -= 0.05
    return int(len(df) * 0.70), 0.70

split_idx, split_pct = find_split()
X_tr, X_te = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
print(f"\nSplit: {split_pct:.0%} train / {1-split_pct:.0%} test")
print(f"  Train: {X_tr.shape[0]:,}  Test: {X_te.shape[0]:,}")
for lc in ['incident_in_24h','incident_in_72h','incident_in_7d']:
    print(f"  {lc}: train={df.iloc[:split_idx][lc].sum():,}  test={df.iloc[split_idx:][lc].sum():,}")


# =============================================================
# 2B. CONCEPT DRIFT DETECTION
# =============================================================
print("\n" + "=" * 70)
print("PILLAR 2: Drift Detection (PSI + KS test)")
print("=" * 70)

def compute_psi(expected, actual, bins=10):
    """Population Stability Index — detects covariate drift."""
    e_hist, bin_edges = np.histogram(expected, bins=bins)
    a_hist, _         = np.histogram(actual,   bins=bin_edges)
    e_pct = np.clip(e_hist / len(expected), 1e-6, 1)
    a_pct = np.clip(a_hist / len(actual),   1e-6, 1)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

drift_report = {}
drift_features = ['cpu_load','mem_used_pct','ping_latency','packet_loss',
                  'cpu_slope_6h','ram_slope_6h']

for feat in drift_features:
    train_vals = df.iloc[:split_idx][feat].dropna().values
    test_vals  = df.iloc[split_idx:][feat].dropna().values
    psi = compute_psi(train_vals, test_vals)
    ks_stat, ks_pval = sp_stats.ks_2samp(train_vals, test_vals)
    status = "OK" if psi < 0.10 else ("WARN" if psi < 0.25 else "DRIFT")
    drift_report[feat] = {
        'psi':     round(psi, 4),
        'ks_stat': round(float(ks_stat), 4),
        'ks_pval': round(float(ks_pval), 6),
        'status':  status,
    }
    print(f"  {feat:20s}  PSI={psi:.4f}  KS={ks_stat:.4f}  p={ks_pval:.4f}  [{status}]")


# =============================================================
# 3. EXPANDED BENCHMARK
# =============================================================
print("\n" + "=" * 70)
print("PILLAR 3: Expanded Benchmark")
print("=" * 70)

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier

def best_f2_threshold(y_true, y_prob):
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    denom = 4*prec + rec
    f2 = np.where(denom == 0, 0, (5*prec*rec) / np.maximum(denom, 1e-9))
    return float(thresh[int(np.argmax(f2[:-1]))]) if len(thresh) > 0 else 0.5

def eval_clf(y_true, y_prob, threshold=None):
    if threshold is None:
        threshold = best_f2_threshold(y_true, y_prob)
    yc = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, yc) if len(np.unique(y_true)) > 1 else np.zeros((2,2),dtype=int)
    return {
        'roc_auc':   round(float(roc_auc_score(y_true, y_prob)), 4),
        'pr_auc':    round(float(average_precision_score(y_true, y_prob)), 4),
        'f1':        round(float(f1_score(y_true, yc, zero_division=0)), 4),
        'threshold': round(float(threshold), 4),
        'tp': int(cm[1,1]) if cm.shape==(2,2) else 0,
        'fp': int(cm[0,1]) if cm.shape==(2,2) else 0,
        'fn': int(cm[1,0]) if cm.shape==(2,2) else 0,
        'tn': int(cm[0,0]) if cm.shape==(2,2) else 0,
    }

results = {}

# ── 3A. Isolation Forest Pre-filter ──
if not args.skip_iforest:
    print("\n[3a] Isolation Forest (unsupervised anomaly pre-filter)...")
    t0 = time.time()
    iso = IsolationForest(
        n_estimators=200, contamination=0.08, max_features=0.8,
        random_state=42, n_jobs=-1
    )
    iso.fit(X_tr)
    iso_scores_test = -iso.score_samples(X_te)  # higher = more anomalous
    elapsed_iso = time.time() - t0

    # Evaluate as a standalone anomaly detector
    for horizon, lc in [('24h','incident_in_24h'),('72h','incident_in_72h')]:
        y_te = df.iloc[split_idx:][lc]
        r = eval_clf(y_te, iso_scores_test / iso_scores_test.max())
        results[f'iforest_{horizon}'] = {**r, 'train_time_s': round(elapsed_iso, 1)}
        print(f"  IForest[{horizon}] ROC={r['roc_auc']}  PR-AUC={r['pr_auc']}")

    # Add anomaly score as a feature for supervised models
    iso_scores_train = -iso.score_samples(X_tr)
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    X_tr['iforest_score'] = iso_scores_train
    X_te['iforest_score'] = iso_scores_test
    FEATURE_COLS.append('iforest_score')
    print(f"  Added iforest_score to feature matrix -> {X_tr.shape[1]} features")

# ── 3B. XGBoost (Optuna-tuned) ──
import xgboost as xgb
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

val_split = int(split_idx * 0.85)
X_otr, X_ova = X_tr.iloc[:val_split], X_tr.iloc[val_split:]

def tune_xgb(y_otr, y_ova, spw, n_trials):
    def objective(trial):
        p = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 800, step=100),
            'max_depth':        trial.suggest_int('max_depth', 4, 9),
            'learning_rate':    trial.suggest_float('learning_rate', 0.02, 0.12, log=True),
            'subsample':        trial.suggest_float('subsample', 0.65, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.50, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 25),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.05, 3.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
            'gamma':            trial.suggest_float('gamma', 0.0, 2.0),
            'scale_pos_weight': spw,
            'random_state': 42, 'verbosity': 0, 'tree_method': 'hist',
            'eval_metric': 'aucpr', 'early_stopping_rounds': 30,
        }
        clf = xgb.XGBClassifier(**p)
        clf.fit(X_otr, y_otr, eval_set=[(X_ova, y_ova)], verbose=False)
        return average_precision_score(y_ova, clf.predict_proba(X_ova)[:,1])
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

print(f"\n[3b] XGBoost {'(Optuna ' + str(args.trials) + ' trials)' if HAS_OPTUNA and not args.skip_optuna else '(baseline)'}...")
xgb_models = {}
for horizon, lc in [('24h','incident_in_24h'),('72h','incident_in_72h'),('7d','incident_in_7d')]:
    y_tr = df.iloc[:split_idx][lc]
    y_te = df.iloc[split_idx:][lc]
    spw = max(1, int((y_tr==0).sum()/max(1,(y_tr==1).sum())))

    if HAS_OPTUNA and not args.skip_optuna:
        y_otr_h = df.iloc[:val_split][lc]
        y_ova_h = df.iloc[val_split:split_idx][lc]
        t0 = time.time()
        bp, bv = tune_xgb(y_otr_h, y_ova_h, spw, args.trials)
        print(f"  [{horizon}] Optuna best val PR-AUC: {bv:.4f}  ({time.time()-t0:.0f}s)")
        params = {**bp, 'scale_pos_weight': spw, 'random_state': 42,
                  'verbosity': 0, 'tree_method': 'hist',
                  'eval_metric': 'aucpr', 'early_stopping_rounds': 30}
    else:
        params = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
                      reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=spw,
                      random_state=42, verbosity=0, tree_method='hist',
                      eval_metric='aucpr', early_stopping_rounds=30)

    t0 = time.time()
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    elapsed = time.time() - t0
    yp = clf.predict_proba(X_te)[:,1]
    r = eval_clf(y_te, yp)
    fi = pd.Series(clf.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
    results[f'xgb_{horizon}'] = {
        **r, 'train_time_s': round(elapsed, 1),
        'top10_features': {k: round(float(v),4) for k,v in fi.head(10).items()},
    }
    xgb_models[horizon] = clf
    print(f"  XGB [{horizon}] ROC={r['roc_auc']}  PR-AUC={r['pr_auc']}  F1={r['f1']}  ({elapsed:.1f}s)")

# XGBoost TTF regression
y_tr_reg = df.iloc[:split_idx]['ttf_hours'].clip(0,720)
y_te_reg = df.iloc[split_idx:]['ttf_hours'].clip(0,720)
t0 = time.time()
xgb_reg = xgb.XGBRegressor(
    n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.7, min_child_weight=5, reg_alpha=0.3, reg_lambda=1.5,
    random_state=42, verbosity=0, tree_method='hist', early_stopping_rounds=30
)
xgb_reg.fit(X_tr, y_tr_reg, eval_set=[(X_te, y_te_reg)], verbose=False)
yp_reg = xgb_reg.predict(X_te)
elapsed = time.time() - t0
results['xgb_ttf'] = {
    'mae': round(float(mean_absolute_error(y_te_reg, yp_reg)), 2),
    'rmse': round(float(mean_squared_error(y_te_reg, yp_reg)**0.5), 2),
    'r2': round(float(r2_score(y_te_reg, yp_reg)), 4),
    'time': round(elapsed, 1),
}
print(f"  XGB [TTF] MAE={results['xgb_ttf']['mae']}h  R2={results['xgb_ttf']['r2']}")

# ── 3C. LightGBM ──
print("\n[3c] LightGBM...")
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  LightGBM not installed -> skipped.  pip install lightgbm")

if HAS_LGB:
    for horizon, lc in [('24h','incident_in_24h'),('72h','incident_in_72h'),('7d','incident_in_7d')]:
        y_tr = df.iloc[:split_idx][lc]
        y_te = df.iloc[split_idx:][lc]
        spw = max(1, int((y_tr==0).sum()/max(1,(y_tr==1).sum())))
        t0 = time.time()
        clf = lgb.LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=10, scale_pos_weight=spw,
            reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                callbacks=[lgb.early_stopping(30, verbose=False)])
        elapsed = time.time() - t0
        yp = clf.predict_proba(X_te)[:,1]
        r = eval_clf(y_te, yp)
        results[f'lgb_{horizon}'] = {**r, 'train_time_s': round(elapsed, 1)}
        print(f"  LGB [{horizon}] ROC={r['roc_auc']}  PR-AUC={r['pr_auc']}  F1={r['f1']}  ({elapsed:.1f}s)")

# ── 3D. CatBoost ──
if not args.skip_catboost:
    print("\n[3d] CatBoost...")
    try:
        from catboost import CatBoostClassifier
        HAS_CB = True
    except ImportError:
        HAS_CB = False
        print("  CatBoost not installed -> skipped.  pip install catboost")

    if HAS_CB:
        for horizon, lc in [('24h','incident_in_24h'),('72h','incident_in_72h'),('7d','incident_in_7d')]:
            y_tr = df.iloc[:split_idx][lc]
            y_te = df.iloc[split_idx:][lc]
            spw = max(1, int((y_tr==0).sum()/max(1,(y_tr==1).sum())))
            t0 = time.time()
            clf = CatBoostClassifier(
                iterations=400, depth=6, learning_rate=0.05,
                scale_pos_weight=spw, l2_leaf_reg=3.0,
                random_seed=42, verbose=0, task_type='CPU',
                eval_metric='PRAUC', early_stopping_rounds=30,
            )
            clf.fit(X_tr, y_tr, eval_set=(X_te, y_te), verbose=False)
            elapsed = time.time() - t0
            yp = clf.predict_proba(X_te)[:,1]
            r = eval_clf(y_te, yp)
            results[f'cat_{horizon}'] = {**r, 'train_time_s': round(elapsed, 1)}
            print(f"  CAT [{horizon}] ROC={r['roc_auc']}  PR-AUC={r['pr_auc']}  F1={r['f1']}  ({elapsed:.1f}s)")

# ── 3E. Random Forest ──
print("\n[3e] Random Forest...")
for horizon, lc in [('24h','incident_in_24h'),('72h','incident_in_72h'),('7d','incident_in_7d')]:
    y_tr = df.iloc[:split_idx][lc]
    y_te = df.iloc[split_idx:][lc]
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    yp = rf.predict_proba(X_te)[:,1]
    r = eval_clf(y_te, yp)
    results[f'rf_{horizon}'] = {**r, 'train_time_s': round(elapsed, 1)}
    print(f"  RF  [{horizon}] ROC={r['roc_auc']}  PR-AUC={r['pr_auc']}  F1={r['f1']}  ({elapsed:.1f}s)")


# =============================================================
# 4. SHAP EXPLAINABILITY
# =============================================================
print("\n" + "=" * 70)
print("PILLAR 4: SHAP Explainability")
print("=" * 70)

shap_data = {}
if not args.skip_shap:
    try:
        import shap
        HAS_SHAP = True
    except ImportError:
        HAS_SHAP = False
        print("SHAP not installed -> pip install shap")

    if HAS_SHAP:
        for horizon in ['24h', '72h', '7d']:
            print(f"\n  Computing SHAP for XGBoost {horizon}...")
            model = xgb_models[horizon]
            explainer = shap.TreeExplainer(model)

            # Sample 500 test rows for SHAP (speed)
            sample_idx = np.random.choice(len(X_te), min(500, len(X_te)), replace=False)
            X_sample = X_te.iloc[sample_idx]
            shap_values = explainer.shap_values(X_sample)

            # Global importance (mean |SHAP|)
            mean_abs_shap = pd.Series(
                np.abs(shap_values).mean(axis=0),
                index=X_tr.columns
            ).sort_values(ascending=False)

            shap_data[horizon] = {
                'global_top15': {k: round(float(v), 4) for k, v in mean_abs_shap.head(15).items()},
            }
            top5 = list(mean_abs_shap.head(5).items())
            print(f"    Top-5 SHAP: " + ", ".join(f"{k}={v:.4f}" for k,v in top5))

        # Single-prediction explanation example (last crash in test)
        crash_in_test = df.iloc[split_idx:].index[df.iloc[split_idx:]['is_crash'] == 1]
        if len(crash_in_test) > 0:
            crash_row_idx = crash_in_test[0] - split_idx
            if 0 <= crash_row_idx < len(X_te):
                sv = explainer.shap_values(X_te.iloc[[crash_row_idx]])
                top_reasons = pd.Series(sv[0], index=X_tr.columns).abs().sort_values(ascending=False).head(5)
                shap_data['example_crash_explanation'] = {
                    'row_index': int(crash_row_idx),
                    'top5_reasons': {k: round(float(v), 4) for k, v in top_reasons.items()},
                }
                print(f"\n  Example crash explanation (row {crash_row_idx}):")
                for k, v in top_reasons.items():
                    val = X_te.iloc[crash_row_idx][k]
                    print(f"    {k:30s}  |SHAP|={v:.4f}  value={val:.3f}")
else:
    print("  Skipped (use --skip-shap=false to enable)")


# =============================================================
# 5. HEALTH SCORE (Grafana-ready)
# =============================================================
def compute_health_score(cpu, mem, ping, loss, ttf_pred=None):
    cpu, mem, ping, loss = [np.asarray(x, dtype=float) for x in [cpu, mem, ping, loss]]
    n_cpu  = np.clip((cpu-20)/70, 0, 1)
    n_mem  = np.clip((mem-35)/55, 0, 1)
    n_ping = np.clip((ping-20)/200, 0, 1)
    n_loss = np.clip(loss/15, 0, 1)
    comp = 0.35*n_mem + 0.30*n_cpu + 0.20*n_ping + 0.15*n_loss
    if ttf_pred is not None:
        n_ttf = np.clip(1.0 - np.asarray(ttf_pred, dtype=float)/720, 0, 1)
        comp = 0.6*comp + 0.4*n_ttf
    return np.round((1.0 - np.clip(comp, 0, 1)) * 100, 1)


# =============================================================
# 6. SAVE EVERYTHING
# =============================================================
def to_native(obj):
    if isinstance(obj, dict):  return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    return obj

all_out = {
    'dataset': {
        'rows': int(len(df)), 'features': int(X_tr.shape[1]),
        'crash_pct': round(float(df['is_crash'].mean()*100), 3),
        'split_train_pct': round(split_pct, 3),
    },
    'drift_report': drift_report,
    'benchmark':    results,
    'shap':         shap_data,
}

with open(OUT / 'advanced_results.json', 'w') as f:
    json.dump(to_native(all_out), f, indent=2)

test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
test_df['xgb_ttf_pred'] = yp_reg
test_df['health_pred']  = compute_health_score(
    test_df['cpu_load'], test_df['mem_used_pct'],
    test_df['ping_latency'], test_df['packet_loss'],
    ttf_pred=yp_reg
)
test_df[['timestamp','cpu_load','mem_used_pct','ping_latency','packet_loss',
         'wan_status','is_crash','ttf_hours','xgb_ttf_pred',
         'health_score','health_pred',
         'incident_in_24h','incident_in_72h','incident_in_7d']].to_csv(
    OUT / 'test_predictions.csv', index=False
)

print(f"\nSaved -> {OUT / 'advanced_results.json'}")
print(f"Saved -> {OUT / 'test_predictions.csv'}")

# =============================================================
# FINAL COMPARISON TABLE
# =============================================================
print("\n" + "=" * 80)
print("FULL BENCHMARK COMPARISON")
print("=" * 80)
print(f"{'Model':<25} {'Horizon':<8} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1':<8} {'Time'}")
print("-" * 80)

model_order = ['xgb','lgb','cat','rf','iforest']
model_labels = {'xgb':'XGBoost (Optuna)','lgb':'LightGBM','cat':'CatBoost',
                'rf':'RandomForest','iforest':'IsolationForest'}

for prefix in model_order:
    for h in ['24h','72h','7d']:
        key = f'{prefix}_{h}'
        if key in results:
            r = results[key]
            label = model_labels.get(prefix, prefix)
            print(f"{label:<25} {h:<8} {r['roc_auc']:<10} {r['pr_auc']:<10} "
                  f"{r['f1']:<8} {r.get('train_time_s','?')}s")

if 'xgb_ttf' in results:
    r = results['xgb_ttf']
    print(f"\n{'XGBoost TTF reg':<25} {'-':<8} MAE={r['mae']}h  "
          f"RMSE={r['rmse']}h  R2={r['r2']}")

print("\nDone.")
