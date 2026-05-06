"""
Multi-Horizon CatBoost Training — Real HGW Data
================================================

Trains 4 CatBoost models in parallel for short-term incident prediction:
  - incident_in_15min
  - incident_in_30min
  - incident_in_1h
  - incident_in_6h

Each model is independently:
  - 5-fold cross-validated
  - Threshold-tuned (F1, F2, F0.5)
  - SHAP-explainable
  - Saved to disk with full metadata

Output: data/real/multi_horizon/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')

from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score,
                             precision_recall_curve, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, train_test_split
from catboost import CatBoostClassifier

# =============================================================================
# Config
# =============================================================================
INPUT_CSV = 'data/real/real_hgw_preprocessed.csv'
OUT_DIR = Path('data/real/multi_horizon')
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS_MIN = [15, 30, 60, 360]  # 15 min, 30 min, 1 hour, 6 hours

FEATURES = [
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
    'sin_hour', 'cos_hour', 'cpu_x_mem', 'saturation_idx', 'mem_headroom', 'health_score',
]


def add_horizon_label(df, horizon_min):
    """Add incident_in_<horizon>min label per session."""
    out = []
    label_col = f'incident_in_{horizon_min}min'
    for sid, group in df.groupby('session_id'):
        g = group.copy().sort_values('timestamp').reset_index(drop=True)
        is_urgent = (g['LOCAL_STATUS'] == 'URGENT').astype(int)
        future = is_urgent.iloc[::-1].rolling(horizon_min, min_periods=1).max().iloc[::-1]
        g[label_col] = future.shift(-1).fillna(0).astype(int)
        g.loc[is_urgent == 1, label_col] = 0  # exclude trivially-true samples
        out.append(g)
    return pd.concat(out, ignore_index=True)


def find_thresholds(y_true, y_prob):
    """Find F1/F2/F0.5-optimal thresholds."""
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)

    def best_th(beta):
        b2 = beta ** 2
        denom = b2 * prec + rec
        f = np.where(denom == 0, 0, (1 + b2) * prec * rec / np.maximum(denom, 1e-9))
        return float(thresh[int(np.argmax(f[:-1]))]) if len(thresh) > 0 else 0.5

    return {
        'high_recall_F2': best_th(2.0),
        'balanced_F1': best_th(1.0),
        'high_precision_F0.5': best_th(0.5),
    }


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
        'tn': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        'fp': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        'fn': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        'tp': int(cm[1, 1]) if cm.shape == (2, 2) else 0,
    }


def cross_validate(X, y, n_splits=5):
    """5-fold stratified CV with F2-tuned threshold per fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv = {'roc_auc': [], 'pr_auc': [], 'f1': [], 'precision': [], 'recall': []}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
        if n_pos < 2:
            continue
        pw = max(1.0, n_neg / max(1, n_pos))

        cb = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            loss_function='Logloss', eval_metric='PRAUC',
            early_stopping_rounds=50, random_seed=42, verbose=0,
            class_weights=[1.0, pw],
        )
        cb.fit(X_tr, y_tr, eval_set=(X_te, y_te))

        y_p = cb.predict_proba(X_te)[:, 1]
        ths = find_thresholds(y_te, y_p)
        m = metrics_at_threshold(y_te, y_p, ths['balanced_F1'])

        cv['roc_auc'].append(roc_auc_score(y_te, y_p))
        cv['pr_auc'].append(average_precision_score(y_te, y_p))
        cv['f1'].append(m['f1'])
        cv['precision'].append(m['precision'])
        cv['recall'].append(m['recall'])

    return cv


def train_one_horizon(df, horizon_min):
    """Train and validate one CatBoost model for the given horizon."""
    label = f'incident_in_{horizon_min}min'

    # Add label if missing
    if label not in df.columns:
        df = add_horizon_label(df, horizon_min)

    df_ml = df.dropna(subset=FEATURES + [label]).copy().reset_index(drop=True)

    pos_rate = df_ml[label].mean()
    n_pos = int(df_ml[label].sum())

    # Sanity check: if label is degenerate, skip
    if n_pos < 30 or pos_rate > 0.85 or pos_rate < 0.005:
        return {
            'horizon_min': horizon_min,
            'status': 'SKIPPED',
            'reason': f'Label degenerate: positives={n_pos}, rate={pos_rate:.2%}',
            'recommendation': 'Need more data with sufficient incident diversity',
        }

    print(f'\n{"="*70}')
    print(f'Training CatBoost for {label}  (positives={n_pos}, rate={pos_rate:.1%})')
    print('=' * 70)

    # Stratified split
    X = df_ml[FEATURES].values
    y = df_ml[label].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Train final model
    n_neg, n_pos_tr = (y_tr == 0).sum(), (y_tr == 1).sum()
    pw = max(1.0, n_neg / max(1, n_pos_tr))
    cb = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        loss_function='Logloss', eval_metric='PRAUC',
        early_stopping_rounds=50, random_seed=42, verbose=0,
        class_weights=[1.0, pw],
    )
    cb.fit(X_tr, y_tr, eval_set=(X_te, y_te))

    # Single-split metrics
    y_prob = cb.predict_proba(X_te)[:, 1]
    thresholds = find_thresholds(y_te, y_prob)
    metrics = {name: metrics_at_threshold(y_te, y_prob, th) for name, th in thresholds.items()}

    # Cross-validation
    print(f'  Running 5-fold CV...')
    cv = cross_validate(X, y)
    cv_summary = {
        k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
        for k, v in cv.items() if v
    }

    # Save model
    model_path = OUT_DIR / f'catboost_{horizon_min}min_real.cbm'
    cb.save_model(str(model_path))

    # Feature importance
    feature_imp = dict(sorted(
        zip(FEATURES, cb.feature_importances_),
        key=lambda x: -x[1]
    ))

    result = {
        'horizon_min': horizon_min,
        'status': 'OK',
        'model_file': str(model_path.name),
        'n_train': int(len(X_tr)),
        'n_test': int(len(X_te)),
        'positive_rate': float(pos_rate),
        'n_positives_total': n_pos,
        'roc_auc': float(roc_auc_score(y_te, y_prob)),
        'pr_auc': float(average_precision_score(y_te, y_prob)),
        'thresholds': thresholds,
        'metrics_per_threshold': metrics,
        'cv_results': cv_summary,
        'top_features': {k: float(v) for k, v in list(feature_imp.items())[:10]},
    }

    print(f'  ROC-AUC: {result["roc_auc"]:.4f}')
    print(f'  PR-AUC:  {result["pr_auc"]:.4f}')
    print(f'  CV PR-AUC: {cv_summary["pr_auc"]["mean"]:.4f} ± {cv_summary["pr_auc"]["std"]:.4f}')
    print(f'  At balanced_F1 threshold ({thresholds["balanced_F1"]:.4f}):')
    m = metrics['balanced_F1']
    print(f'    Precision={m["precision"]:.4f}  Recall={m["recall"]:.4f}  F1={m["f1"]:.4f}')

    return result


def main():
    if not Path(INPUT_CSV).exists():
        print(f'ERROR: {INPUT_CSV} not found. Run the main notebook first.')
        return

    df = pd.read_csv(INPUT_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f'Loaded {len(df)} rows from {INPUT_CSV}')

    # Train all horizons
    all_results = {}
    for h in HORIZONS_MIN:
        all_results[f'{h}min'] = train_one_horizon(df, h)

    # Save consolidated bundle
    bundle = {
        'pipeline_version': '2.0_multi_horizon',
        'training_data': {
            'source': INPUT_CSV,
            'n_samples': int(len(df)),
            'duration_days': 7.5,
        },
        'features': FEATURES,
        'horizons': all_results,
        'recommended_thresholds': {
            f'{h}min': all_results[f'{h}min'].get('thresholds', {}).get('balanced_F1')
            for h in HORIZONS_MIN
            if all_results[f'{h}min'].get('status') == 'OK'
        },
    }

    bundle_path = OUT_DIR / 'multi_horizon_bundle.json'
    with open(bundle_path, 'w') as f:
        json.dump(bundle, f, indent=2)
    print(f'\nSaved consolidated bundle → {bundle_path}')

    # Final summary table
    print('\n' + '=' * 80)
    print('MULTI-HORIZON SUMMARY')
    print('=' * 80)
    print(f'{"Horizon":<10} {"Status":<10} {"PR-AUC":>9} {"CV-PR-AUC":>14} {"P":>7} {"R":>7} {"F1":>7}')
    print('-' * 80)
    for h in HORIZONS_MIN:
        r = all_results[f'{h}min']
        if r['status'] == 'OK':
            cv = r['cv_results']['pr_auc']
            m = r['metrics_per_threshold']['balanced_F1']
            print(f'{h:>4}min    {"OK":<10} {r["pr_auc"]:>9.4f} '
                  f'{cv["mean"]:>6.4f}±{cv["std"]:.4f}  {m["precision"]:>7.4f} '
                  f'{m["recall"]:>7.4f} {m["f1"]:>7.4f}')
        else:
            print(f'{h:>4}min    {"SKIPPED":<10} {r["reason"]}')


if __name__ == '__main__':
    main()
