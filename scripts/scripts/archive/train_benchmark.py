"""
HGW Predictive Maintenance — Full ML Benchmark with Optuna Tuning
==================================================================
Models:    XGBoost (Optuna-tuned) + RandomForest + LSTM (14-day lookback)
Horizons:  24h, 72h, 7d binary classification + TTF regression
Output:    data/benchmark_results.json + data/test_predictions.csv

Requirements (install in your venv):
    pip install pandas numpy scikit-learn xgboost optuna
    pip install tensorflow            # optional, for LSTM
    pip install shap                  # optional, for explainability

Usage:
    python train_benchmark.py                     # full pipeline, default 30 Optuna trials
    python train_benchmark.py --trials 50         # more thorough tuning
    python train_benchmark.py --skip-lstm         # skip LSTM if no TF
    python train_benchmark.py --skip-optuna       # use baseline hyperparameters
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data',         default='data/hgw_5yr_bigdata.csv')
parser.add_argument('--out-dir',      default='data')
parser.add_argument('--trials',       type=int, default=30,
                    help='Optuna trials per XGBoost model (default: 30)')
parser.add_argument('--skip-optuna',  action='store_true')
parser.add_argument('--skip-lstm',    action='store_true')
parser.add_argument('--skip-rf',      action='store_true')
parser.add_argument('--lstm-epochs',  type=int, default=20)
args = parser.parse_args()

OUT = Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD & FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv(args.data, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
SPH = 2  # 30-min step → 2 steps per hour
print(f"Rows: {len(df):,}  Cols: {df.shape[1]}  "
      f"Range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

# --- Feature engineering: lags, rolling stats, slopes, interactions ---
print("Building feature matrix...")
parts = []
for col, sh in [('cpu_load','cpu'), ('mem_used_pct','mem'),
                 ('ping_latency','ping'), ('packet_loss','loss')]:
    d = {}
    for lag_h in [1, 3, 6, 12, 24, 72, 168]:
        d[f'{sh}_lag{lag_h}h'] = df[col].shift(lag_h * SPH)
    for win_h in [6, 24, 72, 168]:
        win_s = win_h * SPH
        d[f'{sh}_ma{win_h}h']  = df[col].rolling(win_s, min_periods=1).mean()
        d[f'{sh}_std{win_h}h'] = df[col].rolling(win_s, min_periods=1).std()
        d[f'{sh}_max{win_h}h'] = df[col].rolling(win_s, min_periods=1).max()
    d[f'{sh}_ewm12']    = df[col].ewm(span=12*SPH, adjust=False).mean()
    d[f'{sh}_slope6h']  = df[col].diff(6*SPH)  / 6
    d[f'{sh}_slope24h'] = df[col].diff(24*SPH) / 24
    d[f'{sh}_slope72h'] = df[col].diff(72*SPH) / 72
    parts.append(pd.DataFrame(d, index=df.index))

extra = pd.DataFrame({
    'mem_accel6h':   (df['mem_used_pct'].diff(6*SPH)/6).diff(6*SPH)/6,
    'cpu_accel6h':   (df['cpu_load'].diff(6*SPH)/6).diff(6*SPH)/6,
    'cpu_x_mem':     df['cpu_load'] * df['mem_used_pct'] / 10000,
    'mem_cpu_ratio': df['mem_used_pct'] / df['cpu_load'].replace(0, 1),
    'ping_delta_1h': df['ping_latency'].diff(SPH),
    'sin_hour':      np.sin(2*np.pi*df['hour']/24),
    'cos_hour':      np.cos(2*np.pi*df['hour']/24),
    'sin_dow':       np.sin(2*np.pi*df['dow']/7),
    'cos_dow':       np.cos(2*np.pi*df['dow']/7),
    'sin_month':     np.sin(2*np.pi*df['timestamp'].dt.month/12),
    'cos_month':     np.cos(2*np.pi*df['timestamp'].dt.month/12),
    'is_weekend':    (df['dow'] >= 5).astype(int),
    'is_biz_hour':   ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int),
    'wan_status':    df['wan_status'],
}, index=df.index)

orig = df[['cpu_mean_24h','ram_mean_24h','cpu_std_24h','ram_std_24h',
           'cpu_slope_6h','ram_slope_6h','wan_instability_6h']]

X_all = pd.concat(
    [df[['cpu_load','mem_used_pct','ping_latency','packet_loss']]] + parts + [extra, orig],
    axis=1
).fillna(0)
FEATURE_COLS = X_all.columns.tolist()
print(f"Feature matrix: {X_all.shape[0]:,} x {X_all.shape[1]} features")

# ─────────────────────────────────────────────
# 2. ROBUST TEMPORAL SPLIT
# ─────────────────────────────────────────────
# Find a split point that guarantees crashes in BOTH train and test sets.
# Strategy: walk back from 80% until we have at least 5 crash episodes in test.
crash_idx_array = df.index[df['is_crash'] == 1].to_numpy()

def find_split_with_crashes(min_test_episodes=5, target_pct=0.75):
    """Walk back from target_pct until test contains enough crash episodes."""
    candidate_pct = target_pct
    while candidate_pct >= 0.50:
        sp = int(len(df) * candidate_pct)
        test_crash_count = (crash_idx_array >= sp).sum()
        if test_crash_count >= min_test_episodes * 20:  # ~20 rows per episode
            return sp, candidate_pct
        candidate_pct -= 0.05
    return int(len(df) * 0.70), 0.70

split_idx, split_pct = find_split_with_crashes()
print(f"\nTemporal split: {split_pct:.0%} train / {1-split_pct:.0%} test")
print(f"Train: row 0 -> {split_idx-1:,}  Test: row {split_idx:,} -> {len(df)-1:,}")
print(f"Train crashes: {df.iloc[:split_idx]['is_crash'].sum():,}  "
      f"Test crashes: {df.iloc[split_idx:]['is_crash'].sum():,}")
for lc in ['incident_in_24h', 'incident_in_72h', 'incident_in_7d']:
    tr_pos = df.iloc[:split_idx][lc].sum()
    te_pos = df.iloc[split_idx:][lc].sum()
    print(f"  {lc}: train={tr_pos:,}  test={te_pos:,}")

X_tr, X_te = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
print(f"\nX_tr: {X_tr.shape}  X_te: {X_te.shape}\n")

# ─────────────────────────────────────────────
# 3. METRICS HELPERS
# ─────────────────────────────────────────────
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, precision_recall_curve
)

def best_f2_threshold(y_true, y_prob):
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    denom = 4 * prec + rec
    f2 = np.where(denom == 0, 0, (5 * prec * rec) / np.maximum(denom, 1e-9))
    if len(thresh) == 0:
        return 0.5
    return float(thresh[int(np.argmax(f2[:-1]))])

# ─────────────────────────────────────────────
# 4. XGBOOST + OPTUNA
# ─────────────────────────────────────────────
print("=" * 60)
print("BENCHMARK 1: XGBoost  (Optuna-tuned)" if not args.skip_optuna
      else "BENCHMARK 1: XGBoost  (baseline params)")
print("=" * 60)

import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not installed -> using baseline hyperparameters")
    print("  install with: pip install optuna\n")

results_xgb = {}

def tune_xgb(X_tr, y_tr, X_va, y_va, scale_pos_weight, n_trials):
    """Optuna search optimizing PR-AUC on validation fold."""
    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 800, step=100),
            'max_depth':        trial.suggest_int('max_depth', 4, 9),
            'learning_rate':    trial.suggest_float('learning_rate', 0.02, 0.12, log=True),
            'subsample':        trial.suggest_float('subsample', 0.65, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.55, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 25),
            'reg_alpha':        trial.suggest_float('reg_alpha',  0.1, 3.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
            'gamma':            trial.suggest_float('gamma', 0.0, 2.0),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42, 'verbosity': 0, 'tree_method': 'hist',
            'eval_metric': 'aucpr', 'early_stopping_rounds': 30,
        }
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        yp = clf.predict_proba(X_va)[:, 1]
        return average_precision_score(y_va, yp)

    study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

# Use last 15% of train as Optuna validation fold (still no test leakage)
val_split = int(split_idx * 0.85)
X_otr = X_tr.iloc[:val_split]
X_ova = X_tr.iloc[val_split:]

for horizon, label_col in [('24h','incident_in_24h'),
                            ('72h','incident_in_72h'),
                            ('7d', 'incident_in_7d')]:
    y_tr = df.iloc[:split_idx][label_col]
    y_te = df.iloc[split_idx:][label_col]
    spw = max(1, int((y_tr == 0).sum() / max(1, (y_tr == 1).sum())))

    if HAS_OPTUNA and not args.skip_optuna:
        print(f"\n[{horizon}] Optuna search ({args.trials} trials)...")
        y_otr = df.iloc[:val_split][label_col]
        y_ova = df.iloc[val_split:split_idx][label_col]
        t0 = time.time()
        best_params, best_score = tune_xgb(X_otr, y_otr, X_ova, y_ova, spw, args.trials)
        tune_time = time.time() - t0
        print(f"  best validation PR-AUC: {best_score:.4f}  ({tune_time:.0f}s)")
        params = {**best_params, 'scale_pos_weight': spw,
                   'random_state': 42, 'verbosity': 0, 'tree_method': 'hist',
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
    yp = clf.predict_proba(X_te)[:, 1]
    bt = best_f2_threshold(y_te, yp)
    yc = (yp >= bt).astype(int)
    cm = confusion_matrix(y_te, yc) if len(np.unique(y_te)) > 1 else np.zeros((2, 2))

    fi = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    results_xgb[horizon] = {
        'roc_auc':       round(float(roc_auc_score(y_te, yp)), 4),
        'pr_auc':        round(float(average_precision_score(y_te, yp)), 4),
        'f1':            round(float(f1_score(y_te, yc, zero_division=0)), 4),
        'best_threshold': round(bt, 4),
        'tp': int(cm[1,1]) if cm.shape == (2,2) else 0,
        'fp': int(cm[0,1]) if cm.shape == (2,2) else 0,
        'fn': int(cm[1,0]) if cm.shape == (2,2) else 0,
        'tn': int(cm[0,0]) if cm.shape == (2,2) else 0,
        'train_time_s':  round(elapsed, 1),
        'top10_features': {k: round(float(v), 4) for k, v in fi.head(10).items()},
        'best_params':   {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in params.items()
                            if k not in ('random_state', 'verbosity', 'tree_method',
                                          'eval_metric', 'early_stopping_rounds')},
    }
    r = results_xgb[horizon]
    print(f"  [{horizon}] ROC={r['roc_auc']}  PR-AUC={r['pr_auc']}  "
          f"F1={r['f1']}  TP={r['tp']} FN={r['fn']}  ({elapsed:.1f}s)")

# TTF regression
y_tr_reg = df.iloc[:split_idx]['ttf_hours'].clip(0, 720)
y_te_reg = df.iloc[split_idx:]['ttf_hours'].clip(0, 720)

t0 = time.time()
xgb_reg = xgb.XGBRegressor(
    n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.7, min_child_weight=5, reg_alpha=0.3, reg_lambda=1.5,
    random_state=42, verbosity=0, tree_method='hist', early_stopping_rounds=30
)
xgb_reg.fit(X_tr, y_tr_reg, eval_set=[(X_te, y_te_reg)], verbose=False)
yp_reg = xgb_reg.predict(X_te)
elapsed = time.time() - t0
results_xgb['ttf'] = {
    'mae_hours':  round(float(mean_absolute_error(y_te_reg, yp_reg)), 2),
    'rmse_hours': round(float(mean_squared_error(y_te_reg, yp_reg) ** 0.5), 2),
    'r2':         round(float(r2_score(y_te_reg, yp_reg)), 4),
    'train_time_s': round(elapsed, 1),
}
print(f"  [TTF] MAE={results_xgb['ttf']['mae_hours']}h  "
      f"R²={results_xgb['ttf']['r2']}  ({elapsed:.1f}s)")

# ─────────────────────────────────────────────
# 5. RANDOM FOREST
# ─────────────────────────────────────────────
results_rf = {}
if not args.skip_rf:
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Random Forest")
    print("=" * 60)
    from sklearn.ensemble import RandomForestClassifier

    for horizon, label_col in [('24h','incident_in_24h'),
                                ('72h','incident_in_72h'),
                                ('7d', 'incident_in_7d')]:
        y_tr = df.iloc[:split_idx][label_col]
        y_te = df.iloc[split_idx:][label_col]
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        yp = rf.predict_proba(X_te)[:, 1]
        bt = best_f2_threshold(y_te, yp)
        yc = (yp >= bt).astype(int)
        results_rf[horizon] = {
            'roc_auc':       round(float(roc_auc_score(y_te, yp)), 4),
            'pr_auc':        round(float(average_precision_score(y_te, yp)), 4),
            'f1':            round(float(f1_score(y_te, yc, zero_division=0)), 4),
            'best_threshold': round(bt, 4),
            'train_time_s':  round(elapsed, 1),
        }
        r = results_rf[horizon]
        print(f"  [{horizon}] ROC={r['roc_auc']}  PR-AUC={r['pr_auc']}  "
              f"F1={r['f1']}  ({elapsed:.1f}s)")

# ─────────────────────────────────────────────
# 6. LSTM (14-day lookback)
# ─────────────────────────────────────────────
results_lstm = {}
if not args.skip_lstm:
    print("\n" + "=" * 60)
    print("BENCHMARK 3: LSTM (14-day lookback)")
    print("=" * 60)
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import StandardScaler
        HAS_TF = True
        print(f"TensorFlow {tf.__version__} loaded")
    except ImportError:
        HAS_TF = False
        print("TensorFlow not available -> LSTM skipped.")
        print("  install with: pip install tensorflow")

    if HAS_TF:
        LSTM_COLS = ['cpu_load','mem_used_pct','ping_latency','packet_loss',
                      'cpu_slope_6h','ram_slope_6h','wan_status','health_score']
        LOOKBACK  = 14 * 24 * SPH   # 672 steps
        SUB       = 6               # subsample to 112 timesteps (3h grain)

        scaler = StandardScaler()
        X_lstm_full = df[LSTM_COLS].fillna(0).values
        X_lstm_full[:split_idx] = scaler.fit_transform(X_lstm_full[:split_idx])
        X_lstm_full[split_idx:] = scaler.transform(X_lstm_full[split_idx:])

        def build_seqs(data, labels, start, end, lookback, sub):
            X, y = [], []
            for i in range(start + lookback, end):
                X.append(data[i-lookback:i:sub])
                y.append(labels[i])
            return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

        for horizon, label_col in [('24h','incident_in_24h'),
                                     ('72h','incident_in_72h')]:
            print(f"\n  Training LSTM [{horizon}]...")
            y_full = df[label_col].values

            X_tr_seq, y_tr_seq = build_seqs(X_lstm_full, y_full, 0, split_idx, LOOKBACK, SUB)
            X_te_seq, y_te_seq = build_seqs(X_lstm_full, y_full, split_idx - LOOKBACK,
                                              len(df), LOOKBACK, SUB)

            # Balanced 3:1 undersampling
            pos = np.where(y_tr_seq == 1)[0]
            neg = np.where(y_tr_seq == 0)[0]
            if len(pos) == 0:
                print(f"    [{horizon}] skipped (no positive labels in train)")
                continue
            neg_keep = np.random.choice(neg, min(len(pos)*3, len(neg)), replace=False)
            keep = np.sort(np.concatenate([pos, neg_keep]))
            X_tr_bal, y_tr_bal = X_tr_seq[keep], y_tr_seq[keep]
            print(f"    Train seqs: {X_tr_bal.shape}  Test seqs: {X_te_seq.shape}")

            t0 = time.time()
            model = Sequential([
                LSTM(64, input_shape=(X_tr_bal.shape[1], X_tr_bal.shape[2]),
                     return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
                LSTM(32, dropout=0.2),
                Dense(16, activation='relu'),
                Dropout(0.3),
                Dense(1,  activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
            es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
            history = model.fit(
                X_tr_bal, y_tr_bal, validation_split=0.15,
                epochs=args.lstm_epochs, batch_size=64,
                callbacks=[es], verbose=0
            )
            elapsed = time.time() - t0

            yp = model.predict(X_te_seq, verbose=0).flatten()
            bt = best_f2_threshold(y_te_seq, yp)
            yc = (yp >= bt).astype(int)
            results_lstm[horizon] = {
                'roc_auc':       round(float(roc_auc_score(y_te_seq, yp)), 4),
                'pr_auc':        round(float(average_precision_score(y_te_seq, yp)), 4),
                'f1':            round(float(f1_score(y_te_seq, yc, zero_division=0)), 4),
                'best_threshold': round(bt, 4),
                'epochs':        len(history.history['loss']),
                'train_time_s':  round(elapsed, 1),
            }
            r = results_lstm[horizon]
            print(f"    ROC={r['roc_auc']}  PR-AUC={r['pr_auc']}  "
                  f"F1={r['f1']}  ({elapsed:.1f}s, {r['epochs']} epochs)")

# ─────────────────────────────────────────────
# 7. HEALTH SCORE FUNCTION (Grafana-ready)
# ─────────────────────────────────────────────
def compute_health_score(cpu, mem, ping, loss, ttf_pred=None):
    """
    Grafana-ready health score: 100% (healthy) -> 0% (crash imminent).

    Parameters can be scalars or numpy arrays.
    If ttf_pred is provided (in hours), it is blended into the score.

    Returns: float or array, range [0, 100]
    """
    cpu  = np.asarray(cpu, dtype=float)
    mem  = np.asarray(mem, dtype=float)
    ping = np.asarray(ping, dtype=float)
    loss = np.asarray(loss, dtype=float)
    n_cpu  = np.clip((cpu  - 20) / 70,  0, 1)
    n_mem  = np.clip((mem  - 35) / 55,  0, 1)
    n_ping = np.clip((ping - 20) / 200, 0, 1)
    n_loss = np.clip(loss / 15, 0, 1)
    composite = 0.35*n_mem + 0.30*n_cpu + 0.20*n_ping + 0.15*n_loss
    if ttf_pred is not None:
        n_ttf = np.clip(1.0 - np.asarray(ttf_pred, dtype=float)/720, 0, 1)
        composite = 0.6*composite + 0.4*n_ttf
    return np.round((1.0 - np.clip(composite, 0, 1)) * 100, 1)

# ─────────────────────────────────────────────
# 8. SAVE RESULTS
# ─────────────────────────────────────────────
def to_native(obj):
    """Convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):  return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    return obj

all_results = {
    'dataset': {
        'rows':            int(len(df)),
        'features':        int(X_all.shape[1]),
        'step_minutes':    30,
        'years':           5,
        'crash_rows':      int(df['is_crash'].sum()),
        'crash_pct':       round(float(df['is_crash'].mean()*100), 3),
        'split_pct_train': round(split_pct, 3),
        'date_range':      [str(df['timestamp'].min()), str(df['timestamp'].max())],
    },
    'optuna_used':   bool(HAS_OPTUNA and not args.skip_optuna),
    'optuna_trials': args.trials,
    'xgboost':       results_xgb,
    'random_forest': results_rf if not args.skip_rf else 'skipped',
    'lstm':          results_lstm if results_lstm else 'unavailable',
}

results_path = OUT / 'benchmark_results.json'
with open(results_path, 'w') as f:
    json.dump(to_native(all_results), f, indent=2)
print(f"\nResults saved -> {results_path}")

# Test predictions for dashboard / Grafana
test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
test_df['xgb_ttf_pred']     = yp_reg
test_df['health_pred']      = compute_health_score(
    test_df['cpu_load'], test_df['mem_used_pct'],
    test_df['ping_latency'], test_df['packet_loss'],
    ttf_pred=test_df['xgb_ttf_pred']
)
preds_path = OUT / 'test_predictions.csv'
test_df[['timestamp','cpu_load','mem_used_pct','ping_latency','packet_loss',
         'wan_status','is_crash','ttf_hours','xgb_ttf_pred',
         'health_score','health_pred',
         'incident_in_24h','incident_in_72h','incident_in_7d']].to_csv(preds_path, index=False)
print(f"Test predictions -> {preds_path}")

# ─────────────────────────────────────────────
# 9. FINAL COMPARISON TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("BENCHMARK COMPARISON")
print("=" * 70)
print(f"{'Model':<22} {'Horizon':<8} {'ROC-AUC':<10} {'PR-AUC':<10} {'F1':<8} {'Time'}")
print("-" * 70)
for h in ['24h','72h','7d']:
    if h in results_xgb:
        r = results_xgb[h]
        print(f"{'XGBoost (Optuna)' if HAS_OPTUNA and not args.skip_optuna else 'XGBoost':<22} "
              f"{h:<8} {r['roc_auc']:<10} {r['pr_auc']:<10} {r['f1']:<8} {r['train_time_s']}s")
if not args.skip_rf:
    for h in ['24h','72h','7d']:
        if h in results_rf:
            r = results_rf[h]
            print(f"{'RandomForest':<22} {h:<8} {r['roc_auc']:<10} {r['pr_auc']:<10} "
                  f"{r['f1']:<8} {r['train_time_s']}s")
for h in results_lstm:
    r = results_lstm[h]
    print(f"{'LSTM (14d lookback)':<22} {h:<8} {r['roc_auc']:<10} {r['pr_auc']:<10} "
          f"{r['f1']:<8} {r['train_time_s']}s")

if 'ttf' in results_xgb:
    r = results_xgb['ttf']
    print(f"\n{'XGBoost TTF reg':<22} {'-':<8} MAE={r['mae_hours']}h  "
          f"RMSE={r['rmse_hours']}h  R2={r['r2']}  ({r['train_time_s']}s)")

print("\nDone.")
