import pandas as pd, numpy as np, json, time, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
    mean_absolute_error, r2_score, confusion_matrix, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

print("Loading...")
df = pd.read_csv('/home/claude/data/hgw_5yr_bigdata.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
SPH = 2
print(f'Rows: {len(df):,}')

# Use EXISTING features + a few key engineered ones (avoid heavy rolling)
print("Feature engineering...")
df['cpu_x_mem'] = df['cpu_load'] * df['mem_used_pct'] / 10000
df['sin_hour'] = np.sin(2*np.pi*df['hour']/24)
df['cos_hour'] = np.cos(2*np.pi*df['hour']/24)
df['sin_month'] = np.sin(2*np.pi*df['timestamp'].dt.month/12)

# Key lags only
for col, sh in [('cpu_load','cpu'),('mem_used_pct','mem')]:
    df[f'{sh}_lag1'] = df[col].shift(SPH)
    df[f'{sh}_lag6'] = df[col].shift(6*SPH)
    df[f'{sh}_lag24'] = df[col].shift(24*SPH)
    df[f'{sh}_lag72'] = df[col].shift(72*SPH)

FCOLS = ['cpu_load','mem_used_pct','ping_latency','packet_loss','wan_status',
         'cpu_mean_24h','ram_mean_24h','cpu_std_24h','ram_std_24h',
         'cpu_slope_6h','ram_slope_6h','wan_instability_6h',
         'cpu_x_mem','sin_hour','cos_hour','sin_month',
         'cpu_lag1','cpu_lag6','cpu_lag24','cpu_lag72',
         'mem_lag1','mem_lag6','mem_lag24','mem_lag72']

Xf = df[FCOLS].fillna(0)
print(f'Features: {len(FCOLS)}')

sp = int(len(df)*0.8)
Xtr, Xte = Xf.iloc[:sp], Xf.iloc[sp:]
print(f'Train: {Xtr.shape[0]:,}  Test: {Xte.shape[0]:,}')

res = {}

# XGBoost classifiers
for h, lc in [('24h','incident_in_24h'),('72h','incident_in_72h'),('7d','incident_in_7d')]:
    ytr = df.loc[:sp-1, lc]
    yte = df.loc[sp:, lc]
    spw = max(1, int((ytr==0).sum()/max(1,(ytr==1).sum())))
    t0 = time.time()
    c = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10, scale_pos_weight=spw,
        reg_alpha=0.5, reg_lambda=2.0, random_state=42, verbosity=0,
        tree_method='hist', eval_metric='aucpr', early_stopping_rounds=20)
    c.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    el = time.time() - t0
    yp = c.predict_proba(Xte)[:,1]
    pr, rc, th = precision_recall_curve(yte, yp)
    f2 = np.where((4*pr+rc)==0, 0, (5*pr*rc)/(4*pr+rc))
    bt = th[np.argmax(f2[:-1])] if len(th) > 0 else 0.5
    yc = (yp >= bt).astype(int)
    cm = confusion_matrix(yte, yc)
    fi = pd.Series(c.feature_importances_, index=FCOLS).sort_values(ascending=False)
    res['xgb_' + h] = {
        'roc': round(float(roc_auc_score(yte, yp)), 4),
        'pr': round(float(average_precision_score(yte, yp)), 4),
        'f1': round(float(f1_score(yte, yc, zero_division=0)), 4),
        'th': round(float(bt), 4),
        'time': round(el, 1),
        'tp': int(cm[1,1]) if cm.shape==(2,2) else 0,
        'fp': int(cm[0,1]) if cm.shape==(2,2) else 0,
        'fn': int(cm[1,0]) if cm.shape==(2,2) else 0,
        'fi': {k: round(float(v), 4) for k, v in fi.head(5).items()}
    }
    r = res['xgb_' + h]
    print(f'XGB[{h}] ROC={r["roc"]} PR={r["pr"]} F1={r["f1"]} ({el:.1f}s)')

# TTF
print("TTF regression...")
ytr_r = df.loc[:sp-1, 'ttf_hours'].clip(0, 720)
yte_r = df.loc[sp:, 'ttf_hours'].clip(0, 720)
t0 = time.time()
xr = xgb.XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.08, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, random_state=42,
    verbosity=0, tree_method='hist', early_stopping_rounds=20)
xr.fit(Xtr, ytr_r, eval_set=[(Xte, yte_r)], verbose=False)
el = time.time() - t0
yp_r = xr.predict(Xte)
res['xgb_ttf'] = {
    'mae': round(float(mean_absolute_error(yte_r, yp_r)), 2),
    'r2': round(float(r2_score(yte_r, yp_r)), 4),
    'time': round(el, 1)
}
print(f'XGB[TTF] MAE={res["xgb_ttf"]["mae"]}h R2={res["xgb_ttf"]["r2"]}')

# Random Forest (faster settings)
print("Random Forest...")
for h, lc in [('24h','incident_in_24h'),('72h','incident_in_72h'),('7d','incident_in_7d')]:
    ytr = df.loc[:sp-1, lc]
    yte = df.loc[sp:, lc]
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=80, max_depth=10, min_samples_leaf=15,
        class_weight='balanced', random_state=42, n_jobs=1)
    rf.fit(Xtr, ytr)
    el = time.time() - t0
    yp = rf.predict_proba(Xte)[:,1]
    yc = rf.predict(Xte)
    res['rf_' + h] = {
        'roc': round(float(roc_auc_score(yte, yp)), 4),
        'pr': round(float(average_precision_score(yte, yp)), 4),
        'f1': round(float(f1_score(yte, yc, zero_division=0)), 4),
        'time': round(el, 1)
    }
    r = res['rf_' + h]
    print(f'RF [{h}] ROC={r["roc"]} PR={r["pr"]} F1={r["f1"]}')

# Save
print("Saving...")
tdf = df.iloc[sp:].copy().reset_index(drop=True)
tdf['xgb_ttf_pred'] = yp_r
tdf[['timestamp','cpu_load','mem_used_pct','ping_latency','wan_status','is_crash',
     'ttf_hours','xgb_ttf_pred','health_score',
     'incident_in_24h','incident_in_72h','incident_in_7d']].to_csv(
    '/home/claude/data/test_predictions.csv', index=False)
with open('/home/claude/data/benchmark_results.json', 'w') as f:
    json.dump(res, f, indent=2)
print('\nDone.')
print(json.dumps(res, indent=2))
