"""
02_ml_pipeline_ttf.py
======================
Pipeline ML complet sur le dataset synthétique HGW avec PIDs.

Étapes :
  1. Chargement & nettoyage
  2. Feature Engineering (slopes, rolling stats, lag features)
  3. Modèle de régression TTF  — XGBoost (Days_to_failure)
  4. Modèle de classification  — XGBoost (is_crash dans les N prochains pas)
  5. Root Cause Analysis (RCA) — corrélation cwmp vs RAM totale
  6. Sauvegarde des artefacts (modèles, graphiques)

Dépendances :
  pip install pandas numpy scikit-learn xgboost matplotlib shap
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # sans affichage GUI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_absolute_error, r2_score,
                             classification_report, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import joblib
import json
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
CSV_IN       = "synthetic_hgw_pid.csv"
OUTPUT_DIR   = Path("ml_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MEM_TOTAL_MB = 936
STEP_MIN     = 15           # minutes entre deux snapshots
ROLLING_WIN  = [5, 20, 60]  # fenêtres en nombre de pas (75 min, 5h, 15h)
LOOKAHEAD    = 8            # pas (= 2h) → horizon de classification
CRASH_THRESH = 90           # % RAM : seuil WARNING pour la visualisation


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & NETTOYAGE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. Chargement du dataset")
print("=" * 60)

df = pd.read_csv(CSV_IN, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"  Lignes : {len(df):,}  |  Colonnes : {df.shape[1]}")
print(f"  Période : {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"  Crashs  : {df['is_crash'].sum():,} ({df['is_crash'].mean()*100:.2f}%)")
print(f"  RAM max : {df['ram_total_pct'].max():.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Feature Engineering")
print("=" * 60)

# ── Slopes (pente sur W pas) ──────────────────────────────────────────────────
for col in ["ram_total_pct", "pid_cwmp_vmrss_mb", "pid_dhcp_vmrss_mb",
            "pid_nemo_vmrss_mb", "cpu_total"]:
    for w in [5, 20]:
        df[f"{col}_slope_{w}"] = (
            df[col].diff(w) / w
        )

# ── Rolling statistics ────────────────────────────────────────────────────────
for col in ["ram_total_pct", "pid_cwmp_vmrss_mb", "cpu_total",
            "ram_cwmp_pct"]:
    for w in ROLLING_WIN:
        df[f"{col}_rmean_{w}"]  = df[col].rolling(w, min_periods=2).mean()
        df[f"{col}_rstd_{w}"]   = df[col].rolling(w, min_periods=2).std()
        df[f"{col}_rmax_{w}"]   = df[col].rolling(w, min_periods=2).max()

# ── Lag features (valeur N pas avant) ────────────────────────────────────────
for col in ["ram_total_pct", "pid_cwmp_vmrss_mb", "cpu_total"]:
    for lag in [1, 5, 20]:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)

# ── Features dérivées ────────────────────────────────────────────────────────
df["cwmp_share_of_ram"] = df["pid_cwmp_vmrss_mb"] / MEM_TOTAL_MB * 100
df["cwmp_over_total"]   = df["ram_cwmp_pct"] / (df["ram_total_pct"] + 1e-6)
df["wan_fail"]          = (df["wan_state"] == 0).astype(int)

# ── Cible décalée pour la classification ─────────────────────────────────────
# crash_in_N : 1 si un crash arrive dans les LOOKAHEAD prochains pas
df["crash_in_N"] = (
    df["is_crash"]
    .rolling(window=LOOKAHEAD, min_periods=1)
    .max()
    .shift(-LOOKAHEAD)
    .fillna(0)
    .astype(int)
)

# ── Suppression des lignes avec trop de NaN (début de série) ─────────────────
df.dropna(subset=["ram_total_pct_slope_20", "pid_cwmp_vmrss_mb_rmean_60"],
          inplace=True)
df.reset_index(drop=True, inplace=True)

# Liste des features
FEATURE_COLS = [c for c in df.columns if c not in
                ("timestamp", "days_to_failure", "is_crash", "crash_in_N")]

print(f"  Features générées : {len(FEATURE_COLS)}")
print(f"  Lignes après nettoyage : {len(df):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODÈLE REGRESSION — Days to Failure (TTF)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Régression TTF — XGBoost")
print("=" * 60)

# On entraîne uniquement sur les snapshots avant crash (TTF > 0)
mask_reg = (df["days_to_failure"] > 0) & (~df[FEATURE_COLS].isnull().any(axis=1))
df_reg   = df[mask_reg].copy()

X_reg = df_reg[FEATURE_COLS].fillna(0)
y_reg = df_reg["days_to_failure"]

# Split temporel (80/20)
split = int(len(X_reg) * 0.8)
X_tr, X_te = X_reg.iloc[:split], X_reg.iloc[split:]
y_tr, y_te = y_reg.iloc[:split], y_reg.iloc[split:]

reg_model = xgb.XGBRegressor(
    n_estimators       = 400,
    learning_rate      = 0.05,
    max_depth          = 6,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_weight   = 5,
    early_stopping_rounds = 30,
    eval_metric        = "mae",
    random_state       = 42,
    verbosity          = 0,
)
reg_model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              verbose=False)

y_pred_reg = reg_model.predict(X_te)

mae = mean_absolute_error(y_te, y_pred_reg)
r2  = r2_score(y_te, y_pred_reg)
print(f"  MAE  : {mae:.3f} jours")
print(f"  R²   : {r2:.4f}")

joblib.dump(reg_model, OUTPUT_DIR / "xgb_ttf_regression.pkl")

# ── Graphique prédiction vs réel ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ts_te = df_reg["timestamp"].iloc[split:].values
ax.plot(ts_te, y_te.values, label="TTF réel",   color="#185FA5", lw=1, alpha=0.7)
ax.plot(ts_te, y_pred_reg,  label="TTF prédit", color="#E24B4A", lw=1, alpha=0.7,
        linestyle="--")
ax.set_title("Days-to-Failure : réel vs prédit (jeu de test temporel)")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Jours avant crash")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_ttf_regression.png", dpi=150)
plt.close()
print(f"  → fig_ttf_regression.png sauvegardé")


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODÈLE CLASSIFICATION — Crash dans les 2h ?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"4. Classification — Crash dans les {LOOKAHEAD * STEP_MIN} min ?")
print("=" * 60)

mask_clf = ~df[FEATURE_COLS].isnull().any(axis=1)
df_clf   = df[mask_clf].copy()

X_clf = df_clf[FEATURE_COLS].fillna(0)
y_clf = df_clf["crash_in_N"]

split_c = int(len(X_clf) * 0.8)
X_ctr, X_cte = X_clf.iloc[:split_c], X_clf.iloc[split_c:]
y_ctr, y_cte = y_clf.iloc[:split_c], y_clf.iloc[split_c:]

neg, pos  = (y_ctr == 0).sum(), (y_ctr == 1).sum()
scale_pos = neg / max(pos, 1)

clf_model = xgb.XGBClassifier(
    n_estimators      = 400,
    learning_rate     = 0.05,
    max_depth         = 5,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = scale_pos,
    early_stopping_rounds = 30,
    eval_metric       = "aucpr",
    random_state      = 42,
    verbosity         = 0,
)
clf_model.fit(X_ctr, y_ctr,
              eval_set=[(X_cte, y_cte)],
              verbose=False)

y_prob_clf = clf_model.predict_proba(X_cte)[:, 1]
y_pred_clf = (y_prob_clf >= 0.5).astype(int)

print(classification_report(y_cte, y_pred_clf,
                             target_names=["Normal", "Pré-Crash"],
                             digits=3))
print(f"  AUC-ROC : {roc_auc_score(y_cte, y_prob_clf):.4f}")

joblib.dump(clf_model, OUTPUT_DIR / "xgb_crash_classifier.pkl")

# ── Graphique probabilité de crash ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ts_cte = df_clf["timestamp"].iloc[split_c:].values
ram_cte = df_clf["ram_total_pct"].iloc[split_c:].values

ax1.plot(ts_cte, ram_cte, color="#185FA5", lw=1)
ax1.axhline(CRASH_THRESH, color="#E24B4A", lw=0.8, linestyle="--", label="Seuil 90%")
ax1.set_ylabel("RAM totale (%)")
ax1.legend(fontsize=8)
ax1.set_title("RAM totale et probabilité de crash prédite")

ax2.fill_between(ts_cte, y_prob_clf, alpha=0.5,
                 color="#E24B4A", label="P(crash dans 2h)")
ax2.set_ylabel("Probabilité")
ax2.set_xlabel("Timestamp")
ax2.legend(fontsize=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_crash_probability.png", dpi=150)
plt.close()
print(f"  → fig_crash_probability.png sauvegardé")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ROOT CAUSE ANALYSIS (RCA)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. Root Cause Analysis — isolation du processus fautif")
print("=" * 60)

# ── 5a. Corrélations Pearson ──────────────────────────────────────────────────
process_cols = ["pid_dhcp_vmrss_mb", "pid_cwmp_vmrss_mb", "pid_nemo_vmrss_mb"]
corr_vals = {col: df["ram_total_pct"].corr(df[col]) for col in process_cols}
print("  Corrélation Pearson avec RAM totale :")
for k, v in sorted(corr_vals.items(), key=lambda x: -abs(x[1])):
    print(f"    {k:30s} : {v:+.4f}")

# ── 5b. Importance SHAP (modèle régression) ───────────────────────────────────
print("\n  Calcul SHAP values (modèle TTF) ...")
explainer   = shap.TreeExplainer(reg_model)
# Sous-échantillon de 2000 lignes pour la vitesse
sample_idx  = np.random.choice(len(X_te), min(2000, len(X_te)), replace=False)
shap_vals   = explainer.shap_values(X_te.iloc[sample_idx])

fig, ax = plt.subplots(figsize=(8, 7))
shap.summary_plot(shap_vals, X_te.iloc[sample_idx],
                  plot_type="bar", max_display=15,
                  show=False)
plt.title("SHAP — Top 15 features (modèle TTF)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_shap_ttf.png", dpi=150)
plt.close()
print(f"  → fig_shap_ttf.png sauvegardé")

# ── 5c. Évolution temporelle des 3 processus ─────────────────────────────────
# On prend le premier cycle (0 → premier crash)
first_crash_idx = df[df["is_crash"] == 1].index[0] if df["is_crash"].any() else len(df)
df_cycle = df.iloc[:first_crash_idx + 1].copy()

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

colors = {"pid_dhcp_vmrss_mb": "#639922",
          "pid_cwmp_vmrss_mb": "#E24B4A",
          "pid_nemo_vmrss_mb": "#185FA5"}

labels = {"pid_dhcp_vmrss_mb": "dhcp",
          "pid_cwmp_vmrss_mb": "cwmp-plugin  ← FUITE",
          "pid_nemo_vmrss_mb": "nemo-core"}

for ax, col in zip(axes, process_cols):
    ax.plot(df_cycle["timestamp"], df_cycle[col],
            color=colors[col], lw=1.2, label=labels[col])
    ax.set_ylabel("VmRSS (MB)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Timestamp")
axes[0].set_title("Évolution VmRSS par processus — 1er cycle jusqu'au crash")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %Hh"))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_rca_process_leak.png", dpi=150)
plt.close()
print(f"  → fig_rca_process_leak.png sauvegardé")

# ── 5d. Double axe RAM totale / cwmp ─────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 4))
ax2 = ax1.twinx()

ax1.plot(df_cycle["timestamp"], df_cycle["ram_total_pct"],
         color="#185FA5", lw=1.5, label="RAM totale (%)")
ax1.axhline(CRASH_THRESH, color="#E24B4A", lw=0.8, linestyle="--")
ax1.set_ylabel("RAM totale (%)", color="#185FA5")
ax1.tick_params(axis="y", labelcolor="#185FA5")

ax2.plot(df_cycle["timestamp"], df_cycle["pid_cwmp_vmrss_mb"],
         color="#E24B4A", lw=1.5, linestyle="--", label="cwmp VmRSS (MB)")
ax2.set_ylabel("cwmp VmRSS (MB)", color="#E24B4A")
ax2.tick_params(axis="y", labelcolor="#E24B4A")

ax1.set_title("Corrélation RAM totale ↔ cwmp-plugin VmRSS")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_rca_dual_axis.png", dpi=150)
plt.close()
print(f"  → fig_rca_dual_axis.png sauvegardé")


# ══════════════════════════════════════════════════════════════════════════════
# 6. RÉSUMÉ JSON
# ══════════════════════════════════════════════════════════════════════════════
summary = {
    "dataset": {
        "rows"         : len(df),
        "crash_count"  : int(df["is_crash"].sum()),
        "crash_rate"   : round(df["is_crash"].mean() * 100, 2),
        "ram_max_pct"  : round(float(df["ram_total_pct"].max()), 2),
        "cwmp_max_mb"  : round(float(df["pid_cwmp_vmrss_mb"].max()), 2),
    },
    "regression_ttf": {
        "mae_days" : round(mae, 4),
        "r2"       : round(r2, 4),
    },
    "rca_correlations": {k: round(v, 4) for k, v in corr_vals.items()},
    "feature_count"   : len(FEATURE_COLS),
    "outputs"         : [str(p) for p in sorted(OUTPUT_DIR.iterdir())],
}
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("Pipeline terminé. Artefacts dans :", OUTPUT_DIR)
print("=" * 60)
for p in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {p.name}")
print()
print(json.dumps(summary, indent=2))
