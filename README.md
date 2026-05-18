# HGW Predictive System

Système de maintenance prédictive pour Home Gateway (HGW). Collecte la télémétrie réseau d'une passerelle résidentielle, la stocke dans TimescaleDB, prédit les incidents à court et long terme avec CatBoost et LSTM, et expose les résultats via un dashboard React, Grafana et des alertes email.

L'objectif est d'anticiper les pannes avant qu'elles surviennent, en donnant aux équipes techniques un délai d'intervention exploitable et une explication lisible des causes.

---

## Fonctionnalités

- Collecte temps-réel des métriques HGW (CPU, mémoire, latence, débit WAN, état DHCP)
- Stockage time-series dans PostgreSQL/TimescaleDB avec rétention 90 jours
- Prédiction multi-horizon :
  - **CatBoost** court terme : 15 min, 30 min, 1 h, 6 h
  - **LSTM** long terme : 3 jours
- Explication locale SHAP par prédiction (XAI) — lisible en français
- Détection métier des états critiques indépendante du ML (règles CPU/RAM/état)
- Alertes email structurées avec diagnostic et actions recommandées
- Réentraînement automatique des modèles CatBoost tous les 7 jours
- Interface React avec courbes temps-réel, probabilités multi-horizon et explications
- Dashboard Grafana provisionné automatiquement

---

## Architecture

```
HGW (Telnet)
     │
     ▼
collector/          ← collecte, normalisation, insertion TimescaleDB
     │
     ▼
TimescaleDB         ← monitor_snapshots, predictions_log
     │
     ├──▶ predictor/     ← CatBoost (15/30/60/360 min) + LSTM (3 jours)
     │         │
     │         ├──▶ Email alert   (états URGENT / CRITICAL)
     │         └──▶ predictions_log
     │
     ├──▶ backend/       ← FastAPI REST, sert le frontend
     │
     ├──▶ frontend/      ← React SPA (port 8080)
     │
     └──▶ Grafana        ← dashboards (port 3000)
```

---

## Stack Docker

| Container | Port | Rôle |
|---|---:|---|
| `hgw_timescaledb` | 5432 | Base time-series |
| `hgw_collector` | — | Collecte Telnet HGW → DB |
| `hgw_predictor` | — | Prédiction toutes les 5 min + réentraînement |
| `hgw_backend_api` | 8000 | API REST FastAPI |
| `hgw_grafana` | 3000 | Dashboards (admin / admin) |
| `hgw_frontend` | 8080 | Interface React |

---

## Démarrage rapide

### 1. Configuration

```bash
cp .env.example .env
# Éditer .env : HGW_HOST, HGW_PASSWORD, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO
```

### 2. Lancer la stack

```bash
docker compose up --build -d
docker compose ps          # vérifier que tous les services sont Up
```

### 3. Accès

| Interface | URL |
|---|---|
| Frontend React | http://localhost:8080 |
| API backend | http://localhost:8000/health |
| Grafana | http://localhost:3000 (admin / admin) |

### 4. Logs

```bash
docker logs -f hgw_collector
docker logs -f hgw_predictor
docker logs -f hgw_backend_api
```

### 5. Arrêt

```bash
docker compose down        # conserve les données
docker compose down -v     # supprime aussi la base TimescaleDB
```

---

## Modèles

### CatBoost — court terme

4 modèles indépendants, chargés depuis `predictor/multi_horizon/` :

| Horizon | Fichier modèle | Seuil |
|---|---|---:|
| 15 min | `catboost_15min_real.cbm` | 0.6248 |
| 30 min | `catboost_30min_real.cbm` | 0.7833 |
| 1 h | `catboost_60min_real.cbm` | 0.5598 |
| 6 h | `catboost_360min_real.cbm` | 0.7749 |

Les features (43 au total) et seuils sont définis dans `predictor/multi_horizon/multi_horizon_bundle.json`.

### LSTM — 3 jours

- Nécessite **minimum 24 lignes à 30 min de résolution (12 h de données réelles)** pour s'activer
- Entraîné sur données synthétiques ; se calibre progressivement à partir de 30 jours de collecte réelle

Artefacts : `predictor/long_horizon_dl/`

### Explications SHAP (XAI)

Chaque prédiction CatBoost produit une explication locale :
- Les N premières features classées par importance
- Pour chaque feature : libellé lisible, valeur contextualisée en français, sens de l'impact (aggrave / protège), niveau (fort / modéré / faible)
- Une phrase de diagnostic globale générée automatiquement

Consultable dans le frontend (onglet Prédictions) et inclus dans les emails d'alerte.

---

## Alertes email

Une alerte est envoyée quand :
- Le statut métier courant est `URGENT` ou `CRITICAL` (règles CPU/RAM), **ou**
- Un modèle ML prédit un incident (`PREDICTED_INCIDENT` ou `CRITICAL`)

Cooldown : 60 minutes par type d'alerte. L'email contient :
- Le statut, l'horizon, la probabilité
- Les causes identifiées par SHAP
- Les actions recommandées

Configuration dans `.env` : `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_EMAIL_TO`, `ALERT_COOLDOWN_MINUTES`.

---

## Tests offline (sans Docker)

```bash
conda create -n hgw python=3.10 -y
conda activate hgw
pip install -r requirements.txt

cd predictor
# Tester sur le dernier point du CSV
python test_models.py "../data/test_full_scenarios.csv" --last

# 10 points aléatoires
python test_models.py "../data/test_full_scenarios.csv" --random 10

# Filtrer les cas urgents
python test_models.py "../data/test_full_scenarios.csv" --urgents 5

# Exporter pour analyse dans le notebook
python test_models.py "../data/test_full_scenarios.csv" --random 10 --export "../data/predictions_now.csv"
```

Réentraîner CatBoost :

```bash
cd predictor
python train_multi_horizon.py
```

---

## Structure du projet

```
.
├── collector/              # Collecte Telnet + insertion DB
├── predictor/              # Moteur de prédiction et entraînement
│   ├── multi_horizon/      # Modèles CatBoost + bundle JSON
│   ├── long_horizon_dl/    # Modèle LSTM + scaler + métadata
│   ├── test_models.py      # Moteur de prédiction + CLI de test
│   ├── predict_service.py  # Daemon Docker (5 min)
│   └── train_multi_horizon.py
├── backend/                # FastAPI REST
├── frontend/               # React SPA (Vite)
├── grafana/                # Dashboards et provisioning
├── sql/                    # Schéma TimescaleDB
├── notebooks/              # Viz.ipynb — analyse des prédictions exportées
├── data/                   # Scénarios de test CSV
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Schéma de données

### `monitor_snapshots`
Télémétrie brute collectée toutes les 5 min :
`timestamp`, `local_status`, `status_reason`, `cpu_usage_percent`, `mem_usage_percent`, `net_latency_ms`, `wan_rx_rate_kbps`, `wan_tx_rate_kbps`, `wan_state`

### `predictions_log`
Prédictions ML stockées à chaque cycle :
`timestamp`, `horizon`, `probability`, `threshold`, `alert`, `decision_level`, `decision_message`, `explainer_json`, `model_version`

---

## Limites connues

- `cwmp_rss_mb`, `dhcp_rss_mb`, `nemo_rss_mb` sont fixés à 0.0 — le collecteur Telnet ne les extrait pas encore
- Le modèle LSTM est entraîné sur des données synthétiques ; les prédictions sont directionnellement valides mais moins calibrées que CatBoost avant 30 jours de données réelles
- La collecte Telnet requiert un accès réseau direct à la HGW (`HGW_HOST` dans `.env`)
