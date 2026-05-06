# HGW Predictive Maintenance

Systeme de maintenance predictive pour Home Gateway (HGW). Le projet collecte la telemetrie d'une passerelle residentielle, stocke les mesures dans TimescaleDB, applique des modeles Machine Learning / Deep Learning multi-horizon, puis expose les resultats dans Grafana et dans des exports CSV pour analyse.

Le but est d'anticiper les incidents avant le crash afin de reduire les interruptions de service, ameliorer la QoS et donner aux equipes techniques un delai d'intervention exploitable.

## Fonctionnalites

- Collecte temps reel des metriques HGW via un service d'ecoute cote PC/serveur.
- Stockage time-series dans PostgreSQL / TimescaleDB.
- Prediction multi-horizon:
  - CatBoost court terme: 15 min, 30 min, 1 h, 6 h.
  - LSTM long terme: 3 jours.
- Prediction continue toutes les 5 minutes pendant toute la journee.
- Reentrainement automatique des modeles tous les 7 jours.
- Detection metier des etats critiques actuels (`URGENT`, `CRITICAL`).
- Explication des alertes via regles metier et explainer ML/SHAP.
- Notification utilisateur uniquement pour les etats `URGENT` ou `CRITICAL`.
- Export CSV des predictions pour notebooks de visualisation.
- Dashboard Grafana pour suivi operationnel.

## Architecture

```text
HGW / CPE
  |
  | Metriques reseau
  v
collector/
  |  collecte + normalisation locale
  v
TimescaleDB
  |  tables monitor_snapshots, predictions_log, vues Grafana
  v
predictor/
  |  CatBoost 15/30/60/360 min + LSTM 3 jours
  v
Grafana / Notebooks / Alertes
```

Documentation d'architecture detaillee et diagrammes UML:

```text
docs/ARCHITECTURE_UML.md
```

### Conception logique

| Couche | Role | Fichiers principaux |
|---|---|---|
| Collecte | Recevoir les metriques HGW et construire des snapshots | `collector/data_collection.py`, `collector/data_logger.py` |
| Stockage | Persister telemetrie, predictions et logs pipeline | `sql/init.sql`, `sql/schema.sql`, `sql/10_pipeline_tables.sql` |
| Prediction | Charger les modeles, calculer le risque et expliquer les alertes | `predictor/test_models.py`, `predictor/predict_multi_horizon.py`, `predictor/predict_service.py` |
| Modeles | Artefacts ML/DL entraines | `predictor/multi_horizon/`, `predictor/long_horizon_dl/` |
| Visualisation | Dashboards temps reel et analyse | `grafana/dashboards/hgw_monitoring.json`, `grafana/dashboards/hgw_predictions.json`, `notebooks/Viz.ipynb` |

## Modeles disponibles

### Tier 1 - CatBoost court terme

Les modeles CatBoost sont utilises pour les alertes immediates et proches. Ils exploitent des features tabulaires issues des mesures recentes: moyennes glissantes, max, pentes CPU/RAM, latence, debit WAN, interactions CPU/RAM et cyclicite horaire.

| Horizon | Fichier | Seuil actuel |
|---|---|---:|
| 15 min | `predictor/multi_horizon/catboost_15min_real.cbm` | 0.6248 |
| 30 min | `predictor/multi_horizon/catboost_30min_real.cbm` | 0.7833 |
| 1 h | `predictor/multi_horizon/catboost_60min_real.cbm` | 0.5598 |
| 6 h | `predictor/multi_horizon/catboost_360min_real.cbm` | 0.7749 |

La configuration des modeles et des features est centralisee dans:

```text
predictor/multi_horizon/multi_horizon_bundle.json
```

### Tier 2 - LSTM long terme

Le modele LSTM couvre l'horizon 3 jours. Il utilise une sequence temporelle avec les features longues: charge CPU, memoire, latence, packet loss, WAN, statistiques 24 h, pentes 6 h et `health_score`.

Artefacts:

```text
predictor/long_horizon_dl/lstm_3day.keras
predictor/long_horizon_dl/lstm_scaler.pkl
predictor/long_horizon_dl/lstm_metdata.json
```

Seuil actuel: `0.3101`.

## Structure du projet

```text
.
|-- collector/                  # Collecte metriques et insertion DB
|-- predictor/                  # Prediction, entrainement, tests offline
|   |-- multi_horizon/          # Modeles CatBoost + bundle
|   |-- long_horizon_dl/        # Modele LSTM + scaler
|   |-- test_models.py          # Test complet sur CSV + export predictions
|   |-- train_multi_horizon.py  # Entrainement CatBoost
|   `-- predict_service.py      # Service de prediction pour Docker
|-- sql/                        # Schemas TimescaleDB
|-- grafana/                    # Dashboards et provisioning
|-- notebooks/                  # Visualisation et experimentation
|-- data/                       # Donnees locales, exports, scenarios de test
|-- docker-compose.yml          # Stack principale
`-- requirements.txt            # Dependances Python locales
```

## Prerequis

- Python 3.10 ou 3.11.
- Docker Desktop ou Docker Engine avec Docker Compose v2.
- 6 Go RAM minimum recommandes si TensorFlow est installe localement.
- Acces reseau a la HGW si la collecte temps reel est active.

Sous Windows, le projet fonctionne depuis PowerShell ou WSL. Pour Docker, WSL est souvent plus stable.

## Setup local Python

Depuis la racine du projet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Avec Conda:

```powershell
conda create -n hgw_project python=3.10 -y
conda activate hgw_project
pip install -r requirements.txt
```

Verifier que les modeles sont charges:

```powershell
cd predictor
python test_models.py "..\data\test_full_scenarios.csv" --random 5
```

Exporter les resultats pour le notebook:

```powershell
cd predictor
python test_models.py "..\data\test_full_scenarios.csv" --random 10 --export "..\data\predictions_now.csv"
```

Le fichier exporte peut ensuite etre lu par:

```text
notebooks/Viz.ipynb
```

## Configuration

Creer un fichier `.env` a partir de `.env.example`:

```powershell
Copy-Item .env.example .env
```

Puis adapter les valeurs:

```env
DB_DSN=postgresql://hgw_user:hgw_password@timescaledb:5432/hgw_monitoring
POSTGRES_USER=hgw_user
POSTGRES_PASSWORD=hgw_password
POSTGRES_DB=hgw_monitoring

HGW_HOST=192.168.1.1
HGW_USER=root
HGW_PASSWORD=change_me
HGW_PORT=23
COLLECTION_INTERVAL=5
```

Important: ne pas mettre de vrais mots de passe dans Git. Le fichier `.env` est ignore par `.gitignore`.

## Deploiement Docker

### 1. Lancer la stack principale

```bash
docker compose up --build -d
```

Services demarres:

| Service | Container | Port | Role |
|---|---|---:|---|
| TimescaleDB | `hgw_timescaledb` | 5432 | Base time-series |
| Collector | `hgw_collector` | - | Collecte HGW |
| SMS | `hgw_sms_service` | 5000 | API notification SMS |
| Predictor | `hgw_predictor` | - | Prediction toutes les 5 min + reentrainement 7 jours |
| Grafana | `hgw_grafana` | 3000 | Dashboard |

Verifier l'etat:

```bash
docker compose ps
docker compose logs -f timescaledb
docker compose logs -f collector
docker compose logs -f predictor
docker compose logs -f sms
```

### 2. Initialiser ou verifier la base

Le fichier `sql/init.sql` est monte au demarrage de TimescaleDB. Il cree notamment:

- `monitor_snapshots`
- `predictions_log`
- `alerts`
- `hourly_stats`

Pour appliquer manuellement un schema supplementaire:

```bash
docker exec -i hgw_timescaledb psql -U hgw_user -d hgw_monitoring < sql/schema.sql
docker exec -i hgw_timescaledb psql -U hgw_user -d hgw_monitoring < sql/10_pipeline_tables.sql
docker exec -i hgw_timescaledb psql -U hgw_user -d hgw_monitoring < sql/20_dashboard_migration.sql
```

Verifier les tables:

```bash
docker exec -it hgw_timescaledb psql -U hgw_user -d hgw_monitoring -c "\dt"
```

### 3. Acceder a Grafana

Ouvrir:

```text
http://localhost:3000
```

Identifiants par defaut Grafana:

```text
admin / admin
```

Datasource PostgreSQL / TimescaleDB:

```text
Host: timescaledb:5432
Database: hgw_monitoring
User: hgw_user
Password: hgw_password
TLS/SSL: disable
```

Dashboards provisionnes automatiquement:

```text
grafana/dashboards/hgw_monitoring.json
grafana/dashboards/hgw_predictions.json
```

Avec Docker Compose, Grafana charge automatiquement la datasource TimescaleDB et ces deux dashboards au demarrage.

Guide de validation complet:

```text
docs/TEST_VALIDATION_GUIDE.md
```

## Commandes utiles

Tester les modeles sur un CSV:

```powershell
cd predictor
python test_models.py "..\data\test_full_scenarios.csv" --last
python test_models.py "..\data\test_full_scenarios.csv" --random 10
python test_models.py "..\data\test_full_scenarios.csv" --urgents 5
python test_models.py "..\data\test_full_scenarios.csv" --time "2026-05-01 08:20:00"
```

Exporter pour visualisation:

```powershell
python test_models.py "..\data\test_full_scenarios.csv" --random 10 --export "..\data\predictions_now.csv"
python test_models.py "..\data\test_full_scenarios.csv" --random 10 --export "..\data\predictions_now.csv" --append-export
```

Entrainer les modeles CatBoost:

```powershell
cd predictor
python train_multi_horizon.py
```

Le service Docker `hgw_predictor` execute deja cette logique en continu:

```text
prediction toutes les 5 minutes
reentrainement tous les 7 jours
notification Email/SMS si alerte
```

Les notifications utilisateur sont limitees aux etats courants `URGENT` ou `CRITICAL`. Les etats `WARNING` et les predictions seules restent consultables dans Grafana.

Surveiller les logs Docker:

```bash
docker compose logs -f
docker logs -f hgw_collector
docker logs -f hgw_predictor
docker logs -f hgw_grafana
```

Arreter:

```bash
docker compose down
```

Supprimer les volumes de donnees Docker:

```bash
docker compose down -v
```

Attention: `down -v` supprime la base TimescaleDB locale.

## Schema de donnees principal

### `monitor_snapshots`

Table historique de collecte brute. Colonnes importantes:

- `timestamp`
- `LOCAL_STATUS`, `STATUS_REASON`
- `CPU_USAGE_PERCENT`
- `MEM_USAGE_PERCENT`
- `WAN_STATE`
- `WAN_RX_RATE_KBPS`, `WAN_TX_RATE_KBPS`
- `NET_LATENCY_MS`, `NET_PING_STATUS`
- `DHCP_PROCESS_STATUS`, `DHCP_DATA_STATE`, `DHCP_V6_STATE`

### `predictions_log`

Table des predictions produites par le moteur ML/DL:

- `timestamp`
- `horizon` ou `horizon_min`
- `probability`
- `threshold`
- `alert`
- `decision_level`
- `decision_message`
- `model_version`

### Tables avancees

`sql/schema.sql` et `sql/10_pipeline_tables.sql` ajoutent des tables plus propres pour une architecture production:

- `hgw_telemetry`
- `hgw_predictions`
- `hgw_incidents`
- `hgw_drift_log`
- `monitor_snapshots_clean`
- `model_training_dataset`
- `pipeline_runs`

## Workflow conseille pour demo / soutenance

1. Lancer la stack Docker.
2. Verifier que TimescaleDB recoit des snapshots.
3. Executer `test_models.py` sur un scenario CSV representatif.
4. Exporter les predictions vers `data/predictions_now.csv`.
5. Ouvrir `notebooks/Viz.ipynb` pour comparer reel vs predit.
6. Montrer Grafana avec:
   - health score;
   - probabilites 24 h / 72 h;
   - CPU/RAM/latence;
   - alertes actives;
   - evenements crash.

## Limites connues

- Le collecteur cible une HGW accessible sur le reseau et une configuration de collecte adaptee au mode de transmission choisi.
- Les performances dependent fortement de la qualite et de la duree des donnees collectees.
- Le modele LSTM 3 jours demande beaucoup plus d'historique que les modeles CatBoost court terme.
- Certains scripts historiques sont conserves dans `corbeille/` et `scripts/archive/`; ils ne sont pas necessaires au lancement principal.
- Les identifiants reels doivent rester dans `.env`, jamais dans le code ni dans le README.

## Pistes d'amelioration

- Ajouter la collecte RSS des processus `cwmp`, `dhcp`, `nemo`.
- Unifier les schemas `monitor_snapshots` et `hgw_telemetry`.
- Brancher les alertes SMS/email sur les sorties `predictions_log`.
- Ajouter un service API REST pour exposer la prediction courante.
- Mettre en place un monitoring de drift et un retraining automatique.
- Un orchestrateur peut etre ajoute en production si besoin.

## Resume

Ce projet propose une chaine complete de maintenance predictive HGW:

```text
Collecte metriques -> TimescaleDB -> Feature engineering -> CatBoost/LSTM -> Alertes -> Grafana/Notebook
```

La force de la conception est l'approche hybride: CatBoost pour les incidents proches, LSTM pour le risque long terme, avec une base time-series et un dashboard permettant de relier les predictions aux mesures reelles.
