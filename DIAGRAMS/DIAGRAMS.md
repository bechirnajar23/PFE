# Diagrammes UML — Système Prédictif HGW

Ce fichier contient tous les diagrammes PlantUML du projet.  
Utilisez https://www.plantuml.com/plantuml/ ou un plugin IDE pour générer les images.

---

## 1. Diagramme de cas d'utilisation (Use Case)

```plantuml
@startuml
!theme plain
title Système Prédictif d'Incidents HGW - Cas d'utilisation

left to right direction

actor "Opérateur Réseau" as Ops
actor "Administrateur Système" as Admin
actor "Client Final" as Client

rectangle "Système Prédictif HGW" {
  usecase "Surveiller métriques\ntemps réel" as UC1
  usecase "Recevoir alertes\nprédictives" as UC2
  usecase "Consulter dashboard\nGrafana" as UC3
  usecase "Intervenir\npréventivement" as UC4
  usecase "Configurer seuils\nd'alerte" as UC5
  usecase "Réentraîner\nmodèles ML/DL" as UC6
  usecase "Exporter données\nhistoriques" as UC7
  usecase "Visualiser\nprédictions" as UC8
  usecase "Bénéficier de\nservice stable" as UC9
}

actor "HGW (CPE)" as HGW
actor "Collecteur Telnet" as Collector
actor "TimescaleDB" as DB

' Relations Opérateur
Ops --> UC1
Ops --> UC2
Ops --> UC3
Ops --> UC4
Ops --> UC8

' Relations Admin
Admin --> UC5
Admin --> UC6
Admin --> UC7

' Relations Client (bénéfice indirect)
UC4 ..> UC9 : <<provides>>
Client --> UC9

' Relations système
HGW --> UC1 : <<provides data>>
Collector --> UC1 : <<collects>>
DB --> UC3 : <<stores>>
UC6 ..> UC2 : <<improves>>

' Notes
note right of UC2
  Alertes envoyées par:
  - Email
  - Telegram
  - Slack
end note

note right of UC6
  Fréquence:
  - Hebdomadaire (auto)
  - À la demande (manuel)
end note

@enduml
```

---

## 2. Diagramme de séquence — Collecte et prédiction

```plantuml
@startuml
!theme plain
title Séquence 1: Collecte de télémétrie et prédiction

participant "HGW\n(CPE)" as HGW
participant "Collecteur\nTelnet" as Collector
participant "TimescaleDB" as DB
participant "Predictor\nML/DL" as Predictor
participant "Grafana\nDashboard" as Grafana
participant "Système\nd'Alerte" as Alert

== Boucle continue (toutes les 12 secondes) ==

Collector -> HGW: SSH/Telnet connect
activate HGW
HGW --> Collector: Connection OK

Collector -> HGW: Execute commands\n(top, ifconfig, ping, ps)
HGW --> Collector: Raw output

Collector -> Collector: Parse métriques\n(CPU, RAM, latence, etc.)
Collector -> Collector: Calculer LOCAL_STATUS\n(NORMAL/WARNING/URGENT)

Collector -> DB: INSERT monitor_snapshots\n(timestamp, métriques, status)
activate DB
DB --> Collector: ACK
deactivate DB

deactivate HGW

== Boucle prédiction (toutes les 5 minutes) ==

Predictor -> DB: SELECT last 60 minutes\nFROM monitor_snapshots
activate DB
DB --> Predictor: DataFrame (60 lignes)
deactivate DB

Predictor -> Predictor: Preprocess features\n(43 features engineered)

alt ML Court terme (15min-6h)
  Predictor -> Predictor: CatBoost.predict()\n(4 modèles)
  Predictor -> Predictor: Comparer proba > seuils
end

alt DL Long terme (3 jours)
  Predictor -> Predictor: Resample to 30min
  Predictor -> Predictor: Build sequences (24h lookback)
  Predictor -> Predictor: Bi-LSTM.predict()
end

Predictor -> Predictor: Agréger résultats\n(earliest_alert)

alt Alerte détectée
  Predictor -> DB: INSERT predictions_log\n(prob, horizon, alert=TRUE)
  activate DB
  DB --> Predictor: ACK
  deactivate DB
  
  Predictor -> Alert: Send alert\n(Telegram/Email)
  activate Alert
  Alert --> Predictor: Sent
  deactivate Alert
else Aucune alerte
  Predictor -> DB: INSERT predictions_log\n(alert=FALSE)
  activate DB
  DB --> Predictor: ACK
  deactivate DB
end

== Visualisation continue ==

Grafana -> DB: Query monitoring + predictions
activate DB
DB --> Grafana: Time series data
deactivate DB

Grafana -> Grafana: Render panels\n(CPU, RAM, alerts)

@enduml
```

---

## 3. Diagramme de séquence — Réentraînement hebdomadaire

```plantuml
@startuml
!theme plain
title Séquence 2: Réentraînement hebdomadaire automatique

participant "Cron/Task\nScheduler" as Cron
participant "weekly_retrain.py" as Script
participant "TimescaleDB" as DB
participant "train_multi_horizon.py" as TrainML
participant "train_bilstm.py" as TrainDL
participant "Système\nFichiers" as FS

== Dimanche 2h00 ==

Cron -> Script: Trigger\n(weekly schedule)
activate Script

Script -> DB: SELECT * FROM monitor_snapshots\nWHERE timestamp >= NOW() - 7 days
activate DB
DB --> Script: DataFrame (N rows)
deactivate DB

Script -> Script: Validate data quality\n(check missing values, outliers)

alt Data quality OK
  Script -> FS: Export to CSV\nweekly_batch_YYYYMMDD.csv
  activate FS
  FS --> Script: File saved
  deactivate FS
  
  Script -> TrainML: subprocess.run(\n  train_multi_horizon.py\n  --input-csv weekly_batch.csv)
  activate TrainML
  
  TrainML -> TrainML: Load data
  TrainML -> TrainML: Feature engineering\n(43 features)
  TrainML -> TrainML: 5-fold CV stratified
  
  loop Pour chaque horizon (15min, 30min, 1h, 6h)
    TrainML -> TrainML: CatBoost.fit()
    TrainML -> TrainML: Optimize threshold\n(F1, F2, F0.5)
    TrainML -> FS: Save model\n(catboost_Xmin_real.cbm)
    activate FS
    FS --> TrainML: Saved
    deactivate FS
  end
  
  TrainML -> FS: Save bundle.json\n(thresholds, metrics, features)
  activate FS
  FS --> TrainML: Saved
  deactivate FS
  
  TrainML --> Script: Training complete\n(exit code 0)
  deactivate TrainML
  
  Script -> Script: Check total data span\n(days since first record)
  
  alt ≥ 30 days real data available
    Script -> TrainDL: subprocess.run(\n  train_bilstm_3d_final.py\n  --input-csv weekly_batch.csv)
    activate TrainDL
    
    TrainDL -> TrainDL: Load real + synthetic data
    TrainDL -> TrainDL: Transfer learning:\n- Freeze Bi-LSTM layers\n- Fine-tune dense head
    
    TrainDL -> FS: Save bilstm_3d_finetuned.keras
    activate FS
    FS --> TrainDL: Saved
    deactivate FS
    
    TrainDL --> Script: Fine-tuning complete
    deactivate TrainDL
  else < 30 days
    Script -> Script: Log: DL skip\n(not enough data)
  end
  
  Script -> Script: Compare new metrics\nvs baseline
  
  alt Metrics improved
    Script -> FS: Deploy new models\n(mv to production/)
    activate FS
    FS --> Script: Deployed
    deactivate FS
    
    Script -> Script: Log: Deployment success
  else Metrics degraded
    Script -> Script: Log: Keep old models\n(manual review needed)
  end
  
else Data quality issues
  Script -> Script: Log: Training aborted\n(send alert to admin)
end

Script --> Cron: Exit (log result)
deactivate Script

@enduml
```

---

## 4. Diagramme d'activité — Processus de prédiction

```plantuml
@startuml
!theme plain
title Activité: Processus de prédiction ML/DL

start

:Timer 5 minutes déclenché;

:Requête TimescaleDB:\nSELECT last 60 min;

if (Données suffisantes?\n(≥60 lignes)) then (oui)
  :Load données dans DataFrame;
  
  partition "Preprocessing" {
    :Resample à 1-min\n(forward fill);
    :Calculer 43 features:\n- Rolling stats\n- Slopes\n- Lags\n- Cycliques;
    :Normalisation (StandardScaler);
  }
  
  partition "ML Predictions (Court terme)" {
    fork
      :CatBoost 15min\npredict_proba();
    fork again
      :CatBoost 30min\npredict_proba();
    fork again
      :CatBoost 1h\npredict_proba();
    fork again
      :CatBoost 6h\npredict_proba();
    end fork
    
    :Comparer probabilités\nvs seuils optimaux;
  }
  
  if (Session ≥ 24h continues?) then (oui)
    partition "DL Prediction (Long terme)" {
      :Resample à 30-min;
      :Build sequence:\n48 samples × 13 features;
      :Bi-LSTM predict();
      :Comparer vs seuil 0.5135;
    }
  else (non)
    :Log: DL skip\n(pas assez lookback);
  endif
  
  :Agréger résultats\ntous horizons;
  
  if (Au moins 1 alerte?) then (oui)
    :Identifier earliest_alert\n(horizon le plus court);
    
    :INSERT predictions_log\n(alert=TRUE, json);
    
    partition "Alerting" {
      :Construire message:\n- Horizon\n- Probabilité\n- Top features (SHAP);
      
      fork
        :Email notification;
      fork again
        :Telegram bot;
      fork again
        :Slack webhook;
      end fork
    }
    
    :Log: Alerte envoyée;
    
  else (non)
    :INSERT predictions_log\n(alert=FALSE);
    :Log: Aucune alerte;
  endif
  
else (non)
  :Log: Données insuffisantes\n(< 60 minutes);
endif

:Attendre 5 minutes;

stop

@enduml
```

---

## 5. Diagramme de déploiement (Deployment)

```plantuml
@startuml
!theme plain
title Diagramme de déploiement - Architecture Docker

node "Serveur de production" {
  
  artifact "docker-compose.yml" as compose
  
  node "Container TimescaleDB" as db_container {
    component "PostgreSQL 16" as postgres
    component "TimescaleDB Extension" as timescale
    database "hgw_monitoring" as db {
      storage "monitor_snapshots" as snap_table
      storage "predictions_log" as pred_table
    }
  }
  
  node "Container Grafana" as grafana_container {
    component "Grafana 10.x" as grafana
    artifact "dashboards/*.json" as dashboards
    artifact "datasources/*.yml" as datasources
  }
  
  node "Container Predictor" as pred_container {
    component "Python 3.11" as python
    artifact "predict_multi_horizon.py" as predictor
    folder "multi_horizon/" as ml_models {
      artifact "catboost_15min_real.cbm" as ml1
      artifact "catboost_30min_real.cbm" as ml2
      artifact "catboost_60min_real.cbm" as ml3
      artifact "catboost_360min_real.cbm" as ml4
    }
    folder "long_horizon_dl/" as dl_models {
      artifact "bilstm_3d.weights.h5" as dl1
      artifact "bilstm_loader.py" as loader
    }
  }
  
  node "Container Collector\n(Optionnel)" as coll_container {
    component "Python 3.11" as python2
    artifact "hgw_telnet_collector.py" as collector
  }
}

cloud "Réseau client" {
  node "HGW (CPE)" as hgw {
    interface "Telnet/SSH\nport 23" as telnet_if
  }
}

actor "Opérateur" as ops
actor "Admin" as admin

' Connections
compose ..> db_container : creates
compose ..> grafana_container : creates
compose ..> pred_container : creates
compose ..> coll_container : creates

coll_container -down-> telnet_if : collect metrics\nevery 12s
coll_container --> db_container : INSERT\nmonitor_snapshots

pred_container --> db_container : SELECT last 60min\nINSERT predictions_log
pred_container ..> ml_models : load models
pred_container ..> dl_models : load model

grafana_container --> db_container : SQL queries

ops -down-> grafana : http://localhost:3000
admin -down-> db_container : psql (port 5432)

note right of pred_container
  Prediction loop: 5 min
  4x CatBoost ML
  1x Bi-LSTM DL
end note

note right of coll_container
  Collection loop: 12s
  Parse: CPU, RAM, latency
  Calculate: LOCAL_STATUS
end note

@enduml
```

---

## 6. Diagramme de classes — Modèle de données

```plantuml
@startuml
!theme plain
title Diagramme de classes - Modèle de données

class MonitorSnapshot {
  + timestamp: TIMESTAMPTZ
  + LOCAL_STATUS: TEXT
  + STATUS_REASON: TEXT
  + CPU_USAGE_PERCENT: INTEGER
  + MEM_USAGE_PERCENT: INTEGER
  + WAN_STATE: TEXT
  + NET_LATENCY_MS: NUMERIC
  + NET_PING_STATUS: TEXT
  + CWMP_PROCESS_STATUS: TEXT
  + DHCP_PROCESS_STATUS: TEXT
  + NEMO_PROCESS_STATUS: TEXT
  + SYSTEM_UPTIME_SECONDS: INTEGER
  --
  + is_urgent(): BOOLEAN
  + is_warning(): BOOLEAN
}

class PredictionLog {
  + timestamp: TIMESTAMPTZ
  + horizon_min: INTEGER
  + probability: NUMERIC
  + alert: BOOLEAN
  + predictions_json: JSONB
  --
  + get_earliest_alert(): JSON
  + get_all_horizons(): LIST
}

class MLModel {
  + model_path: STRING
  + horizon_min: INTEGER
  + threshold: FLOAT
  + features: LIST[STRING]
  + metrics: DICT
  --
  + predict(df: DataFrame): FLOAT
  + predict_proba(df: DataFrame): ARRAY
  + load(): MODEL
}

class DLModel {
  + weights_path: STRING
  + architecture: STRING
  + scaler_path: STRING
  + seq_len: INTEGER
  + lookback_hours: INTEGER
  --
  + predict(sequences: ARRAY): ARRAY
  + build_sequences(df: DataFrame): ARRAY
  + load(): MODEL
}

class Predictor {
  + ml_models: LIST[MLModel]
  + dl_model: DLModel
  + threshold_strategy: STRING
  --
  + predict(df: DataFrame): DICT
  + get_earliest_alert(results: DICT): DICT
  + preprocess(df: DataFrame): DataFrame
}

class FeatureEngine {
  + rolling_windows: DICT
  + lag_periods: LIST[INT]
  --
  + compute_rolling_stats(df: DataFrame): DataFrame
  + compute_slopes(df: DataFrame): DataFrame
  + compute_lags(df: DataFrame): DataFrame
  + compute_cyclicals(df: DataFrame): DataFrame
  + engineer_all(df: DataFrame): DataFrame
}

MonitorSnapshot "1" -- "*" PredictionLog : generates >
Predictor "1" *-- "4" MLModel : contains
Predictor "1" *-- "1" DLModel : contains
Predictor "1" ..> FeatureEngine : uses
FeatureEngine ..> MonitorSnapshot : reads
MLModel ..> MonitorSnapshot : predicts from
DLModel ..> MonitorSnapshot : predicts from

@enduml
```

---

## Description pour IA / Rapport

### Contexte du projet

**Titre** : Système prédictif d'incidents sur Home Gateways (HGW)

**Objectif** : Développer un système de monitoring intelligent capable de prédire les pannes matérielles et logicielles sur des équipements réseau résidentiels (HGW/CPE) avant qu'elles ne surviennent, permettant ainsi une intervention préventive.

**Technologies** :
- **Backend** : Python 3.11, TensorFlow 2.15, CatBoost 1.2, scikit-learn
- **Base de données** : TimescaleDB (PostgreSQL optimisé séries temporelles)
- **Visualisation** : Grafana
- **Déploiement** : Docker, Docker Compose
- **Collecte** : Telnet/SSH vers CLI HGW

### Architecture fonctionnelle

Le système s'articule autour de 4 composants principaux :

1. **Collecteur de télémétrie** : Connexion automatisée toutes les 12 secondes au HGW via Telnet/SSH, exécution de commandes système (`top`, `ifconfig`, `ping`, `ps`), parsing des sorties, calcul d'un statut local (NORMAL/WARNING/URGENT), insertion dans TimescaleDB.

2. **Moteur de prédiction hybride ML/DL** :
   - **Tier ML (court terme)** : 4 modèles CatBoost entraînés sur données réelles, prédiction à 15min, 30min, 1h, 6h
   - **Tier DL (long terme)** : 1 modèle Bi-LSTM entraîné sur données synthétiques, prédiction à 3 jours
   - Exécution toutes les 5 minutes sur fenêtre glissante de 60 minutes
   - Feature engineering : 43 features (rolling stats, slopes, lags, cycliques, interactions)

3. **Système d'alerting** : Comparaison des probabilités prédites vs seuils optimisés (F1, F2, F0.5), identification de l'alerte la plus urgente, notification multi-canal (Telegram, Email, Slack) avec détails (horizon, probabilité, top features contributives via SHAP).

4. **Dashboard Grafana** : Visualisation temps réel des métriques (CPU, RAM, latence), état système, historique des prédictions, configuration des alertes Grafana natives.

### Workflow de prédiction

1. Timer déclenche requête TimescaleDB (SELECT last 60 min)
2. Preprocessing : resample 1-min, feature engineering (43 features)
3. Prédiction ML : 4 modèles CatBoost en parallèle
4. Prédiction DL : Si ≥24h lookback, Bi-LSTM sur séquences 30-min
5. Agrégation : Identifier earliest_alert (horizon le plus court)
6. Si alerte : INSERT predictions_log + envoi notifications
7. Sinon : INSERT predictions_log (alert=FALSE)
8. Grafana lit predictions_log et affiche courbes

### Réentraînement automatique

Chaque dimanche à 2h :
1. Export 7 derniers jours depuis TimescaleDB → CSV
2. Validation qualité données (missing values, outliers)
3. Entraînement 4 modèles CatBoost (5-fold CV)
4. Si ≥30 jours données : Fine-tuning Bi-LSTM (transfer learning)
5. Comparaison métriques vs baseline
6. Déploiement si amélioration constatée

### Performance

**ML (validation 5-fold CV sur 7.5j réels)** :
- 15min : PR-AUC 0.979, Recall 94.6%
- 30min : PR-AUC 0.989, Recall 99.3%
- 1h : PR-AUC 0.994, Recall 96.1%
- 6h : PR-AUC 0.996, Recall 99.3%

**DL (test set 15% synthétique)** :
- 3j : PR-AUC 0.962, Recall 99.1% (10/1167 incidents manqués)

**Test production** : 4/5 incidents réels détectés 5 min avant occurrence (80%).

### Déploiement Docker

3 containers orchestrés via docker-compose :
- `timescaledb` : PostgreSQL 16 + TimescaleDB (port 5432)
- `grafana` : Dashboard (port 3000)
- `predictor` : Service Python avec modèles embarqués

Volumes persistants pour données DB et config Grafana.
Health checks configurés sur tous services.

### Limites et perspectives

**Limites actuelles** :
- Modèle DL entraîné sur synthétique (7.5j réels insuffisants pour horizon 3j)
- Single-gateway (généralisation multi-HGW non validée)
- Features process-level manquantes (cwmp_rss, dhcp_rss, nemo_rss)

**Roadmap** :
- Collecte continue 30+ jours → Fine-tuning DL sur données réelles
- Modèle 7 jours
- Déploiement multi-HGW
- API REST
- AutoML pour optimisation continue
- Explainability avancée (LIME interactif)
