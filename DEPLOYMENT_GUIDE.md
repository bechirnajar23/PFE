# 🏠 HGW Monitoring + ML/DL Prediction System - Guide complet

Système intégré de monitoring et prédiction d'incidents pour Home Gateways.

## 🎯 Architecture

```
┌─────────────┐       ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│   HGW CPE   │ ────→ │  Collector   │ ────→ │ TimescaleDB  │ ────→ │  Predictor   │
│  (Telnet)   │       │  (Telnet)    │       │              │       │   ML + DL    │
└─────────────┘       └──────────────┘       └──────────────┘       └──────────────┘
                                                      ↓                       ↓
                                              ┌──────────────┐       ┌──────────────┐
                                              │   Grafana    │←──────│predictions_log│
                                              │  Dashboard   │       │              │
                                              └──────────────┘       └──────────────┘
```

## 📦 Composants

### 1. **Collector** (Port interne)
- Connexion Telnet au HGW toutes les 5s
- Collecte : CPU, RAM, latence, WAN, DHCP
- Stockage direct dans TimescaleDB

### 2. **TimescaleDB** (Port 5432)
- Base PostgreSQL optimisée séries temporelles
- Tables : `monitor_snapshots`, `predictions_log`
- Rétention : 90 jours

### 3. **Predictor** (Service interne)
- Lit les 60 dernières minutes de données
- Applique 4 modèles ML (15min-6h) + 1 DL (3j)
- Écrit les prédictions dans `predictions_log`
- Fréquence : toutes les 5 minutes

### 4. **Grafana** (Port 3000)
- Dashboard temps réel
- Panels : CPU, RAM, latence, WAN, prédictions ML/DL
- Login : admin/admin

---

## 🚀 Déploiement

### Prérequis
- Docker 20.10+
- Docker Compose 2.0+
- Accès réseau au HGW (Telnet port 23)

### Configuration

Créer un fichier `.env` :

```bash
# HGW Connection
HGW_HOST=192.168.1.1
HGW_USER=root
HGW_PASSWORD=sah

# Collection
COLLECTION_INTERVAL=5

# Prediction
PREDICTION_INTERVAL=300
```

### Démarrage

```bash
# 1. Cloner / extraire le projet
cd hgw-predictor

# 2. Vérifier la structure
ls -la
# Doit contenir: docker-compose-complete.yml, collector/, predictor/, sql/, grafana/

# 3. Copier les modèles ML/DL dans predictor/
cp -r multi_horizon/ predictor/
cp -r long_horizon_dl/ predictor/

# 4. Lancer le stack complet
docker-compose -f docker-compose-complete.yml up -d

# 5. Vérifier les logs
docker-compose -f docker-compose-complete.yml logs -f collector
docker-compose -f docker-compose-complete.yml logs -f predictor

# 6. Accéder à Grafana
# http://localhost:3000 (admin/admin)
```

### Vérification

```bash
# Vérifier les services
docker-compose -f docker-compose-complete.yml ps

# Doit afficher:
# hgw_timescaledb   Up (healthy)
# hgw_collector     Up
# hgw_predictor     Up
# hgw_grafana       Up

# Vérifier les données dans TimescaleDB
docker exec -it hgw_timescaledb psql -U hgw_user -d hgw_monitoring -c "SELECT COUNT(*) FROM monitor_snapshots;"

# Vérifier les prédictions
docker exec -it hgw_timescaledb psql -U hgw_user -d hgw_monitoring -c "SELECT * FROM predictions_log ORDER BY timestamp DESC LIMIT 5;"
```

---

## 📊 Grafana Dashboard

### Accès
1. Ouvrir http://localhost:3000
2. Login : `admin` / `admin`
3. Aller dans Dashboards → HGW Monitoring + ML/DL Predictions

### Panels disponibles

| Panel | Description | Source |
|---|---|---|
| **État HGW** | Statut actuel (NORMAL/WARNING/URGENT) | `monitor_snapshots` |
| **CPU Usage** | Utilisation CPU en temps réel | `monitor_snapshots` |
| **Memory Usage** | Utilisation RAM en temps réel | `monitor_snapshots` |
| **Latency** | Latence réseau (ping 8.8.8.8) | `monitor_snapshots` |
| **WAN Traffic** | Débit RX/TX (KB/s) | `monitor_snapshots` |
| **ML/DL Predictions** | Probabilités par horizon (15min-3j) | `predictions_log` |

### Alertes configurables

Grafana peut envoyer des alertes email/Telegram/Slack quand :
- Probabilité prédiction > seuil (ex: 70%)
- CPU > 85%
- MEM > 90%
- LOCAL_STATUS = URGENT

---

## 🔧 Troubleshooting

### Le collector ne se connecte pas au HGW

```bash
# Vérifier la connectivité
telnet 192.168.1.1 23

# Vérifier les logs
docker logs hgw_collector

# Vérifier les credentials dans .env
cat .env
```

### Pas de données dans TimescaleDB

```bash
# Vérifier que le collector tourne
docker logs hgw_collector | grep "SNAPSHOT"

# Vérifier la connexion DB
docker exec -it hgw_collector python -c "from data_logger import engine; print(engine)"
```

### Pas de prédictions

```bash
# Vérifier les modèles
docker exec -it hgw_predictor ls -la multi_horizon/

# Vérifier les logs predictor
docker logs hgw_predictor

# Vérifier qu'il y a ≥60 lignes de données
docker exec -it hgw_timescaledb psql -U hgw_user -d hgw_monitoring -c "SELECT COUNT(*) FROM monitor_snapshots WHERE timestamp >= NOW() - INTERVAL '1 hour';"
```

### Grafana ne montre pas les panels

```bash
# Vérifier la datasource
docker exec -it hgw_grafana cat /etc/grafana/provisioning/datasources/timescaledb.yml

# Tester la connexion depuis Grafana UI
# Configuration → Data Sources → TimescaleDB → Test

# Recharger le dashboard
# Dashboards → Manage → HGW Monitoring → Settings → JSON Model → Copier-coller le JSON
```

---

## 📁 Structure des fichiers

```
hgw-predictor/
├── docker-compose-complete.yml    ← Orchestration
├── .env                           ← Config (HGW host, passwords)
│
├── collector/                     ← Monitoring
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── config.py
│   ├── telnet_client.py
│   ├── data_collection.py
│   ├── data_logger.py             ← Écriture TimescaleDB
│   └── event_manager.py
│
├── predictor/                     ← ML/DL
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── predict_service.py         ← Service principal
│   ├── predict_multi_horizon.py
│   ├── multi_horizon/             ← 4 modèles CatBoost
│   └── long_horizon_dl/           ← Bi-LSTM 3 jours
│
├── sql/
│   └── init.sql                   ← Schéma DB
│
└── grafana/
    ├── provisioning/
    │   └── datasources/
    │       └── timescaledb.yml
    └── dashboards/
        └── hgw_complete.json
```

---

## 🔄 Maintenance

### Backup de la base

```bash
# Backup complet
docker exec hgw_timescaledb pg_dump -U hgw_user hgw_monitoring > backup_$(date +%Y%m%d).sql

# Restore
docker exec -i hgw_timescaledb psql -U hgw_user hgw_monitoring < backup_20260502.sql
```

### Réentraînement hebdomadaire

```bash
# Export 7 derniers jours
docker exec hgw_timescaledb psql -U hgw_user -d hgw_monitoring -c "\COPY (SELECT * FROM monitor_snapshots WHERE timestamp >= NOW() - INTERVAL '7 days') TO STDOUT CSV HEADER" > weekly_batch.csv

# Réentraîner (hors Docker pour l'instant)
python train_multi_horizon.py --input-csv weekly_batch.csv

# Copier les nouveaux modèles
cp multi_horizon/*.cbm predictor/multi_horizon/

# Redémarrer predictor
docker-compose -f docker-compose-complete.yml restart predictor
```

### Nettoyage

```bash
# Arrêter tout
docker-compose -f docker-compose-complete.yml down

# Supprimer les volumes (⚠️ perte de données)
docker-compose -f docker-compose-complete.yml down -v

# Supprimer les images
docker rmi hgw-predictor_collector hgw-predictor_predictor
```

---

## 📊 Exemples de requêtes SQL

```sql
-- Derniers incidents URGENT
SELECT timestamp, STATUS_REASON, CPU_USAGE_PERCENT, MEM_USAGE_PERCENT
FROM monitor_snapshots
WHERE LOCAL_STATUS = 'URGENT'
ORDER BY timestamp DESC
LIMIT 10;

-- Alertes prédites (24h)
SELECT timestamp, horizon, probability, threshold
FROM predictions_log
WHERE alert = TRUE
  AND timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;

-- Statistiques horaires
SELECT * FROM hourly_stats
ORDER BY hour DESC
LIMIT 24;

-- Taux de détection ML
SELECT
  horizon,
  COUNT(*) AS total_predictions,
  SUM(CASE WHEN alert THEN 1 ELSE 0 END) AS alerts_triggered,
  AVG(probability) AS avg_probability
FROM predictions_log
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY horizon;
```

---

## 🎓 Pour ton PFE

### Démo soutenance

1. **Lancer le système** : `docker-compose up -d`
2. **Montrer Grafana** : Dashboard live avec métriques + prédictions
3. **Provoquer une charge CPU** sur le HGW
4. **Observer** : Prédiction ML détecte l'anomalie avant le seuil URGENT
5. **Expliquer** : "Le modèle 15min a prédit l'incident 5 minutes avant"

### Métriques à présenter

- **Collecte** : 1 snapshot toutes les 5s = 720/heure = 17 280/jour
- **Prédictions** : 1 prédiction toutes les 5min = 288/jour
- **Stockage** : ~500 KB/jour (compressé TimescaleDB)
- **Performance ML** : PR-AUC 0.979-0.996 (validé 5-fold CV)
- **Performance DL** : PR-AUC 0.962, Recall 99.1%

---

## 📞 Support

En cas de problème :
1. Vérifier les logs : `docker-compose logs -f`
2. Vérifier `.env` : credentials corrects
3. Tester connectivité HGW : `telnet 192.168.1.1 23`
4. Vérifier DB : `docker exec -it hgw_timescaledb psql -U hgw_user hgw_monitoring`

Bon courage pour ta soutenance ! 🎓🚀
