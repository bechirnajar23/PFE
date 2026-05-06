# Guide de test et validation HGW

Ce guide permet de valider le systeme cas par cas: provisioning Grafana, collecte, prediction continue, alertes SMS et reentrainement.

## 1. Preparer l'environnement

Depuis la racine du projet:

```bash
docker compose down
docker compose up --build -d
docker compose ps
```

Verifier les logs principaux:

```bash
docker compose logs -f timescaledb
docker compose logs -f collector
docker compose logs -f predictor
docker compose logs -f grafana
```

## 2. Valider Grafana sans import manuel

Ouvrir:

```text
http://localhost:3000
```

Identifiants par defaut:

```text
admin / admin
```

Resultat attendu:

- la datasource `TimescaleDB` existe automatiquement;
- le dashboard `HGW - Monitoring` existe automatiquement;
- le dashboard `HGW - Predictions` existe automatiquement;
- le lien `Onglet Predictions` permet de passer du monitoring vers les predictions;
- le lien `Onglet Monitoring` permet de revenir au monitoring.

Si les dashboards n'apparaissent pas, verifier:

```bash
docker compose logs grafana
```

Puis confirmer les fichiers montes:

```bash
docker exec -it hgw_grafana ls /etc/grafana/dashboards
docker exec -it hgw_grafana ls /etc/grafana/provisioning/datasources
docker exec -it hgw_grafana ls /etc/grafana/provisioning/dashboards
```

Si un panel affiche l'erreur Postgres `default database`, redemarrer Grafana pour recharger la datasource provisionnee:

```bash
docker compose restart grafana
docker compose logs -f grafana
```

La datasource doit utiliser:

```text
Host: timescaledb:5432
Database: hgw_monitoring
User: hgw_user
UID: timescaledb
```

## 3. Valider la base TimescaleDB

Lister les tables:

```bash
docker exec -it hgw_timescaledb psql -U hgw_user -d hgw_monitoring -c "\dt"
```

Si la base existait deja avant les derniers changements, appliquer la migration sans supprimer les donnees:

```bash
docker exec -i hgw_timescaledb psql -U hgw_user -d hgw_monitoring < sql/20_dashboard_migration.sql
```

Verifier les derniers snapshots:

```bash
docker exec -it hgw_timescaledb psql -U hgw_user -d hgw_monitoring -c "SELECT timestamp, local_status, status_reason, cpu_usage_percent, mem_usage_percent, alert_explanation FROM monitor_snapshots ORDER BY timestamp DESC LIMIT 5;"
```

Verifier les predictions:

```bash
docker exec -it hgw_timescaledb psql -U hgw_user -d hgw_monitoring -c "SELECT timestamp, horizon, probability, threshold, alert, decision_level FROM predictions_log ORDER BY timestamp DESC LIMIT 10;"
```

## 4. Cas 1 - Fonctionnement normal

Attendre quelques cycles de collecte.

Resultat attendu:

- `monitor_snapshots` recoit des lignes;
- `HGW - Monitoring` affiche CPU, memoire, latence et WAN;
- le statut reste `NORMAL` ou `WARNING`;
- aucune alerte SMS n'est envoyee si l'etat n'est pas `URGENT` ou `CRITICAL`.

## 5. Cas 2 - Prediction continue toutes les 5 minutes

La prediction tourne avec:

```env
PREDICTION_INTERVAL_SECONDS=300
```

Attendre 5 minutes, puis verifier:

```bash
docker compose logs predictor
docker exec -it hgw_timescaledb psql -U hgw_user -d hgw_monitoring -c "SELECT timestamp, horizon, probability, threshold, alert, decision_level FROM predictions_log ORDER BY timestamp DESC LIMIT 10;"
```

Resultat attendu:

- une nouvelle serie de predictions est inseree environ toutes les 5 minutes;
- les horizons court terme et 3 jours apparaissent dans `HGW - Predictions`.

Pour un test rapide, modifier temporairement `.env`:

```env
PREDICTION_INTERVAL_SECONDS=30
```

Puis redemarrer seulement le predictor:

```bash
docker compose up -d --build predictor
```

Remettre ensuite `300`.

## 6. Cas 3 - Alerte URGENT ou CRITICAL

L'alerte Email/SMS est volontairement limitee aux etats:

```text
URGENT
CRITICAL
```

Inserer un snapshot critique de test:

```bash
docker exec -it hgw_timescaledb psql -U hgw_user -d hgw_monitoring -c "INSERT INTO monitor_snapshots (timestamp, local_status, status_reason, cpu_usage_percent, mem_usage_percent, wan_state, net_latency_ms, dhcp_process_status, net_ping_status, alert_eligible, alert_explanation) VALUES (NOW(), 'CRITICAL', 'test_memory_pressure', 92, 96, 'UP', 180, 'OK', 'OK', true, 'Memoire critique et CPU eleve - scenario de test');"
```

Attendre le prochain cycle de prediction.

Resultat attendu:

- `HGW - Monitoring` affiche l'evenement dans `Evenements URGENT / CRITICAL`;
- `HGW - Predictions` affiche l'explication dans `Pourquoi l'alerte est declenchee`;
- le service SMS tente l'envoi si Twilio est configure.

## 7. Cas 4 - Tester l'API SMS

Verifier que le service repond:

```bash
curl http://localhost:5000/health
```

Tester un envoi:

```bash
curl -X POST http://localhost:5000/sms-alert \
  -H "Content-Type: application/json" \
  -d "{\"level\":\"CRITICAL\",\"message\":\"Test alerte HGW\",\"explanation\":\"Memoire critique et CPU eleve\",\"phone_to\":\"+21629860834\"}"
```

Pour un vrai SMS, remplir dans `.env`:

```env
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_NUMBER=...
ALERT_PHONE_TO=+21629860834
```

Puis redemarrer:

```bash
docker compose up -d --build sms predictor
```

## 8. Cas 5 - Reentrainement tous les 7 jours

La configuration normale est:

```env
RETRAIN_ENABLED=true
RETRAIN_INTERVAL_DAYS=7
RETRAIN_RUN_ON_START=false
```

Pour tester sans attendre 7 jours, modifier temporairement:

```env
RETRAIN_RUN_ON_START=true
```

Puis:

```bash
docker compose up -d --build predictor
docker compose logs -f predictor
```

Resultat attendu:

- le predictor lance la commande de reentrainement;
- l'etat est sauvegarde dans `data/retrain_state.json`.

Remettre ensuite:

```env
RETRAIN_RUN_ON_START=false
```

## 9. Validation finale pour soutenance

Checklist:

- collecte active dans `monitor_snapshots`;
- predictions actives dans `predictions_log`;
- dashboards provisionnes sans import manuel;
- deux vues separees: monitoring et predictions;
- explication d'alerte visible;
- SMS teste via `/sms-alert`;
- alerte envoyee uniquement si l'etat courant est `URGENT` ou `CRITICAL`;
- reentrainement configure tous les 7 jours.
