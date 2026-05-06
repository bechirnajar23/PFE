# Architecture technique et diagrammes UML - HGW Predictive Maintenance

Ce document decrit l'architecture de haut niveau de la solution HGW Predictive Maintenance et fournit les diagrammes UML PlantUML associes.

## 1. Vue d'ensemble

Le projet est une solution de monitoring et de prediction technique pour les Home Gateways (HGW). Il combine collecte temps reel, stockage temporel, analyse ML/DL et visualisation operationnelle.

Objectif principal:

```text
Detecter les signes faibles d'incident, predire les anomalies avant la panne,
et afficher l'etat du reseau en temps reel.
```

## 2. Architecture globale

Le flux principal suit quatre couches:

```text
Ingestion -> Stockage -> Analyse ML/DL -> Visualisation / Alerte
```

### Ingestion

La couche d'ingestion est portee par les collecteurs Python du dossier `collector/`. Dans la conception haut niveau, le PC/serveur ouvre un port d'ecoute dedie a la telemetrie. La HGW envoie ses metriques vers ce port, puis le collecteur receptionne, normalise et prepare les donnees avant stockage.

Donnees collectees:

- CPU: usage global, user, system, idle.
- Memoire: total, libre, utilisee, pourcentage d'utilisation.
- Reseau: etat WAN, debit RX/TX, latence ping, statut ping.
- Services: DHCP process, etats DHCP data et DHCPv6.
- Etat local: `NORMAL`, `WARNING`, `URGENT` avec raison associee.

Le collecteur calcule aussi un statut metier local avant stockage. Par exemple, CPU eleve, memoire critique, WAN non disponible ou DHCP arrete. Cette decision est accompagnee d'une explication structuree qui joue le role d'explainer metier: elle indique quelle regle a declenche l'etat et quelle metrique est responsable.

### Stockage

La couche de stockage repose sur TimescaleDB, base PostgreSQL optimisee pour les series temporelles.

Tables principales:

- `monitor_snapshots`: mesures brutes et etat local de la HGW.
- `predictions_log`: probabilites et decisions produites par les modeles.
- `alerts`: journal optionnel des alertes envoyees.
- `hourly_stats`: vue agregee pour le dashboard.

Le lien principal entre les donnees est temporel:

```text
timestamp + gateway_id
```

Le modele est donc principalement time-series: on relie les mesures, predictions et incidents par la gateway et par le temps, plutot que par un modele relationnel metier complexe.

### Analyse ML/DL

La couche d'analyse est portee par `predictor/`.

Elle fonctionne en continu, 24h/24. Toutes les 5 minutes, elle charge les dernieres mesures depuis TimescaleDB, reconstruit les features necessaires, puis execute les modeles:

- CatBoost court terme: prediction a 15 min, 30 min, 1 h et 6 h.
- LSTM long terme: prediction a 3 jours.

Les predictions sont comparees aux seuils de decision. Si une probabilite depasse son seuil, le systeme marque une alerte predictive dans la base. En parallele, les regles metier detectent aussi les etats critiques actuels comme CPU/RAM critiques ou WAN down.

L'explication de l'alerte combine deux niveaux:

- Regles metier instant t: cause directe comme `high_cpu`, `high_memory`, `dhcp_process_stopped`, `wan_not_up`.
- Explainer ML: les modeles CatBoost peuvent etre expliques par SHAP ou par importance des features pour justifier les predictions.

Le service de prediction porte aussi une planification simple de reentrainement: tous les 7 jours, il peut declencher la commande d'entrainement configuree, sauvegarder les nouveaux artefacts et recharger les modeles actifs sans dependance a un orchestrateur externe.

### Visualisation et alertes

Grafana lit les donnees depuis TimescaleDB et affiche:

- etat global de la HGW;
- health score;
- CPU, RAM, latence, packet loss;
- predictions multi-horizon;
- alertes actives;
- evenements critiques.

Le service de notification optionnel peut envoyer des alertes aux utilisateurs par SMS ou par email. Dans l'implementation actuelle, `predictor/sms_service.py` expose une API Flask pour l'envoi SMS via Twilio, et le service de prediction peut aussi envoyer un email via SMTP.

Regle importante:

```text
Une alerte utilisateur est envoyee uniquement si l'etat courant est URGENT ou CRITICAL.
Les etats NORMAL/WARNING et les predictions seules restent visibles dans Grafana, sans SMS/email utilisateur.
```

## 3. Workflow fonctionnel

1. Le service de collecte ouvre un port d'ecoute sur le PC/serveur.
2. La HGW transmet ses metriques vers ce port.
3. Le collecteur receptionne les donnees et construit un snapshot normalise.
4. Il calcule le statut local: `NORMAL`, `WARNING`, `URGENT` ou `CRITICAL`.
5. Le snapshot est insere dans TimescaleDB.
6. Toutes les 5 minutes, le service de prediction lit les dernieres mesures.
7. Les features ML/DL sont construites.
8. Les modeles CatBoost et LSTM produisent des probabilites d'incident.
9. Les predictions et les explications sont stockees dans TimescaleDB.
10. Grafana affiche les mesures et predictions.
11. Si l'etat courant est `URGENT` ou `CRITICAL`, une alerte est envoyee par email ou par SMS avec la cause.
12. Tous les 7 jours, le service declenche un reentrainement des modeles et recharge les artefacts.

## 4. Diagrammes PlantUML

Les diagrammes sont fournis dans le dossier `DIAGRAMS/`:

- `hgw_usecase.plantuml`: diagramme de cas d'utilisation.
- `hgw_sequence_collecte_stockage.plantuml`: sequence de collecte et stockage des logs.
- `hgw_sequence_prediction_modeles.plantuml`: sequence de prediction par les modeles.
- `hgw_activity_flux_complet.plantuml`: activite globale du log brut vers affichage/alerte.

## 5. Lecture architecturale pour le rapport

Phrase courte utilisable dans le rapport:

```text
L'architecture adoptee suit une chaine data orientee series temporelles. Les collecteurs Python assurent l'ingestion depuis la HGW, TimescaleDB centralise les mesures et predictions, les modeles ML/DL estiment le risque d'incident toutes les 5 minutes sur plusieurs horizons, puis Grafana et les notifications email/SMS exposent les resultats aux utilisateurs. Les modeles sont reentraines automatiquement tous les 7 jours.
```

Justification du choix TimescaleDB:

```text
TimescaleDB est adapte car les donnees sont principalement temporelles: chaque mesure est liee a un timestamp et a une gateway. Cette base conserve la robustesse SQL tout en ajoutant des optimisations time-series utiles pour le monitoring, les agregations et les dashboards Grafana.
```
