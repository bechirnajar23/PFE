# Diagrammes UML actuels

Ce dossier contient les diagrammes PlantUML alignes avec l'architecture actuelle du projet HGW Predictive Maintenance.

## Fonctionnement retenu

- Le PC/serveur ecoute les metriques HGW sur un port dedie.
- La HGW envoie les metriques en temps reel.
- TimescaleDB stocke les snapshots et les predictions.
- Le service de prediction s'execute toutes les 5 minutes, 24h/24.
- Les alertes sont envoyees par email ou par SMS.
- Les modeles sont reentraines automatiquement tous les 7 jours.

## Fichiers

- `hgw_usecase.plantuml`: cas d'utilisation principaux.
- `hgw_sequence_collecte_stockage.plantuml`: collecte temps reel et stockage.
- `hgw_sequence_prediction_modeles.plantuml`: prediction continue, alertes et reentrainement 7 jours.
- `hgw_activity_flux_complet.plantuml`: flux complet du monitoring.

## Generation

Avec l'extension PlantUML de VS Code, ouvrir un fichier `.plantuml`, puis utiliser l'apercu ou l'export PNG/SVG.

Avec PlantUML CLI:

```bash
plantuml DIAGRAMS/hgw_usecase.plantuml
plantuml DIAGRAMS/hgw_sequence_collecte_stockage.plantuml
plantuml DIAGRAMS/hgw_sequence_prediction_modeles.plantuml
plantuml DIAGRAMS/hgw_activity_flux_complet.plantuml
```
