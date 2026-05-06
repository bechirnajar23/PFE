#!/bin/bash
# organize_project.sh - Réorganise automatiquement le projet PFE HGW

set -e

echo "════════════════════════════════════════════════════════════"
echo "  Organisation automatique du projet HGW Predictor"
echo "════════════════════════════════════════════════════════════"

# Créer la structure de dossiers
echo "[1/6] Création de la structure..."
mkdir -p data
mkdir -p collector
mkdir -p predictor/multi_horizon
mkdir -p predictor/long_horizon_dl
mkdir -p notebooks
mkdir -p sql
mkdir -p grafana/provisioning/datasources
mkdir -p grafana/provisioning/dashboards
mkdir -p grafana/dashboards
mkdir -p docs
mkdir -p scripts

# Déplacer les fichiers de données
echo "[2/6] Organisation des données..."
[ -f monitor_snapshots.csv ] && mv monitor_snapshots.csv data/
[ -f real_hgw_preprocessed.csv ] && mv real_hgw_preprocessed.csv data/
[ -f hgw_5yr_bigdata.csv ] && mv hgw_5yr_bigdata.csv data/ 2>/dev/null || true

# Déplacer les fichiers de prédiction
echo "[3/6] Organisation du predictor..."
[ -f predict_multi_horizon.py ] && mv predict_multi_horizon.py predictor/
[ -f train_multi_horizon.py ] && mv train_multi_horizon.py predictor/
[ -f test_models.py ] && mv test_models.py predictor/
[ -f predict_service.py ] && mv predict_service.py predictor/

# Déplacer les modèles (si dans le dossier racine)
if [ -d multi_horizon ] && [ ! -d predictor/multi_horizon/catboost_15min_real.cbm ]; then
    mv multi_horizon/* predictor/multi_horizon/ 2>/dev/null || true
    rmdir multi_horizon 2>/dev/null || true
fi

if [ -d long_horizon_dl ] && [ ! -f predictor/long_horizon_dl/bilstm_3d.weights.h5 ]; then
    mv long_horizon_dl/* predictor/long_horizon_dl/ 2>/dev/null || true
    rmdir long_horizon_dl 2>/dev/null || true
fi

# Déplacer les notebooks
echo "[4/6] Organisation des notebooks..."
[ -f 06_real_data_pipeline.ipynb ] && mv 06_real_data_pipeline.ipynb notebooks/

# Supprimer les fichiers obsolètes
echo "[5/6] Nettoyage des fichiers obsolètes..."
rm -f 01_generate_datasets.py 2>/dev/null || true
rm -f 02_train_catboost_short.py 2>/dev/null || true
rm -f 03_train_bilstm_long.py 2>/dev/null || true
rm -f 03b_evaluate_bilstm.py 2>/dev/null || true
rm -f predict_incident_prod.py 2>/dev/null || true
rm -f predict_db.py 2>/dev/null || true

# Renommer les fichiers de documentation
echo "[6/6] Organisation de la documentation..."
[ -f PFE_FINAL_README.md ] && mv PFE_FINAL_README.md README.md

# Créer .gitignore si absent
if [ ! -f .gitignore ]; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Data
*.csv
*.pkl
!multi_horizon_bundle.json
!bilstm_3d_metadata.json

# Models (sauf bundle/metadata)
*.cbm
*.h5
*.keras

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Docker
.env
docker-compose.override.yml

# Logs
*.log
logs/
EOF
fi

# Créer .env.example
if [ ! -f .env.example ]; then
    cat > .env.example << 'EOF'
# HGW Connection
HGW_HOST=192.168.1.1
HGW_USER=root
HGW_PASSWORD=your_password_here

# Collection
COLLECTION_INTERVAL=5

# Prediction
PREDICTION_INTERVAL=300

# Database
POSTGRES_USER=hgw_user
POSTGRES_PASSWORD=hgw_secure_password
POSTGRES_DB=hgw_monitoring
EOF
fi

echo ""
echo "✅ Organisation terminée !"
echo ""
echo "Structure finale :"
tree -L 2 -I '__pycache__|*.pyc|.git' . || ls -R

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Prochaines étapes :"
echo "════════════════════════════════════════════════════════════"
echo "1. Vérifier la structure : tree -L 3"
echo "2. Copier .env.example → .env et configurer"
echo "3. Lancer : docker-compose up -d"
echo "════════════════════════════════════════════════════════════"
