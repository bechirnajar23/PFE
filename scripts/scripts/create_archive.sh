#!/bin/bash
# create_project_archive.sh - Crée une archive ZIP complète du projet HGW

set -e

PROJECT_NAME="hgw-predictor"
ARCHIVE_NAME="${PROJECT_NAME}_$(date +%Y%m%d_%H%M%S).zip"

echo "════════════════════════════════════════════════════════════════"
echo "  Création de l'archive complète du projet HGW Predictor"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Créer un dossier temporaire pour l'archive
TEMP_DIR="/tmp/${PROJECT_NAME}"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo "[1/5] Copie des fichiers..."

# Copier tous les fichiers depuis /mnt/user-data/outputs
cp -r /mnt/user-data/outputs/* "$TEMP_DIR/" 2>/dev/null || true

# Organiser la structure
cd "$TEMP_DIR"

# Créer les dossiers manquants
mkdir -p data collector predictor/multi_horizon predictor/long_horizon_dl notebooks sql grafana/dashboards scripts

# Déplacer les fichiers aux bons endroits
echo "[2/5] Organisation de la structure..."

# Collector
[ -d collector ] || mkdir -p collector
mv data_logger.py collector/ 2>/dev/null || true

# Predictor
[ -f predict_multi_horizon.py ] && mv predict_multi_horizon.py predictor/ 2>/dev/null || true
[ -f train_multi_horizon.py ] && mv train_multi_horizon.py predictor/ 2>/dev/null || true
[ -f test_models.py ] && mv test_models.py predictor/ 2>/dev/null || true
[ -f predict_service.py ] && mv predict_service.py predictor/ 2>/dev/null || true

# SQL
[ -d sql ] || mkdir -p sql
[ -f init.sql ] && mv init.sql sql/ 2>/dev/null || true

# Grafana
[ -d grafana/dashboards ] || mkdir -p grafana/dashboards
[ -f hgw_complete.json ] && mv hgw_complete.json grafana/dashboards/ 2>/dev/null || true

# Scripts
mv organize_project*.sh scripts/ 2>/dev/null || true
mv organize_project.ps1 scripts/ 2>/dev/null || true

echo "[3/5] Création des fichiers de configuration..."

# Créer .env.example
cat > .env.example << 'EOF'
# HGW Connection
HGW_HOST=192.168.1.1
HGW_USER=root
HGW_PASSWORD=sah

# Collection
COLLECTION_INTERVAL=5

# Prediction  
PREDICTION_INTERVAL=300

# Database
POSTGRES_USER=hgw_user
POSTGRES_PASSWORD=hgw_secure_password
POSTGRES_DB=hgw_monitoring
EOF

# Créer .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
venv/
.venv/

# Data
*.csv
*.pkl

# Models
*.cbm
*.h5

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# Docker
.env
docker-compose.override.yml

# Logs
*.log
EOF

echo "[4/5] Création de l'archive ZIP..."

# Retour au dossier parent
cd /tmp

# Créer l'archive (exclure les fichiers inutiles)
zip -r "$ARCHIVE_NAME" "$PROJECT_NAME" \
    -x "*.pyc" \
    -x "*__pycache__*" \
    -x "*.git*" \
    -x "*.DS_Store" \
    -x "*Thumbs.db" \
    -q

# Copier dans outputs
cp "$ARCHIVE_NAME" /mnt/user-data/outputs/

echo "[5/5] Nettoyage..."
rm -rf "$TEMP_DIR"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Archive créée avec succès !"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📦 Fichier : /mnt/user-data/outputs/$ARCHIVE_NAME"
echo "📊 Taille  : $(du -h /mnt/user-data/outputs/$ARCHIVE_NAME | cut -f1)"
echo ""
echo "Pour extraire :"
echo "  unzip $ARCHIVE_NAME"
echo ""
