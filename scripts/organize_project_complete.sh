#!/bin/bash
# organize_project_complete.sh - Réorganisation automatique complète du projet HGW

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  🚀 Organisation automatique du projet HGW Predictor PFE"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================================
# ÉTAPE 1 : Créer la structure de dossiers
# ============================================================================
echo -e "${YELLOW}[1/8]${NC} Création de la structure de dossiers..."

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

echo -e "${GREEN}✓${NC} Structure créée"

# ============================================================================
# ÉTAPE 2 : Déplacer les données
# ============================================================================
echo -e "${YELLOW}[2/8]${NC} Déplacement des fichiers de données..."

# Déplacer tous les CSV vers data/ (sauf s'ils sont déjà dedans)
for file in *.csv; do
    [ -f "$file" ] && mv "$file" data/ && echo "  → $file → data/"
done 2>/dev/null || true

echo -e "${GREEN}✓${NC} Données organisées"

# ============================================================================
# ÉTAPE 3 : Déplacer les scripts de prédiction
# ============================================================================
echo -e "${YELLOW}[3/8]${NC} Déplacement des scripts de prédiction..."

# Core scripts
[ -f predict_multi_horizon.py ] && mv predict_multi_horizon.py predictor/ && echo "  → predict_multi_horizon.py → predictor/"
[ -f train_multi_horizon.py ] && mv train_multi_horizon.py predictor/ && echo "  → train_multi_horizon.py → predictor/"
[ -f test_models.py ] && mv test_models.py predictor/ && echo "  → test_models.py → predictor/"
[ -f predict_service.py ] && mv predict_service.py predictor/ && echo "  → predict_service.py → predictor/"

echo -e "${GREEN}✓${NC} Scripts de prédiction organisés"

# ============================================================================
# ÉTAPE 4 : Déplacer les modèles
# ============================================================================
echo -e "${YELLOW}[4/8]${NC} Organisation des modèles ML/DL..."

# Modèles ML (si dans racine)
if [ -d multi_horizon ]; then
    if [ ! -f predictor/multi_horizon/catboost_15min_real.cbm ]; then
        mv multi_horizon/* predictor/multi_horizon/ 2>/dev/null && echo "  → multi_horizon/ → predictor/"
        rmdir multi_horizon 2>/dev/null || true
    fi
fi

# Modèles DL (si dans racine)
if [ -d long_horizon_dl ]; then
    if [ ! -f predictor/long_horizon_dl/bilstm_3d.weights.h5 ]; then
        mv long_horizon_dl/* predictor/long_horizon_dl/ 2>/dev/null && echo "  → long_horizon_dl/ → predictor/"
        rmdir long_horizon_dl 2>/dev/null || true
    fi
fi

echo -e "${GREEN}✓${NC} Modèles organisés"

# ============================================================================
# ÉTAPE 5 : Déplacer les notebooks
# ============================================================================
echo -e "${YELLOW}[5/8]${NC} Déplacement des notebooks..."

for file in *.ipynb; do
    [ -f "$file" ] && mv "$file" notebooks/ && echo "  → $file → notebooks/"
done 2>/dev/null || true

echo -e "${GREEN}✓${NC} Notebooks organisés"

# ============================================================================
# ÉTAPE 6 : Copier requirements.txt
# ============================================================================
echo -e "${YELLOW}[6/8]${NC} Copie de requirements.txt..."

if [ -f requirements.txt ]; then
    cp requirements.txt collector/ 2>/dev/null && echo "  → requirements.txt → collector/"
    cp requirements.txt predictor/ 2>/dev/null && echo "  → requirements.txt → predictor/"
fi

echo -e "${GREEN}✓${NC} Requirements copiés"

# ============================================================================
# ÉTAPE 7 : Supprimer les fichiers obsolètes
# ============================================================================
echo -e "${YELLOW}[7/8]${NC} Nettoyage des fichiers obsolètes..."

OBSOLETE_FILES=(
    "01_generate_datasets.py"
    "02_train_catboost_short.py"
    "03_train_bilstm_long.py"
    "03b_evaluate_bilstm.py"
    "03c_retrain_bilstm_safe.py"
    "04_drift_monitor.py"
    "05_predict_service.py"
    "predict_incident_prod.py"
    "predict_db.py"
    "train_benchmark.py"
    "train_lstm_long_horizon.py"
)

for file in "${OBSOLETE_FILES[@]}"; do
    if [ -f "$file" ]; then
        rm "$file" && echo -e "  ${RED}✗${NC} Supprimé : $file"
    fi
done

echo -e "${GREEN}✓${NC} Nettoyage terminé"

# ============================================================================
# ÉTAPE 8 : Renommer la documentation
# ============================================================================
echo -e "${YELLOW}[8/8]${NC} Organisation de la documentation..."

# Renommer PFE_FINAL_README.md en README.md
if [ -f PFE_FINAL_README.md ] && [ ! -f README.md ]; then
    mv PFE_FINAL_README.md README.md && echo "  → PFE_FINAL_README.md → README.md"
elif [ -f PFE_FINAL_README.md ] && [ -f README.md ]; then
    echo "  ⚠️  README.md existe déjà, PFE_FINAL_README.md conservé"
fi

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
ENV/

# Data (garder seulement les exemples)
data/*.csv
!data/example_*.csv
*.pkl
!transfer_scaler.pkl

# Models (sauf metadata)
predictor/multi_horizon/*.cbm
predictor/long_horizon_dl/*.h5
predictor/long_horizon_dl/*.keras

# Jupyter
.ipynb_checkpoints/
notebooks/.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Docker
.env
docker-compose.override.yml

# Logs
*.log
logs/
/tmp/

# Build
dist/
build/
*.egg-info/
EOF
    echo -e "${GREEN}✓${NC} .gitignore créé"
fi

# Créer .env.example si absent
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
    echo -e "${GREEN}✓${NC} .env.example créé"
fi

echo -e "${GREEN}✓${NC} Documentation organisée"

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}✅ Organisation terminée avec succès !${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📂 Structure finale :"
echo ""

# Afficher la structure (si tree disponible)
if command -v tree &> /dev/null; then
    tree -L 2 -I '__pycache__|*.pyc|.git' --dirsfirst
else
    # Fallback si tree n'est pas installé
    find . -maxdepth 2 -type d | sed 's|^\./||' | grep -v '^\.$' | sort
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  📋 Prochaines étapes :"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "1. Vérifier la structure :"
echo "   tree -L 3"
echo ""
echo "2. Configurer l'environnement :"
echo "   cp .env.example .env"
echo "   nano .env  # Éditer avec tes credentials HGW"
echo ""
echo "3. Lancer le système :"
echo "   docker-compose up -d"
echo ""
echo "4. Accéder à Grafana :"
echo "   http://localhost:3000 (admin/admin)"
echo ""
echo "════════════════════════════════════════════════════════════════"

