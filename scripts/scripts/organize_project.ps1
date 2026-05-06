# organize_project.ps1 - Réorganisation automatique du projet HGW (Windows PowerShell)

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Organisation automatique du projet HGW Predictor PFE" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Créer la structure de dossiers
Write-Host "[1/8] Création de la structure..." -ForegroundColor Yellow

$folders = @(
    "data",
    "collector",
    "predictor\multi_horizon",
    "predictor\long_horizon_dl",
    "notebooks",
    "sql",
    "grafana\provisioning\datasources",
    "grafana\provisioning\dashboards",
    "grafana\dashboards",
    "docs",
    "scripts"
)

foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder -Force | Out-Null
    }
}

Write-Host "✓ Structure créée" -ForegroundColor Green

# Déplacer les données
Write-Host "[2/8] Déplacement des fichiers de données..." -ForegroundColor Yellow

Get-ChildItem -Filter "*.csv" | ForEach-Object {
    Move-Item $_.FullName -Destination "data\" -Force
    Write-Host "  → $($_.Name) → data\" -ForegroundColor Gray
}

Write-Host "✓ Données organisées" -ForegroundColor Green

# Déplacer les scripts de prédiction
Write-Host "[3/8] Déplacement des scripts de prédiction..." -ForegroundColor Yellow

$predictorFiles = @(
    "predict_multi_horizon.py",
    "train_multi_horizon.py",
    "test_models.py",
    "predict_service.py"
)

foreach ($file in $predictorFiles) {
    if (Test-Path $file) {
        Move-Item $file -Destination "predictor\" -Force
        Write-Host "  → $file → predictor\" -ForegroundColor Gray
    }
}

Write-Host "✓ Scripts de prédiction organisés" -ForegroundColor Green

# Déplacer les modèles
Write-Host "[4/8] Organisation des modèles ML/DL..." -ForegroundColor Yellow

if (Test-Path "multi_horizon") {
    Move-Item "multi_horizon\*" -Destination "predictor\multi_horizon\" -Force
    Remove-Item "multi_horizon" -Recurse -Force
    Write-Host "  → multi_horizon\ → predictor\" -ForegroundColor Gray
}

if (Test-Path "long_horizon_dl") {
    Move-Item "long_horizon_dl\*" -Destination "predictor\long_horizon_dl\" -Force
    Remove-Item "long_horizon_dl" -Recurse -Force
    Write-Host "  → long_horizon_dl\ → predictor\" -ForegroundColor Gray
}

Write-Host "✓ Modèles organisés" -ForegroundColor Green

# Déplacer les notebooks
Write-Host "[5/8] Déplacement des notebooks..." -ForegroundColor Yellow

Get-ChildItem -Filter "*.ipynb" | ForEach-Object {
    Move-Item $_.FullName -Destination "notebooks\" -Force
    Write-Host "  → $($_.Name) → notebooks\" -ForegroundColor Gray
}

Write-Host "✓ Notebooks organisés" -ForegroundColor Green

# Copier requirements.txt
Write-Host "[6/8] Copie de requirements.txt..." -ForegroundColor Yellow

if (Test-Path "requirements.txt") {
    Copy-Item "requirements.txt" -Destination "collector\" -Force
    Copy-Item "requirements.txt" -Destination "predictor\" -Force
    Write-Host "  → requirements.txt copié" -ForegroundColor Gray
}

Write-Host "✓ Requirements copiés" -ForegroundColor Green

# Supprimer les fichiers obsolètes
Write-Host "[7/8] Nettoyage des fichiers obsolètes..." -ForegroundColor Yellow

$obsoleteFiles = @(
    "01_generate_datasets.py",
    "02_train_catboost_short.py",
    "03_train_bilstm_long.py",
    "03b_evaluate_bilstm.py",
    "03c_retrain_bilstm_safe.py",
    "04_drift_monitor.py",
    "05_predict_service.py",
    "predict_incident_prod.py",
    "predict_db.py"
)

foreach ($file in $obsoleteFiles) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  ✗ Supprimé : $file" -ForegroundColor Red
    }
}

Write-Host "✓ Nettoyage terminé" -ForegroundColor Green

# Renommer la documentation
Write-Host "[8/8] Organisation de la documentation..." -ForegroundColor Yellow

if ((Test-Path "PFE_FINAL_README.md") -and (-not (Test-Path "README.md"))) {
    Rename-Item "PFE_FINAL_README.md" -NewName "README.md"
    Write-Host "  → PFE_FINAL_README.md → README.md" -ForegroundColor Gray
}

# Créer .gitignore
if (-not (Test-Path ".gitignore")) {
    @"
# Python
__pycache__/
*.py[cod]
*.so
.Python
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
*.swp

# OS
.DS_Store
Thumbs.db

# Docker
.env
docker-compose.override.yml

# Logs
*.log
logs/
"@ | Out-File -FilePath ".gitignore" -Encoding utf8
    Write-Host "✓ .gitignore créé" -ForegroundColor Green
}

# Créer .env.example
if (-not (Test-Path ".env.example")) {
    @"
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
"@ | Out-File -FilePath ".env.example" -Encoding utf8
    Write-Host "✓ .env.example créé" -ForegroundColor Green
}

Write-Host "✓ Documentation organisée" -ForegroundColor Green

# Résumé
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  ✅ Organisation terminée avec succès !" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Prochaines étapes :" -ForegroundColor Yellow
Write-Host "1. Vérifier la structure : tree /F (ou dir /s)"
Write-Host "2. Copier .env.example → .env et configurer"
Write-Host "3. Lancer : docker-compose up -d"
Write-Host ""
