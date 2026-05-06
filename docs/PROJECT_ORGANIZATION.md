# 📂 Structure finale du projet HGW Predictor

## 🎯 Organisation recommandée

```
hgw-predictor/                           ← Dossier racine du projet
│
├── 📄 README.md                          ← Documentation principale
├── 📄 DEPLOYMENT_GUIDE.md                ← Guide déploiement Docker
├── 📄 DIAGRAMS.md                        ← Diagrammes UML PlantUML
├── 📄 docker-compose.yml                 ← Orchestration Docker
├── 📄 .env.example                       ← Template configuration
├── 📄 .gitignore                         ← Fichiers à ignorer Git
├── 📄 LICENSE                            ← Licence (optionnel)
│
├── 📂 data/                              ← 💾 DONNÉES
│   ├── monitor_snapshots.csv            ← Données réelles HGW
│   ├── real_hgw_preprocessed.csv        ← Dataset prétraité
│   └── hgw_5yr_bigdata.csv             ← Dataset synthétique (si présent)
│
├── 📂 collector/                         ← 📡 MONITORING TELNET
│   ├── 📄 Dockerfile
│   ├── 📄 requirements.txt
│   ├── 🐍 app.py                        ← Flask API (optionnel)
│   ├── 🐍 config.py                     ← Configuration HGW
│   ├── 🐍 telnet_client.py              ← Client Telnet
│   ├── 🐍 data_collection.py            ← Collecte métriques
│   ├── 🐍 data_logger.py                ← Logger TimescaleDB
│   └── 🐍 event_manager.py              ← Gestion événements
│
├── 📂 predictor/                         ← 🤖 ML/DL ENGINE
│   ├── 📄 Dockerfile
│   ├── 📄 requirements.txt
│   ├── 🐍 predict_service.py            ← Service prédiction (daemon)
│   ├── 🐍 predict_multi_horizon.py      ← Predictor unifié
│   ├── 🐍 train_multi_horizon.py        ← Entraînement ML
│   ├── 🐍 test_models.py                ← Tests interactifs
│   │
│   ├── 📂 multi_horizon/                ← Modèles ML (CatBoost)
│   │   ├── ⚙️ catboost_15min_real.cbm
│   │   ├── ⚙️ catboost_30min_real.cbm
│   │   ├── ⚙️ catboost_60min_real.cbm
│   │   ├── ⚙️ catboost_360min_real.cbm
│   │   └── 📄 multi_horizon_bundle.json
│   │
│   └── 📂 long_horizon_dl/              ← Modèle DL (Bi-LSTM)
│       ├── ⚙️ bilstm_3d.weights.h5
│       ├── 🐍 bilstm_loader.py
│       ├── 📄 bilstm_3d_metadata.json
│       ├── 📦 transfer_scaler.pkl
│       └── 🐍 train_bilstm_3d_final.py
│
├── 📂 notebooks/                         ← 📊 JUPYTER NOTEBOOKS
│   └── 06_real_data_pipeline.ipynb      ← EDA + entraînement
│
├── 📂 sql/                               ← 🗄️ SQL SCRIPTS
│   ├── init.sql                         ← Initialisation TimescaleDB
│   └── queries.sql                      ← Requêtes utiles (optionnel)
│
├── 📂 grafana/                           ← 📈 GRAFANA CONFIG
│   ├── 📂 provisioning/
│   │   ├── 📂 datasources/
│   │   │   └── timescaledb.yml
│   │   └── 📂 dashboards/
│   │       └── dashboard.yml
│   └── 📂 dashboards/
│       └── hgw_complete.json            ← Dashboard ML/DL
│
├── 📂 docs/                              ← 📚 DOCUMENTATION
│   ├── API.md                           ← Documentation API
│   ├── ARCHITECTURE.md                  ← Architecture détaillée
│   ├── MODEL_CARDS.md                   ← Fiches modèles
│   └── TROUBLESHOOTING.md               ← Résolution problèmes
│
└── 📂 scripts/                           ← 🔧 SCRIPTS UTILITAIRES
    ├── organize_project.sh              ← Script d'organisation
    ├── weekly_retrain.py                ← Réentraînement hebdo
    ├── export_data.py                   ← Export DB → CSV
    └── backup_db.sh                     ← Backup TimescaleDB
```

---

## ✅ Fichiers à GARDER

### 📄 Documentation (racine)
- ✅ `README.md` OU `PFE_FINAL_README.md` → Renommer en `README.md`
- ✅ `DEPLOYMENT_GUIDE.md`
- ✅ `DIAGRAMS.md`

### 🐍 Scripts de prédiction (→ `predictor/`)
- ✅ `predict_multi_horizon.py` — **CORE PRODUCTION**
- ✅ `train_multi_horizon.py` — **RÉENTRAÎNEMENT**
- ✅ `test_models.py` — **TESTS/DÉMOS**
- ✅ `predict_service.py` — **DAEMON**

### 💾 Données (→ `data/`)
- ✅ `monitor_snapshots.csv`
- ✅ Autres CSV si présents

### ⚙️ Modèles (déjà dans dossiers)
- ✅ `multi_horizon/` — 4 modèles CatBoost + bundle.json
- ✅ `long_horizon_dl/` — Bi-LSTM + scaler + loader

### 📦 Configuration
- ✅ `requirements.txt` (dupliquer dans `collector/` et `predictor/`)

---

## ❌ Fichiers à SUPPRIMER

```bash
# Anciens scripts obsolètes
❌ 01_generate_datasets.py
❌ 02_train_catboost_short.py
❌ 03_train_bilstm_long.py
❌ 03b_evaluate_bilstm.py
❌ 03c_retrain_bilstm_safe.py

# Anciens predictors (si différents de predict_multi_horizon.py)
❌ predict_incident_prod.py
❌ predict_db.py
❌ 05_predict_service.py (si doublon)
```

---

## 🚀 Comment organiser automatiquement

### Option 1 : Script automatique (Windows/Linux)

Télécharge `organize_project.sh` (fourni ci-dessous) et exécute :

**Linux/Mac :**
```bash
chmod +x organize_project.sh
./organize_project.sh
```

**Windows (PowerShell) :**
```powershell
# Convertir le script en PowerShell ou utiliser Git Bash
bash organize_project.sh
```

### Option 2 : Manuel (étape par étape)

```bash
# 1. Créer la structure
mkdir -p data collector predictor/multi_horizon predictor/long_horizon_dl notebooks sql grafana/dashboards docs scripts

# 2. Déplacer les données
mv monitor_snapshots.csv data/
mv *.csv data/ 2>/dev/null || true

# 3. Déplacer les scripts de prédiction
mv predict_multi_horizon.py predictor/
mv train_multi_horizon.py predictor/
mv test_models.py predictor/
mv predict_service.py predictor/

# 4. Déplacer les notebooks
mv *.ipynb notebooks/ 2>/dev/null || true

# 5. Copier requirements dans les sous-dossiers
cp requirements.txt collector/
cp requirements.txt predictor/

# 6. Supprimer les fichiers obsolètes
rm -f 01_generate_datasets.py 02_train_catboost_short.py 03_*.py

# 7. Renommer README si nécessaire
[ -f PFE_FINAL_README.md ] && mv PFE_FINAL_README.md README.md
```

---

## 📋 Checklist post-organisation

- [ ] Structure de dossiers créée
- [ ] Fichiers déplacés dans les bons dossiers
- [ ] `README.md` à la racine
- [ ] `docker-compose.yml` à la racine
- [ ] Modèles dans `predictor/multi_horizon/` et `predictor/long_horizon_dl/`
- [ ] `.env.example` créé
- [ ] `.gitignore` créé
- [ ] Fichiers obsolètes supprimés
- [ ] Tests : `tree -L 2` ou `ls -R` pour vérifier

---

## 🎓 Pour ton PFE

### Structure minimale pour soutenance

Si tu manques de temps, garde au minimum :

```
hgw-predictor/
├── README.md                     ← Documentation
├── docker-compose.yml            ← Déploiement
├── data/monitor_snapshots.csv    ← Données réelles
├── predictor/
│   ├── predict_multi_horizon.py
│   ├── train_multi_horizon.py
│   ├── test_models.py
│   ├── multi_horizon/            ← 4 modèles ML
│   └── long_horizon_dl/          ← Modèle DL
└── notebooks/
    └── 06_real_data_pipeline.ipynb
```

### Démo rapide

```bash
# Tester les modèles sur données réelles
cd predictor
python test_models.py ../data/monitor_snapshots.csv --urgents 5

# Montrer le notebook
jupyter notebook ../notebooks/06_real_data_pipeline.ipynb
```

---

## 📞 Aide rapide

**Vérifier la structure :**
```bash
tree -L 3 -I '__pycache__|*.pyc'
```

**Compter les fichiers :**
```bash
find . -type f -name "*.py" | wc -l
find . -type f -name "*.cbm" | wc -l
```

**Taille totale :**
```bash
du -sh .
du -sh data/
du -sh predictor/
```

Bon courage ! 🚀
