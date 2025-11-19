# lightML – Génération de presets Lightroom assistée par ML

## Contexte

Ce dépôt explore la génération automatique de presets Lightroom pour le dataset **MMArt-PPR10k**.  
Il s'agit de :
1. Charger la dataset locale (images + requêtes utilisateur) et créer des objets `Sample`.
2. Clustériser les requêtes (SentenceTransformers + KMeans) pour obtenir des styles/genres.
3. Construire un jeu de features par image (stats globales + locales, clipping, histos).
4. Apprendre des offsets de sliders ROC (Exposure, Shadows, etc.) via XGBoost.
5. Générer un preset `.xmp` applicables directement dans Lightroom via l'interface Streamlit (`prod.py`).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prérequis :
- Dataset `MMArt-PPR10k` disponible localement sous `data/MMArt-PPR10k`.

## Étapes principales

### 1. Préparation des données

```bash
python data.py
```
- Crée les clusters (KMeans) et exporte leurs stats / descriptions (`cluster_summary.xlsx`).
- Sauvegarde les features/targets brutes dans `data/processed/{features,targets}.csv`.

### 2. Entraînement des modèles

```bash
python train_model.py
```
- Tuning Optuna pour chaque target (`Exposure2012`, `Shadows2012`, etc.).
- Sauvegarde des modèles XGBoost dans `models/`.

### 3. Génération de preset via Streamlit

```bash
streamlit run prod.py
```
- Dépose une image, choisis un cluster/style, lance le calcul → un `.xmp` est écrit dans `presets/`.
- Le preset porte un UUID/nom unique compatible avec Lightroom.

## Structure du projet

- `tools.py` : features image, lecture/écriture XMP.
- `core/` : classes (`Sample`, `Roc`, `Cluster`).
- `data.py` : pipeline de clustering + export CSV.
- `train_model.py` / `train_test.py` : entraînement global ou expérimentation ciblée.
- `prod.py` : interface Streamlit d'inférence + export preset.
- `presets/` : exemples/génération de `.xmp`.
- `logs/` : courbes d'apprentissage et résumés RMSE.