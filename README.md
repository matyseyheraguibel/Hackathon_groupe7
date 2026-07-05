# Spot Forecast Project

Prévision des prix spot de l'électricité (marché finlandais) et évaluation de la
stratégie d'arbitrage batterie (2 MWh / 1 MW) qui en découle.

Contexte complet du sujet (entreprise NW, objectifs, critère de réussite) : voir
[`PROJECT.md`](./PROJECT.md).

---

## 1. Installation

**Prérequis :** Python 3.10+

```bash
# Cloner le repo
git clone <url-du-repo>
cd <nom-du-repo>

# (recommandé) créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

Aucune donnée externe à télécharger : les jeux de données bruts et traités sont
déjà versionnés dans `data/raw/` et `data/processed/` (voir `PROJECT.md` pour le
détail des sources).

---

## 2. Structure du projet

```
config/                 configuration (vide pour l'instant)
data/
  raw/                   données brutes (prix spot, prévisions de production,
                         vent, gaz, etc.)
  processed/             X.csv / y.csv / scaler_params.csv déjà générés,
                         + les graphiques d'évaluation par modèle
experiment/              scripts d'entraînement "prêts à lancer" pour chaque
                         modèle (un run.py par modèle)
source/
  data_processing/       pipeline de nettoyage + construction de X, y
  price_models/          modèles de prévision (interface commune fit/predict)
  critic/                réseau "critique" (loss économique custom)
  utils/                 métriques de performance, visualisation, dataset
weights/                 poids de modèles entraînés (si sauvegardés)
spot_forecast_project.ipynb   notebook de présentation du projet
gridsearch1.ipynb             notebook de recherche d'hyperparamètres
```

---

## 3. Utiliser le projet

### a) Régénérer les données (optionnel — X.csv / y.csv sont déjà fournis)

```bash
cd source/data_processing
python preprocessing.py
```

Ce script lit `data/raw/data_raw.csv` (+ prévisions de production, vent, etc.),
applique le pipeline de nettoyage (`add_times` → `add_prediction_production` →
`add_wind` → `corriger_doublons` → `lisser_en_horaire` → standardisations) puis
construit et sauvegarde `data/processed/X.csv`, `y.csv` et `scaler_params.csv`.

### b) Entraîner un modèle et évaluer sa performance

Chaque script du dossier `experiment/` charge `X.csv` / `y.csv`, entraîne un
modèle, puis affiche les métriques et sauvegarde un graphique d'évaluation dans
`data/processed/`.




ATTENTION POUR RUN !!!!

```bash
# Depuis la racine du repo
PYTHONPATH=. python3 experiment/run.py           # LightGBM (GradientBoosting)
PYTHONPATH=. python3 experiment/runmlp.py        # MLP
PYTHONPATH=. python3 experiment/runlstm.py       # LSTM
PYTHONPATH=. python3 experiment/runmatyslp.py    # Variante MLP "MatysLP"
PYTHONPATH=. python3 experiment/runmleonardp.py  # Variante MLP "MLeonardP"
PYTHONPATH=. python3 experiment/runmlossp.py     # Variante avec loss custom
PYTHONPATH=. python3 experiment/runmlpaul.py     # Variante MLP "MLPaul"
```









Tous les modèles suivent la même interface (`fit(X, y, eval_set=...)` puis
`predict(X)`), définie dans `source/price_models/priceModel.py`, ce qui permet
de les comparer facilement.

### c) Notebooks

- `spot_forecast_project.ipynb` : présentation du sujet, exploration des
  données et de la démarche (une cellule utilise `google-cloud-bigquery` pour
  interroger l'entrepôt de données interne de NW — non nécessaire pour rejouer
  le reste du notebook, voir `PROJECT.md`).
- `gridsearch1.ipynb` : recherche d'hyperparamètres sur les différents modèles.

Lancer JupyterLab :

```bash
jupyter lab
```

### d) Évaluer la performance "métier" (arbitrage batterie)

La fonction `eval_performance` / `affichage_metrics` de
`source/utils/performance_metric.py` compare le profit réalisé par un modèle
(décidé sur les prix prédits, encaissé aux vrais prix) à deux références :

- **ORACLE** : planning optimal calculé sur les vrais prix (plafond théorique)
- **NAIVE** : planning fixe identique chaque jour (référence "sans modèle")

Chaque script de `experiment/` appelle automatiquement cette évaluation en fin
d'exécution.

---

## 4. Dépannage rapide

- **`ModuleNotFoundError: source`** : lancer les scripts depuis la racine du
  repo (pas depuis `experiment/`), ou ajouter la racine au `PYTHONPATH`.
- **Erreur liée à `google.cloud.bigquery`** : cette dépendance ne concerne que
  la cellule d'exploration des données internes NW dans le notebook de
  présentation ; elle peut être ignorée pour tout le reste du projet.
