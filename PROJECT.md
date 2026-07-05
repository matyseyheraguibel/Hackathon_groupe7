# PROJECT.md — Description du projet

## Contexte du sujet

**Entreprise :** NW, opérateur de stockage d'électricité par batterie (solution
"JBox", ~1000 unités installées en France et en Finlande, ~1.8 GWh). Sa filiale
d'agrégation achète et vend l'électricité stockée sur le marché SPOT.

**Objectif :** développer un algorithme de prévision des prix du marché SPOT
finlandais (J+1), afin d'optimiser les décisions de charge/décharge d'une
batterie **2 MWh / 1 MW**.

**Critère de réussite :** comparer les revenus générés par une stratégie de
trading basée sur les prix prédits à ceux obtenus par une connaissance parfaite
des prix (voir `source/utils/performance_metric.py`).

---

## Démarche

1. **Collecte et nettoyage des données** : prix spot historiques finlandais,
   prévisions de production/consommation (ENTSO-E), production éolienne,
   prix du gaz (Dutch TTF), jours fériés finlandais, température.
2. **Feature engineering** : encodage cyclique du jour de la semaine
   (sin/cos), standardisation glissante des prix sur fenêtre de 30 jours
   (pour absorber la tendance et la saisonnalité sans "voir le futur"),
   construction d'une fenêtre glissante d'historique de prix (`prepare_dataset`)
   pour prédire les 24 prix horaires du jour J+1 à partir des jours précédents.
3. **Modélisation** : plusieurs familles de modèles avec une interface commune
   (`fit` / `predict`, classe abstraite `PriceModel`) pour pouvoir les comparer
   facilement :
   - **LightGBM** (`GradientBoosting`) — baseline robuste sur données tabulaires.
   - **MLP** (plusieurs variantes : `MLPPriceModel`, `MatysLPPriceModel`,
     `MLeonardPPriceModel`, `MLPaulPriceModel`) — réseaux denses avec un
     embedding appris du jour de la semaine, concaténé aux features continues.
   - **LSTM** (`LSTMPriceModel`) — pour capter la dépendance séquentielle de la
     série de prix.
   - **Réseau "critique"** (`criticMLP`, `customEcoLoss`) — tentative
     d'entraîner un modèle avec une fonction de coût alignée directement sur la
     métrique économique d'arbitrage plutôt que sur une simple erreur de
     régression (MSE/RMSE).
4. **Évaluation** : au-delà des métriques statistiques classiques, la métrique
   principale est économique : profit réalisé (décision sur prix prédits,
   encaissement aux vrais prix) rapporté au profit d'un planning **ORACLE**
   (prix parfaitement connus) et à un planning **NAIVE** (heures de
   charge/décharge fixes, sans modèle).

---

## Choix techniques

- **Package Python structuré** (`source/`) plutôt qu'un unique notebook, pour
  pouvoir réutiliser le même pipeline de données et la même interface de
  modèle dans plusieurs expériences (`experiment/run*.py`) et dans les
  notebooks de gridsearch/présentation.
- **Interface commune `fit(X, y, eval_set=...)` / `predict(X)`** pour tous les
  modèles, afin de pouvoir les brancher indifféremment dans le même script
  d'évaluation.
- **Standardisation glissante des prix** (plutôt qu'une normalisation globale)
  pour éviter la fuite d'information (data leakage) entre passé et futur, et
  pour que le modèle apprenne des écarts relatifs plutôt que le niveau absolu
  des prix (qui a beaucoup varié sur la période).
- **LightGBM comme baseline** avant les réseaux de neurones, pour avoir un
  point de comparaison simple et rapide à entraîner.

---

## Difficultés rencontrées et solutions

- **Désalignement de dates lors de la fusion des sources** (prix, production,
  vent, gaz, température n'ayant pas les mêmes fuseaux horaires / granularités
  horaires) : résolu en centralisant la conversion de toutes les séries vers
  un même index horaire en heure locale Paris avant fusion (`add_times`,
  `moyenne_horaire`).
- **Incohérence de dimension d'embedding** dans les modèles à réseaux de
  neurones (le nombre de features continues passé à la construction du réseau
  ne correspondait pas à celui réellement produit par le pipeline de données
  après ajout de nouvelles features) : résolu en calculant dynamiquement
  `num_cont_features` à partir des données d'entrée plutôt qu'en le codant en
  dur.
- **Incohérence dans la fonction de coût métier** (le calcul de profit de la
  métrique d'arbitrage ne correspondait pas exactement à la contrainte réelle
  de la batterie — capacité 2 MWh, 2 cycles max par jour, coût de cycle) :
  corrigé en revérifiant les bornes (`MAX_DISCHARGE`, `CAPACITY`,
  `CYCLE_COST`) et la logique du planning naïf de référence dans
  `performance_metric.py`.
- **Accès aux données internes NW** : l'exploration initiale de certaines
  données passait par l'entrepôt BigQuery interne de NW
  (`google.cloud.bigquery`, visible dans `spot_forecast_project.ipynb`). Cet
  accès n'étant pas disponible en dehors de l'entreprise, les données
  nécessaires ont été exportées en CSV et versionnées dans `data/raw/` /
  `data/processed/` afin que l'ensemble du pipeline (hors cette cellule
  d'exploration) soit exécutable par un tiers sans accès à NW.

---

## Organisation

- Un modèle par personne/duo (`matyslp`, `mleonardp`, `mlossp`, `mlpaul`,
  `lstm`, `gradientBoosting`), partageant le même pipeline de données et la
  même métrique d'évaluation, pour permettre une comparaison directe des
  approches en fin de projet.
- Le notebook `gridsearch1.ipynb` a servi de banc d'essai commun pour la
  recherche d'hyperparamètres sur les différents modèles.
- Le notebook `spot_forecast_project.ipynb` regroupe la présentation du sujet,
  le corpus de sources et l'exploration des données.

---

## Ce qui aurait été fait différemment avec plus de temps

- Automatiser la recherche d'hyperparamètres (au lieu d'un notebook manuel) via
  un script de gridsearch/optuna réutilisable et versionné, avec sauvegarde
  systématique des résultats.
- Ajouter des tests unitaires sur le pipeline de données (notamment
  `prepare_dataset`) pour prévenir les régressions de type désalignement de
  dates.
- Pousser davantage l'approche "réseau critique" avec une fonction de coût
  économique (`customEcoLoss`), qui n'a été qu'esquissée par manque de temps.
- Étendre la zone de prédiction à la France et aux autres pays nordiques,
  comme suggéré dans le sujet initial.
- Mettre en place une validation temporelle plus robuste (walk-forward /
  rolling backtest) plutôt qu'un simple split train/test à une date fixe.
