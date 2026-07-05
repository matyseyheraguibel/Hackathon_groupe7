import pandas as pd
import numpy as np

# Imports des modèles (ajustez les noms de fichiers si nécessaire)
from source.price_models.mleonardp import MLPPriceModelwES
from source.price_models.gradientBoosting import GradientBoosting
from source.price_models.mlpaul import NeuralRouterPriceModel 

# Imports des utilitaires
from source.utils.get_pred_df_inverted import get_pred_df_inverted
from source.utils.visualisation import generate_evaluation_plots
from source.utils.performance_metric import affichage_metrics


csv_path_X = "data/processed/X.csv"
csv_path_Y = "data/processed/y.csv"
csv_coef   = "data/processed/scaler_params.csv"

df_X = pd.read_csv(csv_path_X, index_col=0, parse_dates=True)
df_Y = pd.read_csv(csv_path_Y, index_col=0, parse_dates=True)

print(f"Période totale du dataset : de {df_X.index[0].date()} à {df_X.index[-1].date()}")


ag_eval_limit = "2025-01-01"
train_test_limit = "2026-01-01"

# A. Train Base (< 2025) : Apprentissage brut du MLP et du LightGBM
X_train_base = df_X[df_X.index < ag_eval_limit]
Y_train_base = df_Y[df_Y.index < ag_eval_limit]

# B. Train Meta / AG (2025) : Apprentissage du réseau Routeur (et Early Stopping des modèles de base)
X_ag = df_X[(df_X.index >= ag_eval_limit) & (df_X.index < train_test_limit)]
Y_ag = df_Y[(df_Y.index >= ag_eval_limit) & (df_Y.index < train_test_limit)]

# C. Test Set (>= 2026) : Évaluation de l'Early Stopping du Routeur et Test final
X_test  = df_X[df_X.index >= train_test_limit]
Y_test  = df_Y[df_Y.index >= train_test_limit]

print(f"Tailles - Train Base : {len(X_train_base)} jours | Train Meta : {len(X_ag)} jours | Test : {len(X_test)} jours\n")


# Le modèle MLP de base
mlp_eco = MLPPriceModelwES(
    hidden_sizes=[64],
    scaler_params_path=csv_coef,
    weight_decay=1e-2,
    embedding_dim=4,
    lr=1e-3,
    batch_size=64,
    epochs=200,
    patience=30
)

# Le modèle Gradient Boosting de base
gb_model = GradientBoosting(
    scaler_params_path=csv_coef,
    n_estimators=200,  # Nombre d'arbres max
    learning_rate=0.05,
    patience=20
)

# Le Routeur Neuronal (qui va chapoter les deux autres)
neural_router = NeuralRouterPriceModel(
    mlp_model_instance=mlp_eco,
    gb_model_instance=gb_model,
    scaler_params_path=csv_coef,
    router_hidden_sizes=[32],  # Réseau léger (il ne prédit qu'un vecteur alpha de 0 à 1)
    embedding_dim=4,
    lr=1e-2,
    batch_size=64,
    epochs=500,
    patience=200,
    weight_decay=1e-3
)

# On passe les 3 sets pour que le modèle gère tout en interne sans Leakage
neural_router.fit(X_train_base, Y_train_base, X_ag, Y_ag, eval_set=(X_test, Y_test))



print("\n--- Étape Finale : Génération des prédictions sur 2026 ---")

# Le routeur va appeler les modèles de base, puis faire son mélange dynamique
preds_standardized = neural_router.predict(X_test)

# Désynchronisation des prédictions et de la vérité terrain (Standardisés -> € réels)
df_final_preds = get_pred_df_inverted(X_test, preds_standardized, csv_coef)
df_y_test_inverted = get_pred_df_inverted(X_test, Y_test.values, csv_coef)

# Sauvegarde du dashboard
save_dir = "data/processed/evaluation_neural_router.png"
generate_evaluation_plots(
    df_y_true=df_y_test_inverted,
    df_y_pred=df_final_preds,
    save_path=save_dir,
)

# Affichage des métriques d'arbitrage
affichage_metrics(df_y_test_inverted, df_final_preds)