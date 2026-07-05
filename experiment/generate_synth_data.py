import pandas as pd
import numpy as np
from source.price_models.mleonardp import MLPPriceModelwES
from source.utils.get_pred_df_inverted import get_pred_df_inverted
from source.utils.visualisation import generate_evaluation_plots
from source.utils.performance_metric import affichage_metrics
from source.utils.visualisation2 import generate_seasonal_analysis_v3
from source.utils.generate_synthetic_data import generate_critic_dataset


if __name__ == "__main__":
    
    csv_path_X = "data/processed/X.csv"
    csv_path_Y = "data/processed/y.csv"
    csv_coef   = "data/processed/scaler_params.csv"

    df_X = pd.read_csv(csv_path_X, index_col=0, parse_dates=True)
    df_Y = pd.read_csv(csv_path_Y, index_col=0, parse_dates=True)

    print("Index des données :", df_X.index)

    train_test_limit = "2026-01-01"

    # Séparation Train / Test
    X_train = df_X[df_X.index < train_test_limit]
    X_test  = df_X[df_X.index >= train_test_limit]

    Y_train = df_Y[df_Y.index < train_test_limit]
    Y_test  = df_Y[df_Y.index >= train_test_limit]

    # Initialisation et entraînement
    mlp_eco = MLPPriceModelwES(
        hidden_sizes=[64],
        scaler_params_path=csv_coef,
        weight_decay=1e-2,
        embedding_dim=4,
        lr=1e-3,
        batch_size=64,
        epochs=200,
        patience=30,
    )

    print("--- Entraînement du modèle ---")
    mlp_eco.fit(X_train, Y_train, eval_set=(X_test, Y_test))

    # Évaluation
    df_final_preds     = get_pred_df_inverted(X_test, mlp_eco.predict(X_test), csv_coef)
    df_y_test_inverted = get_pred_df_inverted(X_test, Y_test.values, csv_coef)

    save_dir = "data/processed/evaluation_mleonardp.png"
    generate_evaluation_plots(df_y_test_inverted, df_final_preds, save_path=save_dir)
    generate_seasonal_analysis_v3(df_y_test_inverted, df_final_preds, 'data/processed/evaluation_mleonardp_2.png')
    affichage_metrics(df_y_test_inverted, df_final_preds)

    # =====================================================================
    # GÉNÉRATION DU DATASET CRITIQUE (Uniquement sur le jeu d'entraînement)
    # =====================================================================
    print("\n--- Génération du dataset synthétique pour le Critique ---")
    output = "data/processed/critic_dataset.csv"
    
    # ASTUCE : On passe X_train et Y_train directement à la fonction au lieu des chemins CSV
    # Il faudra juste modifier la fonction generate_critic_dataset pour qu'elle accepte
    # df_X et df_Y en paramètres (au lieu de lire pd.read_csv en interne).
    df_critic = generate_critic_dataset(
        model=mlp_eco,
        df_X=X_train,     # <-- On force l'utilisation exclusive du Train set
        df_Y=Y_train,     # <-- On force l'utilisation exclusive du Train set
        coef_path=csv_coef,
        output_path=output,
        noise_levels=[0.0, 0.1, 0.3, 0.5, 1.0, 1.5]
    )