import pandas as pd
from source.price_models.mleonardp import MLPPriceModelwES
from source.utils.get_pred_df_inverted import get_pred_df_inverted
from source.utils.visualisation import generate_evaluation_plots
from source.utils.performance_metric import affichage_metrics

# 1. Chargement complet
df_X = pd.read_csv("data/processed/X.csv", index_col=0, parse_dates=True)
df_Y = pd.read_csv("data/processed/y.csv", index_col=0, parse_dates=True)

# 2. Définition de la période d'entraînement initiale et de test
start_date = pd.Timestamp("2026-01-01")
weeks = pd.date_range(start=start_date, end=df_X.index.max(), freq='W')

# Containers pour agréger les résultats semaine par semaine
all_preds = []
all_true = []

# 3. BOUCLE D'ENTRAÎNEMENT HEBDOMADAIRE (Rolling Window)
current_train_end = start_date

for i in range(len(weeks) - 1):
    week_start = weeks[i]
    week_end = weeks[i+1]
    
    print(f"\n--- Entraînement pour la semaine du {week_start.date()} ---")
    
    # Dataset dynamique : tout l'historique dispo jusqu'à la semaine courante
    X_train = df_X[df_X.index < week_start]
    Y_train = df_Y[df_Y.index < week_start]
    
    # Semaine cible
    X_test = df_X[(df_X.index >= week_start) & (df_X.index < week_end)]
    Y_test = df_Y[(df_Y.index >= week_start) & (df_Y.index < week_end)]
    
    # Ré-initialisation du modèle pour éviter le surapprentissage cumulé
    model = MLPPriceModelwES(
        hidden_sizes=[64],
        scaler_params_path="data/processed/scaler_params.csv",
        weight_decay=1e-2,
        embedding_dim=4,
        lr=1e-3,
        batch_size=64,
        epochs=200,
        patience=30
    )
    
    model.fit(X_train, Y_train)
    
    # Prédictions
    preds = model.predict(X_test)
    all_preds.append(get_pred_df_inverted(X_test, preds, "data/processed/scaler_params.csv"))
    all_true.append(get_pred_df_inverted(X_test, Y_test.values, "data/processed/scaler_params.csv"))

# 4. CONCATÉNATION FINALE
df_final_preds = pd.concat(all_preds)
df_y_test_inverted = pd.concat(all_true)

# 5. VISUALISATION ET MÉTRIQUES FINALES
generate_evaluation_plots(df_y_test_inverted, df_final_preds, save_path="data/processed/evaluation_rolling.png")
affichage_metrics(df_y_test_inverted, df_final_preds)