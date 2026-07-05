import pandas as pd
from sklearn.model_selection import train_test_split

from source.utils.get_pred_df_inverted import get_pred_df_inverted
from source.utils.visualisation import generate_evaluation_plots
from source.critic.criticMLP import CriticMLP
from source.critic.customEcoLoss import EconomicCriticLoss
from source.price_models.mlossp import MLPPriceModelwESwCL

synth = "data/processed/critic_dataset.csv"
df = pd.read_csv(synth)


X_critic = df.drop(columns=["capture_rate"])
y_critic = df["capture_rate"]

# Split classique (aléatoire) pour le modèle Critique
X_c_train, X_c_val, y_c_train, y_c_val = train_test_split(
    X_critic, y_critic, test_size=0.2, random_state=42
)

print("--- Entraînement du MLP Critique ---")

critic_model = CriticMLP(lr=1e-3, batch_size=128, epochs=150, patience=20)
critic_model.fit(X_c_train, y_c_train, eval_set=(X_c_val, y_c_val))


print("\n--- Chargement des vraies données de marché ---")
csv_path_X = "data/processed/X.csv"
csv_path_Y = "data/processed/y.csv"
csv_coef   = "data/processed/scaler_params.csv"

df_X = pd.read_csv(csv_path_X, index_col=0, parse_dates=True)
df_Y = pd.read_csv(csv_path_Y, index_col=0, parse_dates=True)

# Split temporel stricte pour le backtest
train_test_limit = "2026-01-01"
X_train = df_X[df_X.index < train_test_limit]
X_test  = df_X[df_X.index >= train_test_limit]
Y_train = df_Y[df_Y.index < train_test_limit]
Y_test  = df_Y[df_Y.index >= train_test_limit]

print("--- Initialisation de la Custom Loss ---")
# On crée l'objet de Loss en lui passant le critique fraîchement entraîné
# Commencez avec un beta=0.5. S'il n'y a pas assez d'impact, montez à 1.0 ou 2.0.
custom_criterion = EconomicCriticLoss(
    trained_critic_model=critic_model, 
    alpha=1.0, 
    beta=0.1 
)

print("--- Entraînement du MLPPriceModelwES ---")
# On initialise le modèle principal en lui passant la loss
mlp = MLPPriceModelwESwCL(
    criterion=custom_criterion,
    hidden_sizes=[64],
    scaler_params_path=csv_coef,
    weight_decay=1e-2,
    embedding_dim=4,
    lr=1e-3,
    batch_size=64,
    epochs=500,
    patience=30,
)

mlp.fit(X_train, Y_train, eval_set=(X_test, Y_test))

print("\nEntraînement terminé. Prêt pour l'évaluation !")
df_final_preds    = get_pred_df_inverted(X_test, mlp.predict(X_test), csv_coef)
df_y_test_inverted = get_pred_df_inverted(X_test, Y_test.values, csv_coef)

save_dir = "data/processed/evaluation_mlossp.png"

generate_evaluation_plots(
    df_y_true=df_y_test_inverted,
    df_y_pred=df_final_preds,
    save_path=save_dir,
)

affichage_metrics(df_y_test_inverted, df_final_preds)
