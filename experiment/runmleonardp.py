import pandas as pd
import numpy as np
from source.price_models.mleonardp import MLPPriceModelwES
from source.utils.get_pred_df_inverted import get_pred_df_inverted
from source.utils.visualisation import generate_evaluation_plots
from source.utils.performance_metric import affichage_metrics


from source.utils.visualisation2 import generate_seasonal_analysis_v3

csv_path_X = "data/processed/X.csv"
csv_path_Y = "data/processed/y.csv"
csv_coef   = "data/processed/scaler_params.csv"

df_X = pd.read_csv(csv_path_X, index_col=0, parse_dates=True)
df_Y = pd.read_csv(csv_path_Y, index_col=0, parse_dates=True)

print(df_X.index)

train_test_limit = "2026-01-01"

X_train = df_X[df_X.index < train_test_limit]
X_test  = df_X[df_X.index >= train_test_limit]

Y_train = df_Y[df_Y.index < train_test_limit]
Y_test  = df_Y[df_Y.index >= train_test_limit]


mlp_eco = MLPPriceModelwES(
    hidden_sizes=[64],
    scaler_params_path=csv_coef,   # <-- seul paramètre ajouté vs le MLP classique
    weight_decay=1e-2,
    embedding_dim=4,
    lr=1e-3,
    batch_size=64,
    epochs=200,
    patience=30,
)

mlp_eco.fit(X_train, Y_train, eval_set=(X_test, Y_test))

df_final_preds    = get_pred_df_inverted(X_test, mlp_eco.predict(X_test), csv_coef)
df_y_test_inverted = get_pred_df_inverted(X_test, Y_test.values, csv_coef)

save_dir = "data/processed/evaluation_mleonardp.png"

generate_evaluation_plots(
    df_y_true=df_y_test_inverted,
    df_y_pred=df_final_preds,
    save_path=save_dir,
)

generate_seasonal_analysis_v3(df_y_test_inverted, df_final_preds, 'data/processed/evaluation_mleonardp_2.png')

affichage_metrics(df_y_test_inverted, df_final_preds)
