import pandas as pd
import numpy as np
from source import price_models
from source.utils.get_pred_df_inverted import get_pred_df_inverted
from source.utils.visualisation import generate_evaluation_plots
from source.utils.performance_metric import affichage_metrics
from source.utils.performance_metric import eval_performance



csv_path_X = "data/processed/X.csv"
csv_path_Y = "data/processed/y.csv"

csv_coef = "data/processed/scaler_params.csv"

df_X = pd.read_csv(csv_path_X, index_col=0, parse_dates=True)
df_Y = pd.read_csv(csv_path_Y, index_col=0, parse_dates=True)

print(df_X.index)

train_test_limit = "2026-01-01"

X_train = df_X[df_X.index < train_test_limit]
X_test = df_X[df_X.index >= train_test_limit]

Y_train = df_Y[df_Y.index < train_test_limit]
Y_test = df_Y[df_Y.index >= train_test_limit]


mlpp = price_models.MLPPriceModel(hidden_sizes=[64],
                 weight_decay = 1e-2,
                 embedding_dim=4, 
                 lr=1e-3, 
                 batch_size=64, 
                 epochs=200, 
                 patience=10)

mlpp.fit(X_train, Y_train, eval_set=(X_test, Y_test), metric_funs=[eval_performance])


df_final_preds = get_pred_df_inverted(X_test, mlpp.predict(X_test), csv_coef)
df_y_test_inverted = get_pred_df_inverted(X_test, Y_test.values, csv_coef)



# À la toute fin de ton run.py

# df_final_preds est le DataFrame qui sort de ta fonction get_pred_df_inverted
# Y_test est ton jeu de test d'origine chargé au début du script

save_dir = "data/processed/evaluation_mlp.png"

generate_evaluation_plots(
    df_y_true=df_y_test_inverted, 
    df_y_pred=df_final_preds, 
    save_path=save_dir
)



affichage_metrics(df_y_test_inverted,df_final_preds)