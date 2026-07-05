import pandas as pd
from source import price_models
from source.utils.get_pred_df_inverted import get_pred_df_inverted
from source.utils.visualisation import generate_evaluation_plots
from source.utils.performance_metric import eval_performance
from source.utils.performance_metric import eval_all
from source.utils.performance_metric import affichage_metrics


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


gBoost = price_models.GradientBoosting(n_estimators = 200, min_child_samples = 10, num_leaves = 15, objective="regression")

gBoost.fit(X_train, Y_train, eval_set=(X_test, Y_test))

df_final_preds = get_pred_df_inverted(X_test, gBoost.predict(X_test), csv_coef)
df_y_test_inverted = get_pred_df_inverted(X_test, Y_test.values, csv_coef)





save_dir = "data/processed/evaluation_lightgbm.png"

generate_evaluation_plots(
    df_y_true=df_y_test_inverted, 
    df_y_pred=df_final_preds, 
    save_path=save_dir
)



affichage_metrics(df_y_test_inverted,df_final_preds)