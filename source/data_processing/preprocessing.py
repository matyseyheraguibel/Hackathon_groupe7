import pandas as pd
import numpy as np

#TRAITEMENT TABLE

from encodage_prix import standardize_prices_rolling
from add_times import add_times
from drop_duplicate import corriger_doublons
from moyenne_horaire import lisser_en_horaire
from concatenate import concat
from add_prediction_production import add_prediction_production
from add_prediction_production import df_prod
from encodage_pred_prod import standardize_pred_prod_rolling
from prepare_dataset import prepare_dataset
from add_wind import add_wind
from add_wind import df_wind
from encodage_wind import standardize_wind
from holidays import generer_df_feries_finlande
from encodage_pred_prod import add_rolling_seasonal_anomaly
from add_gas import add_gas_features_to_X
from add_temperature import add_temperature

gas_path = "../../data/raw/Dutch TTF Natural Gas Futures Historical Data.csv"
 
df=pd.read_csv("../../data/raw/data_raw.csv")

print(df.columns)
df=add_times(df)
print(df.columns)
df=add_prediction_production(df,df_prod)
print(df.columns)
df=add_wind(df,df_wind)
print(df.columns)
df=corriger_doublons(df)
print(df.columns)
df=lisser_en_horaire(df)
print(df.columns)
df=standardize_pred_prod_rolling(df, window_days=30, pred_col='Day-ahead Total Load Forecast (MW)', time_col='delivery_date', output_col='standardized_pred_prod')
#df=add_rolling_seasonal_anomaly(df,datetime_col = 'delivery_date', target_col = 'Day-ahead Total Load Forecast (MW)', window_days = 30)
df=standardize_wind(df, window_days=30, pred_col='wind_prediction(MW)', time_col='delivery_date', output_col='standardized_wind')
df, scaler_params = standardize_prices_rolling(df=df, window_days=30, price_col='spot_price', time_col='delivery_date', output_col='standardized_price')

X, y = prepare_dataset(df, window_days=1)
df_feries = generer_df_feries_finlande(start_date='2020-01-01', end_date=None)
X = X.join(df_feries, how="outer")
#X = add_gas_features_to_X(X, gas_path)
#X = add_temperature(X)
#X = X.dropna()
#y = y.reindex(X.index)


print('X head:')
print(X.tail(186))
print('y head:')
print(y.head())
print(df.head())
print(X.shape)
print(y.shape)

#df.to_csv('votre_nom_de_fichier.csv', index=False)

X.to_csv("../../data/processed/X.csv")
y.to_csv("../../data/processed/y.csv")
scaler_params.to_csv("../../data/processed/scaler_params.csv", index=False)
