import pandas as pd
import numpy as np


def standardize_prices_rolling(df, window_days:int, price_col='spot_price', time_col='delivery_date', output_col='standardized_price'):
    """Standardise les prix: (prix - moyenne_mobile) / ecart_type_mobile

    Retourne
    --------
    result : pd.DataFrame
        DataFrame original enrichi de la colonne output_col (prix standardises).
        Les lignes sans historique suffisant (debut de serie) sont supprimees.
    scaler_params : pd.DataFrame
        DataFrame avec une ligne par jour (snapshot a 11h du jour N), colonnes :
            date_N       : date du jour N (objet datetime.date)
            rolling_mean : moyenne mobile a 11h du jour N
            rolling_std  : ecart-type mobile a 11h du jour N
        Utilisation pour destandardiser les predictions du jour N+1 :
            spot_price_pred = standardized_pred * rolling_std + rolling_mean
        Le join se fait : scaler_params.set_index('date_N').loc[date_N]
        ou date_N = date_cible_N+1 - timedelta(days=1)
    """

    window_hours = window_days * 24

    result = df.copy()
    result[time_col] = pd.to_datetime(result[time_col])
    result = result.sort_values(time_col).set_index(time_col)

    window = f'{window_hours}h'
    mean = result[price_col].rolling(window=window, min_periods=window_hours).mean()
    std  = result[price_col].rolling(window=window, min_periods=window_hours).std()

    result[output_col] = (result[price_col] - mean) / std.replace(0, np.nan)
    result = result.dropna(subset=[output_col])

    # Construire scaler_params : une ligne par jour, snapshot a 23h
    # On reindexe mean/std sur les memes lignes que result (apres dropna)
    params_full = pd.DataFrame({
        'rolling_mean': mean,
        'rolling_std':  std,
    }, index=mean.index)
    params_full = params_full.loc[result.index]  # aligner apres dropna

    # Filtrer uniquement les slots a 23h
    params_23h = params_full[params_full.index.hour == 23].copy()
    params_23h.index = params_23h.index.normalize()  # garder juste la date (minuit)
    params_23h.index.name = 'date_N'
    scaler_params = params_23h.reset_index()

    return result.reset_index(), scaler_params
