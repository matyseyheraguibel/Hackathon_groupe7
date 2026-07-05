import pandas as pd
import numpy as np


def standardize_wind(df, window_days:int, pred_col='wind_prediction(MW)', time_col='delivery_date', output_col='standardized_wind'):
    """Standardise les previsions prod: (prev - moyenne_mobile) / écart_type_mobile"""

    window_hours = window_days * 24
    
    result = df.copy()  # Copie pour ne pas modifier l'original
    result[time_col] = pd.to_datetime(result[time_col])  # Convertir en datetime
    result = result.sort_values(time_col).set_index(time_col)  # Trier par temps et mettre en index
    
    window = f'{window_hours}h'  # Format pour rolling window (ex: '30D')
    mean = result[pred_col].rolling(window=window, min_periods=window_hours).mean()  # Moyenne mobile
    std = result[pred_col].rolling(window=window, min_periods=window_hours).std()  # Écart-type mobile
    
    # Standardiser: (prev - moyenne) / écart-type, remplacer div par 0 par NaN
    result[output_col] = (result[pred_col] - mean) / std.replace(0, np.nan)

    result = result.dropna(subset=[output_col])
    
    return result.reset_index()  # Remettre la colonne temps en colonne normale

