import pandas as pd
import numpy as np

#La standardisation
def standardize_pred_prod_rolling(df, window_days:int, pred_col='Day-ahead Total Load Forecast (MW)', time_col='delivery_date', output_col='standardized_pred_prod'):
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



#La difference avec la moyenne saisonière
def add_rolling_seasonal_anomaly(
    df: pd.DataFrame,
    datetime_col: str,
    target_col: str,
    window_days: int = 30
) -> pd.DataFrame:
    """
    Ajoute un écart entre la valeur actuelle et la moyenne des 30 jours précédents
    conditionnée par heure + type de jour (weekday/weekend).

    Parameters
    ----------
    df : pd.DataFrame
    datetime_col : str
        colonne datetime
    target_col : str
        variable à transformer
    window_days : int
        taille de la fenêtre en jours (par défaut 30)

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # --- features temporelles ---
    df["hour"] = df[datetime_col].dt.hour
    df["is_weekend"] = (df[datetime_col].dt.dayofweek >= 5).astype(int)

    # tri obligatoire
    df = df.sort_values(datetime_col)

    window = window_days * 24  # conversion jours -> heures

    # --- fonction appliquée par groupe ---
    def compute_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(datetime_col).set_index(datetime_col)

        group["rolling_mean"] = (
            group[target_col]
            .shift(24)  # évite fuite sur la journée courante
            .rolling(window=window, min_periods=24)
            .mean()
        )

        group["seasonal_rolling_anomaly"] = (
            group[target_col] - group["rolling_mean"]
        )

        return group.reset_index()

    # --- application par heure + type de jour ---
    df = df.groupby(["hour", "is_weekend"], group_keys=False).apply(compute_group)

    # nettoyage
    df = df.drop(columns=["hour", "is_weekend"])

    return df