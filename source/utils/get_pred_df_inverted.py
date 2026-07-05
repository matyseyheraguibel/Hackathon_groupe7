import pandas as pd
import numpy as np

def get_pred_df_inverted(X_test, prediction, csv_path_coef):
    # 1. Chargement du CSV
    df_coef = pd.read_csv(csv_path_coef)
    first_col_name = df_coef.columns[0]
    
    # LA SÉCURITÉ ABSOLUE : On extrait "AAAA-MM-JJ" du texte brut avant de convertir.
    # Aucun décalage UTC possible, on garde la vraie date locale.
    df_coef[first_col_name] = pd.to_datetime(df_coef[first_col_name].astype(str).str[:10])
    
    # On groupe par jour pour écraser les doublons éventuels
    df_coef = df_coef.groupby(first_col_name).mean()
    
    # 2. Harmonisation de l'index de X_test
    target_dates = pd.to_datetime(X_test.index).normalize()
    
    # 3. PROPAGATION (ffill) ... (Ton code existant)
    full_index = df_coef.index.union(target_dates)
    df_coef_extended = df_coef.reindex(full_index).ffill()
    
    # 4. LA CORRECTION : On décale d'un jour pour utiliser les stats du Jour N
    # On veut les coefficients calculés sur l'historique jusqu'au Jour N,
    # pour prédire le Jour N+1.
    dates_pour_coef = target_dates - pd.Timedelta(days=1)
    
    # On récupère les coefficients valides à la fin du Jour N
    mu = df_coef_extended.loc[dates_pour_coef, ['rolling_mean']].values
    sigma = df_coef_extended.loc[dates_pour_coef, ['rolling_std']].values

    # 5. Inversion de la standardisation
    y_pred_real_prices = (prediction * sigma) + mu

    # 6. Reconstruction du DataFrame final avec l'index d'origine de X_test
    df_final_preds = pd.DataFrame(
        y_pred_real_prices,
        index=X_test.index,
        columns=[f"pred_real_h_{i:02d}" for i in range(24)]
    )
    
    return df_final_preds