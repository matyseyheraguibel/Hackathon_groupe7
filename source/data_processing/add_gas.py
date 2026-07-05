import pandas as pd
import numpy as np

def add_gas_features_to_X(X: pd.DataFrame, gas_csv_path: str) -> pd.DataFrame:
    """
    Charge l'historique du gaz, calcule les features (prix, pente, volatilité),
    décale d'un jour pour éviter le Data Leakage, et fusionne avec le dataset X.
    """
   # Data de investing (il faut se créer un compte)
    df_gas = pd.read_csv(
        gas_csv_path, 
        sep=',', 
        parse_dates=['Date'], 
        # Si format européen (JJ/MM/AAAA) : dayfirst=True
        dayfirst=True 
    )
    
    df_gas['Date'] = pd.to_datetime(df_gas['Date'])    
    df_gas = df_gas.set_index('Date').sort_index()

    # Handling week ends
    # On force un pas journalier et on prolonge le prix du vendredi sur le samedi/dimanche
    df_gas = df_gas.resample('D').ffill()

    # 3. Création des features
    df_gas['gas_price'] = df_gas['Price']
    # Pente sur 7 jours (Momentum)
    df_gas['mm7'] = df_gas['gas_price'].rolling(window=7).mean()
    df_gas['gas_slope_7'] = df_gas['mm7'] - df_gas['mm7'].shift(1)
    # Volatilité sur 7 jours (Écart-type des rendements)
    df_gas['gas_volatility_7'] = df_gas['gas_price'].pct_change().rolling(window=7).std()

    # DATA LEAKAGE PREVENTION
    # Pour prédire l'électricité du jour J, on ne connaît que le gaz du jour J-1
    df_gas_shifted = df_gas.shift(1)
    
    # On ne garde que les features créées
    features_gaz = df_gas_shifted[['mm7']]

    # 5. Alignement et Jointure avec X
    # On s'assure que l'index de X est bien de type datetime
    X = X.join(features_gaz, how="outer")

              
    return X