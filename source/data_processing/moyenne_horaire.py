import pandas as pd

def lisser_en_horaire(df):
    """
    Transforme un DataFrame avec des pas de 15 min (ou mixtes 1h/15min)
    en un DataFrame strictement horaire en moyennant les prix.
    """
    df = df.copy()
    df = df.reset_index(drop=True)

    df = df.loc[:, ~df.columns.duplicated()]
    
    # 1. S'assurer que la colonne est bien au format datetime
    if not pd.api.types.is_datetime64_any_dtype(df['delivery_date']):
        df['delivery_date'] = pd.to_datetime(df['delivery_date'])

    # 2. Règles explicites pour certaines colonnes
    regles_explicites = {
        'spot_price': 'mean',
        'Day-ahead Total Load Forecast (MW)': 'mean',
        'wind_prediction(MW)': 'mean',
    }

   
    
    # 3. Pour toutes les autres colonnes (hors delivery_date qui est l'index du resample),
    #    on applique 'min' par défaut pour conserver la première valeur de chaque heure
    colonnes_restantes = [c for c in df.columns if c != 'delivery_date' and c not in regles_explicites]
    regles_defaut = {col: 'min' for col in colonnes_restantes}

    # Fusion des règles
    regles_agregation = {**regles_explicites, **regles_defaut}

    # 4. Rééchantillonnage
    df_horaire = df.resample('1h', on='delivery_date').agg(regles_agregation)

    # 5. Réinitialiser l'index pour que 'delivery_date' redevienne une colonne normale
    df_horaire = df_horaire.reset_index()
    
    df = df.dropna(subset=["delivery_date"])
    df = df.set_index("delivery_date")
    
    return df_horaire

