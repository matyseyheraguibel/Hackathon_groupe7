import pandas as pd
def corriger_doublons(df):

    # 1. Vérification et conversion de la colonne en datetime (par sécurité)
    if not pd.api.types.is_datetime64_any_dtype(df['delivery_date']):
        df['delivery_date'] = pd.to_datetime(df['delivery_date'],utc='True')
        
    # 2. Suppression des doublons basés sur la date de livraison UTC

    df_clean = df.drop_duplicates(subset=['delivery_date'], keep='first').copy()
    
    # 3. Tri chronologique pour s'assurer que l'ordre est parfait (renumérotation de l'index)
    df_clean = df_clean.sort_values(by='delivery_date')
    
    # 4. Réinitialisation de l'index pour avoir un DataFrame propre
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean