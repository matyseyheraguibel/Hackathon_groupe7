import pandas as pd

# 1. Charger le fichier CSV
df = pd.read_csv("spot_forecast_2026_NW.csv")

# 2. Convertir la colonne de temps en format datetime
df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
# 3. Extraire la date (pour les lignes) et l'heure (pour les colonnes)
df['Date'] = df['start_time'].dt.date
df['Heure'] = df['start_time'].dt.hour

# 4. Créer le tableau croisé (pivot_table)
df_pivot = df.pivot_table(
    index='Date',                           # Une ligne par jour
    columns='Heure',                        # Une colonne par heure
    values='spot_price_forecast_ensemble',  # La valeur à remplir (le prix spot)
    aggfunc='mean'                          # Moyenne au cas où il y aurait des doublons
)

# 5. (Optionnel) Renommer les colonnes pour un affichage plus clair (ex: '00h', '01h'...)
df_pivot.columns = [f"{heure:02d}h" for heure in df_pivot.columns]

# Afficher les 5 premières lignes pour vérifier le résultat
print(df_pivot.head())