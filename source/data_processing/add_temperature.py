import requests
import pandas as pd
from datetime import date, timedelta

def add_temperature(X, start_date='2020-01-01'):
    # Coordonnées géographiques d'Helsinki
    latitude = 60.1695
    longitude = 24.9354
    
    # L'API d'archive a un décalage d'environ 5 jours pour les données consolidées
    end_date = (date.today() - timedelta(days=5)).strftime('%Y-%m-%d')
    
    # Point de terminaison de l'API Historique d'Open-Meteo
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Paramètres de la requête
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean", # Demande la moyenne journalière à 2m du sol
        "timezone": "Europe/Helsinki"
    }
    
    # Appel à l'API
    response = requests.get(url, params=params)
    
    # Vérification que la requête a fonctionné
    response.raise_for_status()
    
    # Extraction des données JSON
    data = response.json()
    
    # Création du DataFrame pandas
    df = pd.DataFrame({
        'Date': pd.to_datetime(data['daily']['time']),
        'temperature_moyenne': data['daily']['temperature_2m_mean']
    })
    
    # Mettre la date en index pour faciliter les fusions futures avec vos autres données
    df.set_index('Date', inplace=True)
    X = X.join(df, how="outer")
    
    return X

