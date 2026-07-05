import pandas as pd
import numpy as np

def add_times(df):
    
    # Conversion en datetime avec timezone
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], utc=True).dt.tz_convert("Europe/Paris")

    # Conversion en heure locale Paris (gère automatiquement été/hiver)
    df["date_paris"] = df["delivery_date"]
    
    # 1. Jour de la semaine (0=lundi, 6=dimanche)
    df["day_of_week"] = df["date_paris"].dt.dayofweek

    # 2. Semaine (0 à 51) avec sinus
    df["week"] = df["date_paris"].dt.isocalendar().week - 1
    df["sin_week"] = np.sin((df["date_paris"].dt.isocalendar().week - 1)*2*np.pi/52)
    df["cos_week"] = np.cos((df["date_paris"].dt.isocalendar().week - 1)*2*np.pi/52)

    # 3. Heure locale (0 à 23)
    df["hour"] = df["date_paris"].dt.hour

    #on indexe pour pouvoir bien concatener les df apres
    df = df.set_index("date_paris")
    
    return(df)