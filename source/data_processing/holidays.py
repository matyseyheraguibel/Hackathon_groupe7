import pandas as pd
from datetime import date, timedelta

def calculer_paques(annee):
    """Calcule la date du dimanche de Pâques pour une année donnée (Algorithme de Meeus/Jones/Butcher)"""
    a = annee % 19
    b = annee // 100
    c = annee % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mois = (h + l - 7 * m + 114) // 31
    jour = ((h + l - 7 * m + 114) % 31) + 1
    return date(annee, mois, jour)

def jours_feries_finlande_annee(annee):
    """Retourne un ensemble (set) de toutes les dates fériées finlandaises pour une année donnée"""
    feries = set()
    
    # --- 1. Les dates fixes ---
    feries.add(date(annee, 1, 1))   # Nouvel An (Uudenvuodenpäivä)
    feries.add(date(annee, 1, 6))   # Épiphanie (Loppiainen)
    feries.add(date(annee, 5, 1))   # Fête du Travail (Vappu)
    feries.add(date(annee, 12, 6))  # Fête de l'Indépendance (Itsenäisyyspäivä)
    feries.add(date(annee, 12, 24)) # Veille de Noël (Jouluaatto) - de facto férié en FI
    feries.add(date(annee, 12, 25)) # Noël (Joulupäivä)
    feries.add(date(annee, 12, 26)) # Saint-Étienne (Tapaninpäivä)
    
    # --- 2. Les dates mobiles liées à Pâques ---
    paques = calculer_paques(annee)
    feries.add(paques - timedelta(days=2))  # Vendredi Saint (Pitkäperjantai)
    feries.add(paques)                      # Dimanche de Pâques (Pääsiäispäivä)
    feries.add(paques + timedelta(days=1))  # Lundi de Pâques (Toinen pääsiäispäivä)
    feries.add(paques + timedelta(days=39)) # Ascension (Helatorstai)
    feries.add(paques + timedelta(days=49)) # Pentecôte (Helluntaipäivä)
    
    # --- 3. Les dates mobiles liées aux week-ends ---
    # Juhannuspäivä (Saint-Jean) : Toujours le samedi entre le 20 et le 26 juin
    juhannus = date(annee, 6, 20)
    while juhannus.weekday() != 5: # 5 correspond au Samedi
        juhannus += timedelta(days=1)
    feries.add(juhannus)
    feries.add(juhannus - timedelta(days=1)) # Veille de la St-Jean (Juhannusaatto, vendredi)

    # Pyhäinpäivä (Toussaint) : Toujours le samedi entre le 31 octobre et le 6 novembre
    pyhainpaiva = date(annee, 10, 31)
    while pyhainpaiva.weekday() != 5:
        pyhainpaiva += timedelta(days=1)
    feries.add(pyhainpaiva)
    
    return feries

def generer_df_feries_finlande(start_date='2020-01-01', end_date=None):
    if end_date is None:
        end_date = date.today()
        
    # Générer la séquence de tous les jours
    dates = pd.date_range(start=start_date, end=end_date)
    
    # Trouver toutes les années impliquées dans l'intervalle
    annees = dates.year.unique()
    
    # Compiler tous les jours fériés de ces années dans un set (pour la rapidité de recherche)
    tous_les_feries = set()
    for annee in annees:
        tous_les_feries.update(jours_feries_finlande_annee(annee))
        
    # Créer le DataFrame
    df = pd.DataFrame({'Date': dates})
    
    # Ajouter la colonne binaire : 1 si la date est dans le set des jours fériés, sinon 0
    df['Est_Ferie'] = df['Date'].dt.date.apply(lambda x: 1 if x in tous_les_feries else 0)
    
    # Mettre la date en index
    df.set_index('Date', inplace=True)
    
    return df

# === EXÉCUTION ===
df_finlande = generer_df_feries_finlande()

# Afficher les premières lignes
print("--- Aperçu du DataFrame ---")
print(df_finlande.tail(10))

# Afficher quelques jours fériés spécifiques de 2020 pour vérifier
print("\n--- Jours Fériés trouvés en 2020 ---")
print(df_finlande[(df_finlande.index.year == 2020) & (df_finlande['Est_Ferie'] == 1)].tail(15))