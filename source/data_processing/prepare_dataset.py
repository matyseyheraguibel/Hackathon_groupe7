import pandas as pd
import numpy as np


def prepare_dataset(
    df: pd.DataFrame,
    window_days: int = 7,
    price_col: str = "standardized_price",
    date_col: str = "delivery_date",
):
    """
    Construit les DataFrames X et y pour l entrainement d un modele de prevision
    du prix spot electrique sur la journee suivante (24h).

    Attend un DataFrame deja traite par le pipeline de preprocessing, avec au
    minimum les colonnes suivantes :
        date_paris            datetime heure locale Paris (avec ou sans tzinfo)
        day_of_week           entier 0=lundi ... 6=dimanche
        sin_week, cos_week    encodage cyclique de la semaine (float)
        standardized_price     prix standardise par fenetre glissante (float)
        standardized_pred_prod prevision de production standardisee (float, optionnelle)

    Structure de X (une ligne = un jour a predire, index = date cible N+1) :
        col 0       : day_of_week   du jour N+1  (entier 0-6)
        col 1       : sin_week      du jour N+1
        col 2       : cos_week      du jour N+1
        col 3...    : prix standardises en remontant depuis 11h du jour N
                      ordre : price_N-0_h11, price_N-0_h10, ..., price_N-0_h00,
                              price_N-1_h23, ..., price_N-{T}_h12

    Structure de y (meme index que X) :
        colonnes h00, h01, ..., h23  ->  prix standardise du jour N+1 heure par heure

    Parametres
    ----------
    df : pd.DataFrame
        DataFrame produit par le pipeline (corriger_doublons -> add_times ->
        lisser_en_horaire -> standardize_prices_rolling). Doit etre trie
        chronologiquement et contenir une ligne par heure.
    window_days : int
        Nombre de jours d historique a inclure dans X. La fenetre couvre
        window_days * 24 slots en remontant depuis 11h du jour N inclus.
        Defaut : 7 jours.
    price_col : str
        Colonne de prix a utiliser dans X et y. Defaut : 'standardized_price'.
    date_col : str
        Colonne datetime heure locale Paris. Defaut : 'date_paris'.

    Retourne
    -------
    X : pd.DataFrame, shape (n_samples, 3 + window_days * 24)
        Index : date du jour cible (N+1), nom 'date_target'
        Colonnes : day_of_week_target, sin_week_target, cos_week_target,
                   price_N-0_h11, price_N-0_h10, ..., price_N-{T}_h12
    y : pd.DataFrame, shape (n_samples, 24)
        Index : meme index que X
        Colonnes : h00, h01, ..., h23
    """

    # ── 1. Extraction de la date et de l heure depuis date_paris ────────────
    # Les fuseaux horaires ont deja ete traites en amont : on extrait juste
    # la date locale et l heure locale depuis la colonne existante.
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    # Si tzinfo present, on travaille directement sur les valeurs locales
    if dt.dt.tz is not None:
        dt = dt.dt.tz_localize(None)

    df["_date"] = dt.dt.date    # objet datetime.date, cle principale du lookup
    df["_hour"] = dt.dt.hour    # entier 0-23

    # Trier par ordre chronologique (securite)
    df = df.sort_values(["_date", "_hour"]).reset_index(drop=True)

    # ── 2. Dictionnaires de lookup rapides ───────────────────────────────────
    # (date, heure) -> prix standardise
    price_lookup = dict(zip(
        zip(df["_date"], df["_hour"]),
        df[price_col]
    ))

    # (date, heure) -> standardized_pred_prod (optionnel)
    prod_lookup = {}
    if "standardized_pred_prod" in df.columns:
        prod_lookup = dict(zip(
            zip(df["_date"], df["_hour"]),
            df["standardized_pred_prod"]
        ))

    wind_lookup = {}
    if "standardized_wind" in df.columns:
        wind_lookup = dict(zip(
            zip(df["_date"], df["_hour"]),
            df["standardized_wind"]
        ))

    # date -> (day_of_week, sin_week, cos_week) du premier slot de ce jour
    # Ces valeurs sont identiques pour toutes les heures d un meme jour,
    # on prend simplement la premiere occurrence.
    df_first = df.groupby("_date").first().reset_index()
    temporal_lookup = {
        row["_date"]: (row["day_of_week"], row["sin_week"], row["cos_week"])
        for _, row in df_first[["_date", "day_of_week", "sin_week", "cos_week"]].iterrows()
    }

    # ── 3. Jours disponibles et parametres de fenetre ───────────────────────
    all_dates    = sorted(df["_date"].unique())
    window_hours = window_days * 24

    # ── 4. Noms des colonnes ─────────────────────────────────────────────────
    feature_names = ["day_of_week_target", "sin_week_target", "cos_week_target"]
    for i in range(window_hours):
        # On part de 23h en reculant
        abs_hour = 23 - i
        days_back = 0
        if abs_hour < 0:
            days_back = (-abs_hour - 1) // 24 + 1
            abs_hour  = 23 - ((-abs_hour - 1) % 24)
        feature_names.append(f"price_N-{days_back}_h{abs_hour:02d}")

    prod_col_names = [f"pred_prod_h{h:02d}" for h in range(24)]
    wind_col_names = [f"wind_h{h:02d}" for h in range(24)]
    y_col_names    = [f"h{h:02d}" for h in range(24)]

    # ── 5. Boucle principale ─────────────────────────────────────────────────
    X_rows  = []
    y_rows  = []
    indices = []

    for idx in range(1, len(all_dates)):
        target_date = all_dates[idx]       # jour N+1 (a predire)
        day_N       = all_dates[idx - 1]   # jour N   (dernier jour connu)

        # a. Encodage temporel du jour cible (colonnes deja calculees en amont)
        if target_date not in temporal_lookup:
            continue
        dow, sin_w, cos_w = temporal_lookup[target_date]

        # b. Cible y : prix standardises des 24h du jour N+1 (0h -> 23h)
        y_prices     = []
        valid_target = True
        for h in range(24):
            val = price_lookup.get((target_date, h))
            if val is None or (isinstance(val, float) and np.isnan(val)):
                valid_target = False
                break
            y_prices.append(float(val))
        if not valid_target:
            continue

        # c. Fenetre historique : en remontant depuis 11h du jour N
        x_prices      = []
        valid_history = True
        start_dt      = pd.Timestamp(day_N) + pd.Timedelta(hours=23)

        for step in range(window_hours):
            current_dt   = start_dt - pd.Timedelta(hours=step)
            current_date = current_dt.date()
            current_hour = current_dt.hour
            val = price_lookup.get((current_date, current_hour))
            if val is None or (isinstance(val, float) and np.isnan(val)):
                valid_history = False
                break
            x_prices.append(float(val))
        if not valid_history:
            continue

        # d. Prevision de production du jour N+1 (24 valeurs horaires, h00->h23)
        prod_values = []
        if prod_lookup:
            valid_prod = True
            for h in range(24):
                val = prod_lookup.get((target_date, h))
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    valid_prod = False
                    break
                prod_values.append(float(val))
            if not valid_prod:
                continue

        wind_values = []
        if wind_lookup:
            valid_wind = True
            for h in range(24):
                val = wind_lookup.get((target_date, h))
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    valid_wind = False
                    break
                wind_values.append(float(val))
            if not valid_wind:
                continue

        # e. Assemblage simultane (garantit X[i] <-> y[i] <-> indices[i])
        X_rows.append([float(dow), float(sin_w), float(cos_w)] + x_prices + prod_values + wind_values)
        y_rows.append(y_prices)
        indices.append(target_date)

    # ── 6. Construction des DataFrames ───────────────────────────────────────
    all_feature_names = ( feature_names + (prod_col_names if prod_lookup else []) + (wind_col_names if wind_lookup else []))
    X = pd.DataFrame(X_rows, index=indices, columns=all_feature_names)
    y = pd.DataFrame(y_rows, index=indices, columns=y_col_names)

    X.index.name = "date_target"
    y.index.name = "date_target"
    return X, y





"""
# ── Point d entree autonome (test rapide) ────────────────────────────────────
if __name__ == "__main__":
    import os, sys

    # On reproduit le pipeline complet pour obtenir le DataFrame attendu
    sys.path.insert(0, os.path.dirname(__file__))
    from encodage_prix   import standardize_prices_rolling
    from add_times       import add_times
    from drop_duplicate  import corriger_doublons
    from moyenne_horaire import lisser_en_horaire

    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw", "data_raw.csv"
    )
    df = pd.read_csv(csv_path)
    df = corriger_doublons(df)
    df = add_times(df)
    df = lisser_en_horaire(df)
    df = standardize_prices_rolling(df=df, window_days=30, price_col='spot_price',
                                    time_col='date_paris', output_col='standardized_price')

    print("\nColonnes du DataFrame en entree :", list(df.columns))
    print("Shape en entree                  :", df.shape)

    for T in [7, 14]:
        X, y = prepare_dataset(df, window_days=T)
        print(f"\n=== window_days={T} ===")
        print(f"  X shape     : {X.shape}  -> {X.shape[0]} exemples, {X.shape[1]} features")
        print(f"  y shape     : {y.shape}")
        print(f"  Index[0]    : {X.index[0]}  (premier jour cible)")
        print(f"  Index[-1]   : {X.index[-1]}  (dernier jour cible)")
        print(f"  X.columns[:5]  : {list(X.columns[:5])}")
        print(f"  X.columns[-1]  : {X.columns[-1]}")
        print(f"  y.columns      : {list(y.columns)}")
        print(f"  X.iloc[0, :5]  : {X.iloc[0, :5].values}")
        print(f"  y.iloc[0, :5]  : {y.iloc[0, :5].values}")
"""