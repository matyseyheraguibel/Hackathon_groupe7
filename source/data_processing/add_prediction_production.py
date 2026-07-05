import pandas as pd
df_prod = pd.read_csv("../../data/raw/prediction_production/production_entsoe_pred.csv")
def add_prediction_production(df, df_prod):
    df_prod["date"] = (
    pd.to_datetime(df_prod["date"])
      .dt.tz_localize(
          "Europe/Paris",
          nonexistent="shift_forward",  # ou "raise", "NaT"
          ambiguous="infer"             # ou True/False selon les données
      )
)
    df_prod = df_prod.set_index("date").sort_index()
    
    df_concat = df.join(df_prod, how="outer")

    df_concat = df_concat.dropna()
    return df_concat