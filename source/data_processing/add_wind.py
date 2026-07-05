import pandas as pd
df_wind = pd.read_csv("../../data/raw/vent/wind_entsoe_pred.csv")
def add_wind(df, df_wind):
    df_wind["date"] = (
    pd.to_datetime(df_wind["date"])
      .dt.tz_localize(
          "Europe/Paris",
          nonexistent="shift_forward",  # ou "raise", "NaT"
          ambiguous="infer"             # ou True/False selon les données
      )
)
    df_wind = df_wind.set_index("date").sort_index()
    df_concat = df.join(df_wind, how="outer")

    df_concat = df_concat.dropna()
    return df_concat