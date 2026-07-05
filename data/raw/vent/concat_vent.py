import pandas as pd

df = pd.concat(
    [pd.read_csv("wind2020.csv"), pd.read_csv("wind2021.csv"), pd.read_csv("wind2022.csv"), pd.read_csv("wind2023.csv"), pd.read_csv("wind2024.csv"), pd.read_csv("wind2025.csv"), pd.read_csv("wind2026.csv")],
    ignore_index=True
)

df = df.drop(["Area", "Day-ahead (MW)", "Actual (MW)", "Intraday (MW)"], axis=1)

df = df.rename(columns={"MTU (CET/CEST)": "date"})
df = df.rename(columns={"Current (MW)": "wind_prediction(MW)"})

df["date"] = df["date"].str.split(" - ").str[0]
df["date"] = df["date"].str.replace(r"\s*\(.*\)", "", regex=True)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)


df["wind_prediction(MW)"] = pd.to_numeric(
    df["wind_prediction(MW)"],
    errors="coerce"
)    



#couper au 30 juin
df['date'] = pd.to_datetime(df['date'])
cutoff = pd.Timestamp("2026-06-30 22:00:00")
df_cut = df[df['date'] < cutoff]

df_cut.to_csv("wind_entsoe_pred.csv", index=False)