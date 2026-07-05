import pandas as pd

df = pd.concat(
    [pd.read_csv("pred_2020.csv"), pd.read_csv("pred_2021.csv"), pd.read_csv("pred_2022.csv"), pd.read_csv("pred_2023.csv"), pd.read_csv("pred_2024.csv"), pd.read_csv("pred_2025.csv"), pd.read_csv("pred_2026.csv")],
    ignore_index=True
)

df = df.drop(["Area", "Actual Total Load (MW)"], axis=1)

df = df.rename(columns={"MTU (CET/CEST)": "date"})
df["date"] = df["date"].str.split(" - ").str[0]
df["date"] = df["date"].str.replace(r"\s*\(.*\)", "", regex=True)
df["date"] = pd.to_datetime(df["date"], dayfirst=True)


df["Day-ahead Total Load Forecast (MW)"] = pd.to_numeric(
    df["Day-ahead Total Load Forecast (MW)"],
    errors="coerce"
)    



#couper au 30 juin
df['date'] = pd.to_datetime(df['date'])
cutoff = pd.Timestamp("2026-06-30 22:00:00")
df_cut = df[df['date'] < cutoff]

df_cut.to_csv("production_entsoe_pred.csv", index=False)