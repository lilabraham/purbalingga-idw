# scripts/03_qc_missing_in_population.py
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE/"data/raw/kejadian/data_titik_kasus_2024.csv"
POP = BASE/"data/processed/penduduk_clean_for_join.csv"

raw = pd.read_csv(RAW, dtype={"CC_4": str})
pop = pd.read_csv(POP, dtype={"CC_4": str})

tot_raw = int(pd.to_numeric(raw["count"], errors="coerce").fillna(0).sum())
print("Total count (raw):", tot_raw)

# group kejadian per CC_4 dari raw
g = (raw.dropna(subset=["CC_4"])
        .query("CC_4 != ''")
        .groupby(["CC_4","NAME_4"], as_index=False)["count"].sum()
        .sort_values("count", ascending=False))

# CC_4 yang tidak ada di tabel penduduk
missing = g[~g["CC_4"].isin(pop["CC_4"])]
print("\nCC_4 di data kasus TIDAK ADA di tabel penduduk (akan hilang saat merge pop->counts):")
print(missing.to_string(index=False) if not missing.empty else "— tidak ada —")

print("\nRingkas:")
print("- jumlah CC_4 unik di RAW :", g['CC_4'].nunique())
print("- jumlah CC_4 di PENDUDUK :", pop['CC_4'].nunique())
print("- CC_4 raw yg hilang di penduduk :", len(missing))
print("- total count pada CC_4 yg hilang :", int(missing['count'].sum()) if not missing.empty else 0)
