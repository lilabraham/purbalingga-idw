# scripts/03_qc_results.py
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
AGG  = BASE / "outputs" / "data" / "aggregated_laju_100k.csv"
INC  = BASE / "data" / "raw" / "kejadian" / "data_titik_kasus_2024.csv"

assert AGG.exists(), f"Tidak ketemu: {AGG}"
agg = pd.read_csv(AGG)

print("== QA aggregated_laju_100k ==")
print("Baris (desa):", len(agg))
print("Kolom:", list(agg.columns))
print("Total kejadian (sum):", int(agg["kejadian_count"].sum()))
print("Desa penduduk=0:", int((agg["penduduk"] == 0).sum()))
print("Desa laju>0:", int((agg["laju_100k"] > 0).sum()))
print("\nTop 5 laju per 100k:")
print(agg.sort_values("laju_100k", ascending=False).head(5)[["CC_4","NAME_4","penduduk","kejadian_count","laju_100k"]])

print("\nBottom 5 laju per 100k (penduduk>0):")
print(agg[agg["penduduk"]>0].sort_values("laju_100k", ascending=True).head(5)[["CC_4","NAME_4","penduduk","kejadian_count","laju_100k"]])
