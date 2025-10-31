# scripts/17_make_cuplikan_titik.py
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]

SAMP_CSV = BASE / "outputs" / "data" / "idw_samples.csv"
AGG_CSV  = BASE / "outputs" / "data" / "aggregated_laju_100k.csv"
OUT_DIR  = BASE / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV  = OUT_DIR / "cuplikan_titik_2024.csv"

CSV_KEY = "CC_4"
CSV_NAME = "NAME_4"

LAT_CANDS = ["lat","latitude","Lat","Latitude","y","Y"]
LON_CANDS = ["lon","longitude","Lon","Longitude","x","X","lng","Lng"]

def pick_col(cols, cands):
    for c in cands:
        if c in cols:
            return c
    return None

# --- Load
assert SAMP_CSV.exists(), f"Tidak ketemu {SAMP_CSV}"
assert AGG_CSV.exists(), f"Tidak ketemu {AGG_CSV}"

df_s = pd.read_csv(SAMP_CSV, dtype={CSV_KEY: str})
df_a = pd.read_csv(AGG_CSV,  dtype={CSV_KEY: str})

# Normalisasi key/nama
df_s[CSV_KEY] = df_s[CSV_KEY].astype(str).str.strip()
if CSV_NAME in df_s.columns:
    df_s[CSV_NAME] = df_s[CSV_NAME].astype(str).str.strip()

df_a[CSV_KEY] = df_a[CSV_KEY].astype(str).str.strip()
if CSV_NAME in df_a.columns:
    df_a[CSV_NAME] = df_a[CSV_NAME].astype(str).str.strip()

# Deteksi lat/lon & nilai
lat_col = pick_col(df_s.columns, LAT_CANDS)
lon_col = pick_col(df_s.columns, LON_CANDS)
if not lat_col or not lon_col:
    raise KeyError(f"Kolom lat/lon tidak ditemukan di {SAMP_CSV.name}. Ada: {df_s.columns.tolist()}")

val_col = "value" if "value" in df_s.columns else None

# Gabung untuk dapatkan kasus & penduduk & laju (agg)
df = df_s.merge(
    df_a[[c for c in [CSV_KEY, CSV_NAME, "kejadian_count", "penduduk", "laju_100k"] if c in df_a.columns]],
    on=CSV_KEY, how="left", suffixes=("", "_agg")
)

# Rekonsiliasi laju:
# - jika 'laju_100k' (dari agregat) ada → pakai itu, jika tidak pakai 'value' dari sampel
if "laju_100k" in df.columns:
    df["laju_out"] = pd.to_numeric(df["laju_100k"], errors="coerce")
elif val_col:
    df["laju_out"] = pd.to_numeric(df[val_col], errors="coerce")
else:
    # fallback jika keduanya tidak ada (harusnya tidak terjadi)
    df["laju_out"] = np.nan

# Casting numerik & rapikan
df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
if "kejadian_count" in df.columns:
    df["kejadian_count"] = pd.to_numeric(df["kejadian_count"], errors="coerce").astype("Int64")
if "penduduk" in df.columns:
    df["penduduk"] = pd.to_numeric(df["penduduk"], errors="coerce")

# Pilih kolom keluaran
cols_out = [c for c in [CSV_KEY, CSV_NAME] if c in df.columns] + [
    "_lon","_lat","kejadian_count","penduduk","laju_out"
]
df_out = df[cols_out].rename(columns={
    "_lon":"lon",
    "_lat":"lat",
    "laju_out":"laju_100k"
})

# Urut & ambil cuplikan (top-N)
TOP_N = 12
df_out = df_out.sort_values("laju_100k", ascending=False).head(TOP_N).copy()

# Bulatkan tampilan angka (tanpa mengubah tipe terlalu agresif)
df_out["laju_100k"] = df_out["laju_100k"].round(2)
if "penduduk" in df_out.columns:
    df_out["penduduk"] = df_out["penduduk"].round(0)

# Tambah kolom No urut (1..N)
df_out.insert(0, "No", range(1, len(df_out) + 1))

# Simpan
df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"[OK] Tersimpan cuplikan {len(df_out)} baris → {OUT_CSV}")

# Cetak preview 5 baris
print("\nPreview:")
print(df_out.head(5).to_string(index=False))
