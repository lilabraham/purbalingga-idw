# scripts/make_idw_excel_dataset.py
from pathlib import Path
import re
import pandas as pd
import geopandas as gpd

BASE = Path(".")
P_BND = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"
P_PTS = BASE / "data" / "raw" / "kejadian" / "data_titik_kasus_2024.csv"
P_POP = BASE / "data" / "processed" / "penduduk_clean_for_join.csv"
P_OUT = BASE / "outputs" / "excel" / "data_idw_2024.csv"
P_OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------- Helper kolom ----------
def pick(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
        for cc in df.columns:  # case-insensitive
            if cc.lower() == c.lower():
                return cc
    if required:
        raise ValueError(f"Kolom tidak ditemukan. Harus salah satu dari: {candidates}")
    return None

# ---------- Normalisasi kunci CC_4 ----------
def only_digits(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\D", "", regex=True)

def norm_cc4_series(s: pd.Series, width: int) -> pd.Series:
    return only_digits(s).str.zfill(width)

# ---------- 1) Boundary desa ----------
gdf_desa = gpd.read_file(P_BND)
if gdf_desa.crs is None:
    # asumsi WGS84 jika hilang
    gdf_desa.set_crs(4326, inplace=True)

id_desa_col = pick(gdf_desa, ["CC_4"])
gdf_desa = gdf_desa[[id_desa_col, "geometry"]].rename(columns={id_desa_col: "CC_4"})

# Tentukan lebar standar CC_4 dari boundary (paling aman)
std_width = int(only_digits(gdf_desa["CC_4"]).str.len().mode().iat[0])
gdf_desa["CC_4"] = norm_cc4_series(gdf_desa["CC_4"], std_width)

# Centroid UTM49S (meter)
gdf_desa_utm = gdf_desa.to_crs(32749)
gdf_cent = gdf_desa_utm.copy()
gdf_cent["X_UTM"] = gdf_cent.geometry.centroid.x
gdf_cent["Y_UTM"] = gdf_cent.geometry.centroid.y
centroids = gdf_cent[["CC_4", "X_UTM", "Y_UTM"]].copy()
centroids["CC_4"] = norm_cc4_series(centroids["CC_4"], std_width)

# ---------- 2) Titik kejadian ----------
df_pts = pd.read_csv(P_PTS)
lon_col   = pick(df_pts, ["lon","Lon","longitude","Longitude","LON"])
lat_col   = pick(df_pts, ["lat","Lat","latitude","Latitude","LAT"])
count_col = pick(df_pts, ["count","jumlah","kejadian","Jumlah_Kasus","CASE_COUNT","kasus"])

gdf_pts = gpd.GeoDataFrame(
    df_pts,
    geometry=gpd.points_from_xy(df_pts[lon_col], df_pts[lat_col]),
    crs=4326
)

# Join titik→desa (within) dan agregasi jumlah kasus per desa (2024)
joined = gpd.sjoin(gdf_pts[[count_col, "geometry"]], gdf_desa[["CC_4", "geometry"]],
                   predicate="within", how="inner")

agg = (
    joined.groupby("CC_4", as_index=False)[count_col].sum()
          .rename(columns={count_col: "Jumlah_Kasus"})
)
agg["CC_4"] = norm_cc4_series(agg["CC_4"], std_width)
agg["Jumlah_Kasus"] = pd.to_numeric(agg["Jumlah_Kasus"], errors="coerce").fillna(0)

# ---------- 3) Penduduk ----------
df_pop   = pd.read_csv(P_POP)
cc4_col  = pick(df_pop, ["CC_4"])
pop_col  = pick(df_pop, ["penduduk","Penduduk","POP","Pop","jumlah_penduduk"])

pop = df_pop[[cc4_col, pop_col]].rename(columns={cc4_col: "CC_4", pop_col: "Jumlah_Populasi"})
pop["CC_4"] = norm_cc4_series(pop["CC_4"], std_width)
pop["Jumlah_Populasi"] = pd.to_numeric(pop["Jumlah_Populasi"], errors="coerce")

# ---------- 4) Merge & hitung laju ----------
# Debug kecil kalau lebar/format masih beda
# print("LEN agg:", agg["CC_4"].str.len().value_counts().head(),
#       "\nLEN pop:", pop["CC_4"].str.len().value_counts().head())

tab = agg.merge(pop, on="CC_4", how="left")
tab = tab.loc[(tab["Jumlah_Populasi"] > 0) & tab["Jumlah_Populasi"].notna()].copy()
tab["Laju_per_100k"] = (tab["Jumlah_Kasus"] / tab["Jumlah_Populasi"]) * 100000.0

# ---------- 5) Tambahkan koordinat centroid ----------
out = (
    tab.merge(centroids, on="CC_4", how="left")
       .rename(columns={"CC_4": "ID_Titik"})
       [["ID_Titik", "X_UTM", "Y_UTM", "Jumlah_Kasus", "Jumlah_Populasi", "Laju_per_100k"]]
       .sort_values("ID_Titik")
)

# ---------- 6) Simpan ----------
out.to_csv(P_OUT, index=False)
print(f"Selesai → {P_OUT}")
print(out.head(8))
print(f"\nRingkas: desa valid = {len(out)}, total_kasus = {int(out['Jumlah_Kasus'].sum())}")
