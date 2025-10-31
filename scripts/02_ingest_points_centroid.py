# scripts/02_ingest_points_centroid.py
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import re

BASE = Path(__file__).resolve().parents[1]

# ----- INPUT -----
IN_POINTS = BASE / "data" / "raw" / "kejadian" / "data_titik_kasus_2024.csv"
BOUNDARY  = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"

# Sumber penduduk (prioritas -> fallback):
POP_PRIMARY   = BASE / "data" / "processed" / "penduduk_clean_for_join.csv"  # CC_4, penduduk, NAME_4
POP_SECONDARY = BASE / "data" / "raw" / "penduduk" / "penduduk_bersih.xlsx"  # CC_4, NAME_4, PENDUDUK
OLD_AGG       = BASE / "outputs" / "data" / "aggregated_laju_100k.csv"       # fallback terakhir

# ----- OUTPUT -----
OUT_DIR       = BASE / "outputs" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MISSING_TPL   = BASE / "data" / "raw" / "penduduk" / "penduduk_2024_MISSING_TEMPLATE.csv"
MISS_DIAG_CSV = OUT_DIR / "penduduk_missing_diagnostic.csv"
FILL_LOG_CSV  = OUT_DIR / "penduduk_fill_log.csv"

# ---------- Utils normalisasi ----------
def norm_code(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).replace("\ufeff","").replace("\u200b","").strip()
    return re.sub(r"[^\d]", "", s)  # sisakan digit saja

def norm_name_raw(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).upper().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_name(s: str) -> str:
    """Uppercase, rapikan spasi, hilangkan awalan DESA/KEL/KELURAHAN/DS."""
    s0 = norm_name_raw(s)
    for pref in ("DESA ", "KELURAHAN ", "KEL. ", "KEL ", "DS. ", "DS "):
        if s0.startswith(pref):
            s0 = s0[len(pref):].strip()
            break
    return s0

# 1) Baca titik & validasi kolom
df = pd.read_csv(IN_POINTS)
col_lon = next((c for c in ["lon","longitude","x","lng","Lon","Longitude","X"] if c in df.columns), None)
col_lat = next((c for c in ["lat","latitude","y","Lat","Latitude","Y"] if c in df.columns), None)
col_cnt = next((c for c in ["count","kejadian","jumlah","kasus","n","N","jumlah_kejadian"] if c in df.columns), None)
assert all([col_lon, col_lat, col_cnt]), f"Kolom wajib: lon/lat/count. Ada: {df.columns.tolist()}"

df["_lon"] = pd.to_numeric(df[col_lon], errors="coerce")
df["_lat"] = pd.to_numeric(df[col_lat], errors="coerce")
df["_cnt"] = pd.to_numeric(df[col_cnt], errors="coerce").fillna(0).astype(int)
df = df.dropna(subset=["_lon","_lat"]).copy()
print(f"- Titik masuk: {len(df)}")

# 2) Spatial join ke desa (GeoPandas, WGS84)
gpts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["_lon"], df["_lat"]), crs="EPSG:4326")
gadm = gpd.read_file(BOUNDARY)[["CC_4","NAME_4","geometry"]].copy()
gadm["CC_4"]  = gadm["CC_4"].astype(str).apply(norm_code)
gadm["NAME_4"] = gadm["NAME_4"].apply(norm_name)
gadm = gadm[~gadm.geometry.is_empty & gadm.geometry.notnull()].copy()

joined = gpd.sjoin(gpts, gadm, how="left", predicate="within")
n_unmatched = int(joined["CC_4"].isna().sum())
if n_unmatched:
    print(f"PERINGATAN: {n_unmatched} titik di luar batas studi (akan dibuang).")

joined = joined.dropna(subset=["CC_4"]).copy()
joined["CC_4"]   = joined["CC_4"].astype(str).apply(norm_code)
joined["NAME_4"] = joined["NAME_4"].apply(norm_name)
print(f"- Titik terpasang ke desa (within): {len(joined)}")

# 3) Agregasi per desa (pakai nama hasil agregasi -> NAME_4_agg)
agg = (
    joined.groupby(["CC_4","NAME_4"], as_index=False)
          .agg(kejadian_count=("_cnt","sum"))
          .rename(columns={"NAME_4":"NAME_4_agg"})
)
agg["NAME_4_agg_norm"] = agg["NAME_4_agg"].apply(norm_name)
print(f"- Desa unik terisi: {len(agg)}")

# 4) Muat penduduk (CSV -> Excel -> OLD_AGG) + normalisasi
def load_population():
    # A: processed CSV (prioritas)
    if POP_PRIMARY.exists():
        pop = pd.read_csv(POP_PRIMARY, dtype=str)
        if "CC_4" not in pop.columns:
            raise ValueError(f"{POP_PRIMARY.name} tidak punya kolom CC_4")
        if "penduduk" not in pop.columns:
            cand = [c for c in pop.columns if c.lower()=="penduduk"]
            if not cand:
                raise ValueError(f"{POP_PRIMARY.name} tidak punya kolom penduduk")
            pop["penduduk"] = pd.to_numeric(pop[cand[0]], errors="coerce")
        else:
            pop["penduduk"] = pd.to_numeric(pop["penduduk"], errors="coerce")
        pop["CC_4"]   = pop["CC_4"].apply(norm_code)
        pop["NAME_4"] = pop.get("NAME_4","").apply(norm_name)
        print(f"- Sumber penduduk: {POP_PRIMARY.name} (baris: {len(pop)})")
        return pop[["CC_4","penduduk","NAME_4"]], "PRIMARY"

    # B: raw Excel
    if POP_SECONDARY.exists():
        xls = pd.ExcelFile(POP_SECONDARY)
        sheet = xls.sheet_names[0]
        pop = pd.read_excel(xls, sheet_name=sheet, dtype=str)
        if "CC_4" not in pop.columns:
            raise ValueError(f"{POP_SECONDARY.name} tidak punya kolom CC_4")
        cand = [c for c in pop.columns if c.lower()=="penduduk" or c.upper()=="PENDUDUK"]
        if not cand:
            raise ValueError(f"{POP_SECONDARY.name} tidak punya kolom PENDUDUK/penduduk")
        pop["penduduk"] = pd.to_numeric(pop[cand[0]], errors="coerce")
        pop["CC_4"]   = pop["CC_4"].apply(norm_code)
        pop["NAME_4"] = pop.get("NAME_4","").apply(norm_name) if "NAME_4" in pop.columns else ""
        print(f"- Sumber penduduk: {POP_SECONDARY.name} (sheet: {sheet}, baris: {len(pop)})")
        return pop[["CC_4","penduduk","NAME_4"]], "SECONDARY"

    # C: fallback
    assert OLD_AGG.exists(), f"Tidak ada file penduduk; {OLD_AGG} tidak ditemukan."
    pop = pd.read_csv(OLD_AGG, dtype=str)
    assert "CC_4" in pop.columns and "penduduk" in pop.columns, f"{OLD_AGG.name} tidak lengkap."
    pop["penduduk"] = pd.to_numeric(pop["penduduk"], errors="coerce")
    pop["CC_4"]   = pop["CC_4"].apply(norm_code)
    pop["NAME_4"] = pop.get("NAME_4","").apply(norm_name)
    print(f"- Sumber penduduk: {OLD_AGG.name} (baris: {len(pop)})")
    return pop[["CC_4","penduduk","NAME_4"]], "OLD_AGG"

pop, pop_src = load_population()
pop = pop.sort_values(["CC_4"]).drop_duplicates(subset=["CC_4"], keep="first")
pop["NAME_4_norm"] = pop["NAME_4"].apply(norm_name)

# 5) Merge by code + Fallback by name (hanya dari sumber PRIMARY)
aggM = agg.merge(pop[["CC_4","penduduk"]], on="CC_4", how="left", suffixes=("_agg","_pop"))

fill_logs = []
# fallback by NAME_4 (hanya jika sumber PRIMARY tersedia)
if pop_src == "PRIMARY":
    name_map = pop.dropna(subset=["NAME_4_norm","penduduk"]) \
                  .drop_duplicates("NAME_4_norm") \
                  .set_index("NAME_4_norm")["penduduk"]
    need_fill = aggM["penduduk"].isna() & aggM["NAME_4_agg_norm"].isin(name_map.index)
    if need_fill.any():
        aggM.loc[need_fill, "penduduk"] = aggM.loc[need_fill, "NAME_4_agg_norm"].map(name_map)
        for _, r in aggM.loc[need_fill, ["CC_4","NAME_4_agg","penduduk"]].iterrows():
            fill_logs.append({"CC_4": r["CC_4"], "NAME_4": r["NAME_4_agg"],
                              "penduduk_used": r["penduduk"], "method": "NAME_MATCH_PRIMARY"})

# tulis log jika ada
if fill_logs:
    pd.DataFrame(fill_logs).to_csv(FILL_LOG_CSV, index=False, encoding="utf-8")
    print(f"- Isi penduduk via fallback nama: {len(fill_logs)} desa (log: {FILL_LOG_CSV})")

# cek yang masih hilang
missing_mask = aggM["penduduk"].isna()
missing_ppl = int(missing_mask.sum())

if missing_ppl:
    # siapkan template & diagnostik
    tpl = aggM.loc[missing_mask, ["CC_4","NAME_4_agg"]].drop_duplicates().rename(columns={"NAME_4_agg":"NAME_4"})
    MISSING_TPL.parent.mkdir(parents=True, exist_ok=True)
    tpl.assign(penduduk="").to_csv(MISSING_TPL, index=False, encoding="utf-8")

    diags = []
    for _, r in tpl.iterrows():
        nm = norm_name(r["NAME_4"])
        cc = r["CC_4"]
        cand = pop[pop["NAME_4_norm"] == nm]
        if not cand.empty:
            for _, c in cand.iterrows():
                diags.append({"CC_4_missing": cc, "NAME_4_missing": nm,
                              "CC_4_pop": c["CC_4"], "NAME_4_pop": c["NAME_4"], "penduduk_pop": c["penduduk"]})
        else:
            diags.append({"CC_4_missing": cc, "NAME_4_missing": nm,
                          "CC_4_pop": "", "NAME_4_pop": "", "penduduk_pop": ""})
    pd.DataFrame(diags).to_csv(MISS_DIAG_CSV, index=False, encoding="utf-8")

    print("\nERROR: Ada desa tanpa data penduduk. Lengkapi template & lihat diagnostik:")
    for _, r in tpl.iterrows():
        print(f" - CC_4={r['CC_4']} | NAME_4={r['NAME_4']}")
    raise ValueError(
        f"{missing_ppl} desa tidak punya data penduduk. "
        f"Isi {MISSING_TPL.name} atau koreksi kunci sumber penduduk. "
        f"Lihat diagnostik: {MISS_DIAG_CSV.name}"
    )

# 6) Hitung laju & simpan aggregated (pakai NAME_4_agg sebagai nama baku)
agg_final = aggM[["CC_4","NAME_4_agg","kejadian_count","penduduk"]].rename(columns={"NAME_4_agg":"NAME_4"})
agg_final["penduduk"] = pd.to_numeric(agg_final["penduduk"], errors="coerce")
if (agg_final["penduduk"] <= 0).any() or agg_final["penduduk"].isna().any():
    bad = agg_final[(agg_final["penduduk"] <= 0) | (agg_final["penduduk"].isna())][["CC_4","NAME_4","penduduk"]]
    raise ValueError(f"Ditemukan penduduk <=0/NaN di:\n{bad}")

agg_final["laju_100k"] = 100000.0 * agg_final["kejadian_count"] / agg_final["penduduk"]
agg_out = OUT_DIR / "aggregated_laju_100k.csv"
agg_final[["CC_4","NAME_4","kejadian_count","penduduk","laju_100k"]].to_csv(
    agg_out, index=False, encoding="utf-8", float_format="%.6f"
)
print(f"- Simpan: {agg_out}")

# Hitung centroid DI UTM (32749) → akurat
gadm_utm = gadm.to_crs(32749).copy()
cent_utm = gadm_utm.geometry.centroid
gadm_utm["east"]  = cent_utm.x
gadm_utm["north"] = cent_utm.y

# Proyeksikan centroid ke WGS84 untuk lon/lat titik sampel
gadm_cent_wgs = gpd.GeoDataFrame(
    gadm_utm[["CC_4","NAME_4","east","north"]].copy(),
    geometry=cent_utm, crs=32749
).to_crs(4326)
gadm_cent_wgs["lon"] = gadm_cent_wgs.geometry.x
gadm_cent_wgs["lat"] = gadm_cent_wgs.geometry.y

# Gabungkan ke agregat
samples = agg_final.merge(
    gadm_cent_wgs[["CC_4","NAME_4","lat","lon","east","north"]],
    on="CC_4", how="left", suffixes=("","_bnd")
)
samples["value"] = samples["laju_100k"]

samples_out = OUT_DIR / "idw_samples.csv"
samples[["CC_4","NAME_4","lat","lon","east","north","value"]].to_csv(
    samples_out, index=False, encoding="utf-8", float_format="%.6f"
)
print(f"- Simpan: {samples_out}")

# 8) QC ringkas
print("\nQC:")
print(" - Total kejadian (raw)     :", int(df['_cnt'].sum()))
print(" - Total kejadian (per desa):", int(agg_final['kejadian_count'].sum()))
print(" - Duplikat CC_4?           :", bool(agg_final.duplicated('CC_4').any()))
print(" - Nilai laju negatif?      :", bool((agg_final['laju_100k'] < 0).any()))
