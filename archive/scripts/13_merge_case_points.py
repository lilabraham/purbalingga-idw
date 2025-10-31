# scripts/13_merge_case_points.py
from pathlib import Path
import json, math
import pandas as pd
import numpy as np
from shapely.geometry import shape, Point
from shapely.prepared import prep
from shapely.ops import unary_union
from pyproj import Transformer

BASE = Path(__file__).resolve().parents[1]

# INPUT
BOUNDARY_GJ = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"
MASTER_CSV  = BASE / "data" / "raw" / "kejadian" / "data_titik_kasus_2024.csv"   # file utama (lama)
NEW_ENRICH  = BASE / "outputs" / "data" / "kejadian_enriched.csv"                # data baru hasil 12_enrich...

# OUTPUT (backup & arsip)
OUT_DIR     = BASE / "outputs" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_CSV  = OUT_DIR / "data_titik_kasus_2024_backup.csv"
MERGED_CSV  = OUT_DIR / "data_titik_kasus_2024_merged_preview.csv"  # pratinjau hasil gabung

# Konfigurasi
ROUND_DECIMALS = 6   # pembulatan koordinat untuk deduplikasi
FALLBACK_RADIUS_M = 200.0

# Kunci properti yang umum di boundary
KEY_DESA   = ["NAME_4","DESA","DESA_KEL","NAMDESA","nama_desa","nama","desa_kel"]
KEY_KODE   = ["CC_4","KODE","KODE_DESA","KD_DESA","kode_desa","cc_4"]
KEY_KEC    = ["NAME_3","KECAMATAN","kecamatan","NAMKEC","nama_kecamatan"]

def pick(props: dict, keys: list):
    for k in keys:
        if k in props and props[k] not in (None, "", "None", 0):
            return props[k]
    return None

def detect_cols(df: pd.DataFrame):
    lon_col = next((c for c in df.columns if c.lower() in ("lon","longitude","lng","x")), None)
    lat_col = next((c for c in df.columns if c.lower() in ("lat","latitude","y")), None)
    cnt_col = next((c for c in df.columns if c.lower() in ("count","jumlah","jumlah_kasus")), None)
    cc_col  = next((c for c in df.columns if c.upper() == "CC_4" or c.lower() == "cc_4"), None)
    nm_col  = next((c for c in df.columns if c in ("NAME_4","DESA","DESA_KEL","NAMDESA","nama_desa","nama","desa_kel")), None)
    kec_col = next((c for c in df.columns if c in ("NAME_3","KECAMATAN","kecamatan","NAMKEC","nama_kecamatan")), None)
    return lon_col, lat_col, cnt_col, cc_col, nm_col, kec_col

def load_boundary():
    assert BOUNDARY_GJ.exists(), f"Tidak ketemu boundary: {BOUNDARY_GJ}"
    gj = json.loads(BOUNDARY_GJ.read_text(encoding="utf-8"))
    feats = gj.get("features", [])
    if not feats:
        raise RuntimeError("GeoJSON boundary tidak punya 'features'.")
    geoms = []
    for ft in feats:
        geom = shape(ft["geometry"])
        props = ft.get("properties", {})
        geoms.append({
            "geom": geom,
            "prep": prep(geom),
            "desa": str(pick(props, KEY_DESA) or ""),
            "kode": str(pick(props, KEY_KODE) or ""),
            "kec":  str(pick(props, KEY_KEC)  or "")
        })
    return geoms

def enrich_missing_admin(df: pd.DataFrame, geoms: list, lon_col: str, lat_col: str):
    """Isi CC_4/NAME_4/KECAMATAN untuk baris yang masih kosong."""
    wgs_to_utm49s = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)
    # Precompute centroid UTM utk fallback
    centers = []
    for g in geoms:
        cx, cy = g["geom"].centroid.x, g["geom"].centroid.y
        ex, ny = wgs_to_utm49s.transform(cx, cy)
        centers.append((ex, ny))

    def _find(lon, lat):
        pt = Point(lon, lat)
        # tahap 1: covers
        for g in geoms:
            if g["prep"].covers(pt):
                return g["kode"], g["desa"], g["kec"], "covers"
        # tahap 2: intersects
        for g in geoms:
            if g["prep"].intersects(pt):
                return g["kode"], g["desa"], g["kec"], "intersects"
        # tahap 3: nearest centroid <= 200 m
        ex, ny = wgs_to_utm49s.transform(lon, lat)
        dists = [math.hypot(ex-cx, ny-cy) for (cx,cy) in centers]
        idx = int(np.argmin(dists))
        if dists[idx] <= FALLBACK_RADIUS_M:
            g = geoms[idx]
            return g["kode"], g["desa"], g["kec"], f"nearest<={int(FALLBACK_RADIUS_M)}m"
        return "", "", "", "unmatched"

    # siapkan kolom jika belum ada
    for col in ["CC_4","NAME_4","KECAMATAN","_match"]:
        if col not in df.columns:
            df[col] = ""

    needs = df[(df["CC_4"]=="") | (df["NAME_4"]=="")]
    if not needs.empty:
        idxs = needs.index.tolist()
        for i in idxs:
            lon = float(df.at[i, lon_col]); lat = float(df.at[i, lat_col])
            kode, desa, kec, how = _find(lon, lat)
            if df.at[i, "CC_4"] == "": df.at[i, "CC_4"] = kode
            if df.at[i, "NAME_4"] == "": df.at[i, "NAME_4"] = desa
            if df.at[i, "KECAMATAN"] == "": df.at[i, "KECAMATAN"] = kec
            df.at[i, "_match"] = how
    return df

def unify_df(df: pd.DataFrame):
    lon_col, lat_col, cnt_col, cc_col, nm_col, kec_col = detect_cols(df)

    assert lon_col and lat_col, f"Kolom lon/lat tidak ditemukan. Ada: {df.columns.tolist()}"
    if cnt_col is None:
        df["count"] = 1
        cnt_col = "count"

    # type & clamp
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[cnt_col] = pd.to_numeric(df[cnt_col], errors="coerce").fillna(1).astype(int)

    out = pd.DataFrame({
        "lon": df[lon_col],
        "lat": df[lat_col],
        "count": df[cnt_col],
        "CC_4": df[cc_col] if cc_col else "",
        "NAME_4": df[nm_col] if nm_col else "",
        "KECAMATAN": df[kec_col] if kec_col else "",
    })
    # pastikan string
    for col in ["CC_4","NAME_4","KECAMATAN"]:
        if col in out.columns:
            out[col] = out[col].astype(str).fillna("")

    # drop NA di lon/lat
    out = out.dropna(subset=["lon","lat"]).copy()

    return out

def merge_and_dedupe(df_old: pd.DataFrame, df_new: pd.DataFrame):
    # gabung semua
    all_df = pd.concat([df_old, df_new], ignore_index=True)

    # bundarkan koordinat utk dedupe stabil
    all_df["lon_r"] = all_df["lon"].round(ROUND_DECIMALS)
    all_df["lat_r"] = all_df["lat"].round(ROUND_DECIMALS)

    # group
    grp_cols = ["lon_r","lat_r","CC_4","NAME_4","KECAMATAN"]
    # Kalau CC_4/NAME_4 kosong pada sebagian, biar grup tetap nyatu:
    all_df["CC_4"] = all_df["CC_4"].replace("nan","").fillna("")
    all_df["NAME_4"] = all_df["NAME_4"].replace("nan","").fillna("")
    all_df["KECAMATAN"] = all_df["KECAMATAN"].replace("nan","").fillna("")

    agg = (all_df
           .groupby(grp_cols, dropna=False, as_index=False)
           .agg(count=("count","sum"),
                lon=("lon","mean"),  # rata-rata kecil kalau ada variasi 1e-6
                lat=("lat","mean"))
          )

    # bersihkan kolom bantu
    agg = agg.drop(columns=["lon_r","lat_r"], errors="ignore")

    # urutkan rapi
    agg = agg[["lon","lat","count","CC_4","NAME_4","KECAMATAN"]]

    # tipe
    agg["count"] = agg["count"].astype(int)

    return agg

def main():
    assert MASTER_CSV.exists(), f"Tidak ketemu file utama: {MASTER_CSV}"
    assert NEW_ENRICH.exists(), f"Tidak ketemu data baru: {NEW_ENRICH}"

    # load boundary utk enrich missing
    geoms = load_boundary()

    # baca lama & baru → seragamkan kolom
    old_df_raw = pd.read_csv(MASTER_CSV, encoding="utf-8")
    new_df_raw = pd.read_csv(NEW_ENRICH, encoding="utf-8")

    old_df = unify_df(old_df_raw)
    new_df = unify_df(new_df_raw)

    # lengkapi admin utk baris yang kosong
    old_df = enrich_missing_admin(old_df, geoms, "lon", "lat")
    new_df = enrich_missing_admin(new_df, geoms, "lon", "lat")

    # gabung & dedupe
    merged = merge_and_dedupe(old_df, new_df)

    # simpan pratinjau (aman)
    merged.to_csv(MERGED_CSV, index=False, encoding="utf-8")

    # backup lama, lalu overwrite file utama
    try:
        old_df_raw.to_csv(BACKUP_CSV, index=False, encoding="utf-8")
        print(f"Backup file lama → {BACKUP_CSV}")
    except Exception as e:
        print("Gagal buat backup (lanjut overwrite):", e)

    merged.to_csv(MASTER_CSV, index=False, encoding="utf-8")
    print(f"Master diupdate → {MASTER_CSV}")
    print(f"Pratinjau merged → {MERGED_CSV}")
    print("Ringkasan:")
    print(f"- Old rows : {len(old_df)}")
    print(f"- New rows : {len(new_df)}")
    print(f"- Final    : {len(merged)} (setelah dedupe + sum count)")
    # cek yang masih kosong admin:
    n_empty = int(((merged["CC_4"]=="") | (merged["NAME_4"]=="")).sum())
    if n_empty>0:
        print(f"⚠️  {n_empty} baris belum punya admin (di luar batas atau koordinat bermasalah)")
    else:
        print("✅ Semua baris punya CC_4/NAME_4.")

if __name__ == "__main__":
    main()
