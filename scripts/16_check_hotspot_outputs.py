# scripts/16_check_hotspot_outputs.py
from pathlib import Path
import json
import pandas as pd
import numpy as np
import re

BASE = Path(__file__).resolve().parents[1]

CAND_CSV = [
    BASE / "outputs" / "tables" / "hotspot_patches_p95.csv",
    BASE / "outputs" / "tables" / "hotspots_p95.csv",
]
CAND_GJ = [
    BASE / "outputs" / "data" / "hotspots" / "hotspot_patches_p95.geojson",
    BASE / "outputs" / "data" / "hotspots" / "hotspots_p95.geojson",
]
META_PATH = BASE / "outputs" / "data" / "rasters" / "idw_surface_meta.json"

def pick_first(paths):
    return next((p for p in paths if p.exists()), None)

CSV_PATH = pick_first(CAND_CSV)
GJ_PATH  = pick_first(CAND_GJ)

assert CSV_PATH, "CSV hotspot tidak ditemukan."
assert GJ_PATH,  "GeoJSON hotspot tidak ditemukan."

def load_pixel_area_m2(meta_path: Path):
    if not meta_path.exists():
        print(f"[INFO] Meta tidak ditemukan: {meta_path.name}")
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        g = float(meta.get("grid_resolution_m", 0) or 0)
        return g*g if g > 0 else None
    except Exception as e:
        print(f"[INFO] Gagal baca meta: {e}")
        return None

def infer_percentile_from_filename(p: Path):
    m = re.search(r"p(\d{2})", p.name.lower())
    return float(m.group(1)) if m else None

def normalize(df: pd.DataFrame, pixel_area_m2: float | None, csv_path: Path):
    print(f"[INFO] Kolom CSV: {list(df.columns)}")

    # patch_id
    if "patch_id" not in df.columns:
        for alt in ["id","patch","cluster_id","patchID"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"patch_id"})
                break
    if "patch_id" not in df.columns:
        raise ValueError("CSV tidak punya 'patch_id' atau aliasnya.")

    # area -> km2
    if "area_km2" not in df.columns and "area_ha" in df.columns:
        df["area_km2"] = pd.to_numeric(df["area_ha"], errors="coerce") * 0.01

    # n_cells
    if "n_cells" not in df.columns:
        for alt in ["n_pixels","n_pix","pixel_count","cells","ncell","nCells"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"n_cells"})
                break
    if "n_cells" not in df.columns and ("area_km2" in df.columns) and (pixel_area_m2 is not None):
        df["n_cells"] = np.floor((pd.to_numeric(df["area_km2"], errors="coerce")*1_000_000)/pixel_area_m2 + 0.5).astype("Int64")
        print("[INFO] 'n_cells' dihitung dari area_km2 & meta.")

    # mean/max
    if "value_mean" not in df.columns:
        for alt in ["mean_laju","mean_val","mean","avg_value","avg"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"value_mean"})
                break
    if "value_max" not in df.columns:
        for alt in ["max_laju","max_val","max","max_value"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"value_max"})
                break

    # threshold & percentile
    if "threshold" not in df.columns:
        for alt in ["threshold_laju","threshold_value","cutoff","thr"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"threshold"})
                break
    if "percentile" not in df.columns:
        pct = infer_percentile_from_filename(csv_path)
        if pct:
            df["percentile"] = pct

    # centroid
    if "centroid_lon" not in df.columns:
        for alt in ["lon","x","centroid_x"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"centroid_lon"})
                break
    if "centroid_lat" not in df.columns:
        for alt in ["lat","y","centroid_y"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"centroid_lat"})
                break

    # n_desa & join
    if "n_desa" not in df.columns:
        for alt in ["n_units","n_polygons","n_admin","n_villages"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"n_desa"})
                break
    if "desa_joined" not in df.columns:
        for alt in ["names","labels","desa"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"desa_joined"})
                break

    # numerikkan kolom
    for c in ["n_cells","area_km2","value_mean","value_max","threshold","percentile","centroid_lon","centroid_lat","n_desa"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# ==== RUN ====
df = pd.read_csv(CSV_PATH)
pixel_area_m2 = load_pixel_area_m2(META_PATH)
df = normalize(df, pixel_area_m2, CSV_PATH)

gj = json.loads(GJ_PATH.read_text(encoding="utf-8"))
features = gj.get("features", [])
print(f"CSV rows      : {len(df)}")
print(f"GeoJSON feats : {len(features)}")

# bbox & id
min_lon=1e9; min_lat=1e9; max_lon=-1e9; max_lat=-1e9; gj_ids=[]
def upd_bbox(coords):
    if not coords: return
    if isinstance(coords[0][0][0], (float,int)):
        rings=coords
    else:
        rings=[ring for poly in coords for ring in poly]
    global min_lon,min_lat,max_lon,max_lat
    for ring in rings:
        for x,y in ring:
            min_lon=min(min_lon,x); max_lon=max(max_lon,x)
            min_lat=min(min_lat,y); max_lat=max(max_lat,y)

types={}
for ft in features:
    pr=ft.get("properties",{}) or {}
    pid=str(pr.get("patch_id")) if pr.get("patch_id") is not None else None
    if pid: gj_ids.append(pid)
    geom=ft.get("geometry",{}) or {}
    types[geom.get("type","Unknown")] = types.get(geom.get("type","Unknown"),0)+1
    upd_bbox(geom.get("coordinates"))
print(f"Jenis geometri di GeoJSON: {types}")
print(f"BBox semua fitur (lon/lat): W={min_lon:.6f}, S={min_lat:.6f}, E={max_lon:.6f}, N={max_lat:.6f}")

only_csv=set(df["patch_id"].astype(str)) - set(gj_ids)
only_gj = set(gj_ids) - set(df["patch_id"].astype(str))
if only_csv: print(f"[PERINGATAN] Ada patch_id di CSV tapi tidak di GeoJSON: {sorted(list(only_csv))[:10]}...")
if only_gj:  print(f"[PERINGATAN] Ada patch_id di GeoJSON tapi tidak di CSV: {sorted(list(only_gj))[:10]}...")
if (len(df)==len(features)) and (not only_csv) and (not only_gj):
    print("✔ Jumlah & mapping patch_id CSV ↔ GeoJSON konsisten.")

# sanity check
if "n_cells" in df.columns:
    bad=(pd.to_numeric(df["n_cells"],errors="coerce")<=0).sum()
    print(f"Baris n_cells ≤ 0: {bad} (harusnya 0)")
if {"value_mean","value_max"}.issubset(df.columns):
    bad=(pd.to_numeric(df["value_max"],errors="coerce")<pd.to_numeric(df["value_mean"],errors="coerce")).sum()
    print(f"Baris value_max < value_mean: {bad} (harusnya 0)")

# ===== Export tabel ringkas untuk laporan =====
for_paper_cols = [c for c in [
    "patch_id","n_cells","area_km2","value_mean","value_max",
    "n_desa","centroid_lon","centroid_lat","desa_joined","threshold","percentile"
] if c in df.columns]
df_out = df[for_paper_cols].copy()

if "area_km2" in df_out.columns: df_out["area_km2"] = df_out["area_km2"].round(3)
for c in ["value_mean","value_max","threshold"]:
    if c in df_out.columns: df_out[c] = df_out[c].round(2)

OUT_STD = CSV_PATH.with_name("hotspot_patches_p95_normalized.csv")
df.to_csv(OUT_STD, index=False, encoding="utf-8", float_format="%.4f")
OUT_PAPER = CSV_PATH.with_name("hotspot_patches_p95_for_paper.csv")
df_out.to_csv(OUT_PAPER, index=False, encoding="utf-8")

print(f"\n[INFO] CSV terstandar  : {OUT_STD}")
print(f"[INFO] Tabel utk laporan: {OUT_PAPER}")
print("\nSelesai cek.")
