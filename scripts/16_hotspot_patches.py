# -*- coding: utf-8 -*-
"""
Membuat ringkasan patch hotspot dari raster IDW dengan pembersihan morfologi (opsional).
- Ambang default: persentil 95 (P95) dari nilai raster valid
- Morphological opening via SciPy (opsional)
- Patch kecil disaring dengan MIN_PIXELS
Output:
  - outputs/tables/hotspot_patches_p95.csv
  - outputs/data/hotspots/hotspot_patches_p95.geojson   (utama)
  - outputs/data/hotspots/hotspots_p95.geojson          (alias kompatibilitas)
"""

from pathlib import Path
import json
import csv
import numpy as np

import rasterio
from rasterio.features import shapes, rasterize
from shapely.geometry import shape, mapping
from shapely.ops import transform
from pyproj import Transformer

# ---- Morphological opening (opsional) ----
try:
    from scipy.ndimage import binary_opening, generate_binary_structure, label
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ---------------- Konfigurasi ----------------
BASE = Path(__file__).resolve().parents[1]

# Input
TIF_PATH   = BASE / "outputs" / "data" / "rasters" / "idw_surface_wgs84.tif"
GJ_BOUND   = BASE / "data"   / "raw"  / "boundary" / "desa_kasus.geojson"

# Output
OUT_TAB_DIR = BASE / "outputs" / "tables"
OUT_GEO_DIR = BASE / "outputs" / "data" / "hotspots"
OUT_TAB_DIR.mkdir(parents=True, exist_ok=True)
OUT_GEO_DIR.mkdir(parents=True, exist_ok=True)

# Ambang hotspot:
THRESH_MODE = "pctl"   # "pctl" atau "abs"
THRESH_VAL  = 95.0     # pctl: 90/95/97.5/99 | abs: nilai laju (mis. 40.0)

# Parameter lain
CONNECTIVITY = 8       # 4 atau 8
OPENING_ITER = 1       # 0 = nonaktifkan opening
MIN_PIXELS   = 9       # minimal ukuran patch (dalam piksel)
AREA_EPSG    = "EPSG:32749"  # untuk kalkulasi luas (UTM 49S)
# --------------------------------------------

def load_villages(geojson_path: Path):
    gj = json.loads(geojson_path.read_text(encoding="utf-8"))
    feats = []
    for ft in gj.get("features", []):
        props = ft.get("properties", {}) or {}
        nm = (props.get("NAME_4") or props.get("DESA") or props.get("NAMDESA") or "").strip()
        cc = str(props.get("CC_4", "")).strip()
        geom = shape(ft.get("geometry"))
        feats.append({"name": nm if nm else cc, "cc": cc, "geom": geom})
    return feats

def reproject_geom(geom, src="EPSG:4326", dst=AREA_EPSG):
    tfm = Transformer.from_crs(src, dst, always_xy=True).transform
    return transform(tfm, geom)

def polygonize_mask(mask, transform_affine, connectivity=8):
    for gjson, val in shapes(mask.astype(np.uint8), transform=transform_affine, connectivity=connectivity):
        if val == 1:
            geom = shape(gjson)
            try:
                if not geom.is_valid:
                    geom = geom.buffer(0)
            except Exception:
                pass
            yield geom

def zonal_stats_for_geom(geom, arr, transform_affine, nodata):
    geom_list = [mapping(geom)]
    msk = rasterize(
        geom_list,
        out_shape=arr.shape,
        transform=transform_affine,
        fill=0,
        all_touched=False,
        dtype="uint8"
    ).astype(bool)

    vals = arr[msk]
    if nodata is not None:
        vals = vals[vals != nodata]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0, float("nan"), float("nan")
    return vals.size, float(np.nanmean(vals)), float(np.nanmax(vals))

def villages_touching(geom, villages):
    touched = []
    for v in villages:
        try:
            if geom.intersects(v["geom"]):
                touched.append(v["name"] if v["name"] else v["cc"])
        except Exception:
            continue
    return sorted(list(dict.fromkeys(touched))), None

def main():
    assert TIF_PATH.exists(), f"TIF tidak ditemukan: {TIF_PATH}"
    assert GJ_BOUND.exists(), f"GeoJSON batas tidak ditemukan: {GJ_BOUND}"

    villages = load_villages(GJ_BOUND)

    with rasterio.open(TIF_PATH) as ds:
        arr = ds.read(1)
        nod = ds.nodata
        transform_affine = ds.transform

    valid = np.isfinite(arr)
    if nod is not None:
        valid &= (arr != nod)
    vals = arr[valid]
    if vals.size == 0:
        raise RuntimeError("Raster tidak memiliki nilai valid.")

    # Threshold
    if THRESH_MODE == "pctl":
        thr = float(np.percentile(vals, THRESH_VAL))
        suffix = f"p{int(THRESH_VAL)}"
    elif THRESH_MODE == "abs":
        thr = float(THRESH_VAL)
        suffix = f"abs{str(THRESH_VAL).replace('.','_')}"
    else:
        raise ValueError("THRESH_MODE harus 'pctl' atau 'abs'.")

    print(f"[INFO] Ambang hotspot = {THRESH_MODE.upper()} {THRESH_VAL} -> nilai laju >= {thr:.2f}")

    # Mask hotspot awal
    hot_mask_raw = valid & (arr >= thr)

    # Opening (opsional)
    if HAS_SCIPY and OPENING_ITER > 0:
        sci_conn = 2 if CONNECTIVITY == 8 else 1
        selem = generate_binary_structure(2, sci_conn)
        _, n_raw = label(hot_mask_raw, structure=selem)
        hot_mask = binary_opening(hot_mask_raw, structure=selem, iterations=OPENING_ITER)
        _, n_clean = label(hot_mask, structure=selem)
        print(f"[INFO] SciPy opening: komponen sebelum={n_raw}, sesudah={n_clean}, "
              f"piksel True sebelum={int(hot_mask_raw.sum())}, sesudah={int(hot_mask.sum())}")
        morph_note = f"opening(conn={CONNECTIVITY}, iter={OPENING_ITER})"
    else:
        hot_mask = hot_mask_raw
        print(f"[INFO] Morphological opening dilewati (SciPy={'ada' if HAS_SCIPY else 'tidak ada'}, iter={OPENING_ITER}).")
        morph_note = "opening=off"

    patches = list(polygonize_mask(hot_mask, transform_affine, connectivity=CONNECTIVITY))
    if not patches:
        print("[INFO] Tidak ada patch hotspot pada ambang ini.")
        return

    # Bangun fitur & baris CSV
    rows, features = [], []
    for idx, geom in enumerate(patches, start=1):
        try:
            n_px, mean_v, max_v = zonal_stats_for_geom(geom, arr, transform_affine, nod)
            if n_px < MIN_PIXELS:
                continue

            geom_m = reproject_geom(geom)
            area_ha = float(geom_m.area) / 10000.0
            area_km2 = area_ha * 0.01

            c = geom.centroid
            cen_lon, cen_lat = float(c.x), float(c.y)

            desa_list, _ = villages_touching(geom, villages)
            patch_id = f"H{idx}"

            prop = {
                "patch_id": patch_id,
                "threshold_mode": THRESH_MODE,
                "threshold_value": THRESH_VAL,
                "threshold_laju": round(thr, 2),
                "morph": morph_note,

                # nama asli
                "n_pixels": int(n_px),
                "area_ha": round(area_ha, 2),
                "mean_laju": round(mean_v, 2) if np.isfinite(mean_v) else None,
                "max_laju":  round(max_v, 2) if np.isfinite(max_v) else None,
                "centroid_lon": round(cen_lon, 6),
                "centroid_lat": round(cen_lat, 6),
                "n_desa": len(desa_list),
                "desa_joined": ", ".join(desa_list),

                # alias kompatibilitas
                "area_km2": round(area_km2, 3),
                "value_mean": round(mean_v, 2) if np.isfinite(mean_v) else None,
                "value_max":  round(max_v, 2) if np.isfinite(max_v) else None,
                "n_villages": len(desa_list),
            }

            rows.append(prop)
            features.append({"type": "Feature", "properties": prop, "geometry": mapping(geom)})

        except Exception as e:
            print(f"[WARN] Patch {idx} dilewati: {e}")
            continue

    if not rows:
        print("[INFO] Semua patch tersaring oleh MIN_PIXELS.")
        return

    # Urutkan: area besar dahulu, lalu mean tinggi
    rows.sort(key=lambda r: (-r["area_km2"], -(r["value_mean"] if r["value_mean"] is not None else -999)))

    # Tulis CSV
    csv_path = OUT_TAB_DIR / "hotspot_patches_p95.csv"
    fieldnames = [
        "patch_id", "threshold_mode", "threshold_value", "threshold_laju",
        "morph", "n_pixels", "area_ha", "area_km2", "mean_laju", "max_laju",
        "value_mean", "value_max", "centroid_lon", "centroid_lat",
        "n_villages", "n_desa", "desa_joined"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Tulis GeoJSON (dua nama untuk kompatibilitas)
    gj_payload = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}
    }
    # utama (sesuai path yang kamu sebut)
    gj_main = OUT_GEO_DIR / "hotspot_patches_p95.geojson"
    # alias tambahan (kalau ada skrip lama yang mencari nama lain)
    gj_alias = OUT_GEO_DIR / "hotspots_p95.geojson"

    gj_main.write_text(json.dumps(gj_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    gj_alias.write_text(json.dumps(gj_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[Selesai]")
    print(f"- Tabel CSV     : {csv_path}")
    print(f"- GeoJSON utama : {gj_main}")
    print(f"- GeoJSON alias : {gj_alias}")
    print(f"- Jumlah patch  : {len(rows)}")

if __name__ == "__main__":
    main()
