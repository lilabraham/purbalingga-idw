# scripts/12_enrich_points_with_admin.py
import json, math
from pathlib import Path
import pandas as pd
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from pyproj import Geod, Transformer

BASE = Path(__file__).resolve().parents[1]

# INPUT
BOUNDARY_GJ = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"
POINTS_CSV  = BASE / "data" / "raw" / "kejadian" / "points_min.csv"

# OUTPUT
OUT_DIR     = BASE / "outputs" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV     = OUT_DIR / "kejadian_enriched.csv"

# Kunci properti yang umum dipakai di boundary (ubah jika nama di file kamu beda)
KEY_DESA   = ["NAME_4","DESA","DESA_KEL","NAMDESA","nama_desa","nama","desa_kel"]
KEY_KODE   = ["CC_4","KODE","KODE_DESA","KD_DESA","kode_desa","cc_4"]
KEY_KEC    = ["NAME_3","KECAMATAN","kecamatan","NAMKEC","nama_kecamatan"]
# Radius fallback kalau titik mepet tepi (meter)
FALLBACK_RADIUS_M = 200.0

def pick(props: dict, keys: list):
    for k in keys:
        if k in props and props[k] not in (None, "", "None", 0):
            return props[k]
    return None

def main():
    assert BOUNDARY_GJ.exists(), f"Tidak ketemu boundary: {BOUNDARY_GJ}"
    assert POINTS_CSV.exists(),  f"Tidak ketemu CSV titik: {POINTS_CSV}"

    # 1) Baca boundary
    gj = json.loads(BOUNDARY_GJ.read_text(encoding="utf-8"))
    feats = gj.get("features", [])
    if not feats:
        raise RuntimeError("GeoJSON boundary tidak punya 'features'.")

    # Siapkan geometri & properti
    geoms = []
    for ft in feats:
        geom = shape(ft["geometry"])
        props = ft.get("properties", {})
        desa  = str(pick(props, KEY_DESA) or "")
        kode  = str(pick(props, KEY_KODE) or "")
        kec   = str(pick(props, KEY_KEC)  or "")
        geoms.append({
            "geom": geom,
            "prep": prep(geom),
            "desa": desa,
            "kode": kode,
            "kec":  kec
        })

    # 2) Baca titik
    df = pd.read_csv(POINTS_CSV)
    # Normalisasi nama kolom
    col_lon = next((c for c in df.columns if c.lower() in ("lon","longitude","lng","x")), None)
    col_lat = next((c for c in df.columns if c.lower() in ("lat","latitude","y")), None)
    col_cnt = next((c for c in df.columns if c.lower() in ("count","jumlah","jumlah_kasus")), None)
    if not (col_lon and col_lat and col_cnt):
        raise RuntimeError(f"CSV titik harus punya kolom lon,lat,count. Kolom ada: {df.columns.tolist()}")

    # 3) Fungsi cari desa untuk titik
    geod = Geod(ellps="WGS84")
    wgs_to_utm49s = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)
    # Prehitung centroid UTM untuk fallback nearest
    centers = []
    for g in geoms:
        cx, cy = g["geom"].centroid.x, g["geom"].centroid.y  # WGS84
        ex, ny = wgs_to_utm49s.transform(cx, cy)
        centers.append((ex, ny))

    def find_admin(lon: float, lat: float):
        pt = Point(lon, lat)
        # tahap 1: covers/contains cepat
        for g in geoms:
            if g["prep"].covers(pt):
                return g["kode"], g["desa"], g["kec"], "covers"
        # tahap 2: intersects (kadang polygon punya batas garis)
        for g in geoms:
            if g["prep"].intersects(pt):
                return g["kode"], g["desa"], g["kec"], "intersects"
        # tahap 3: fallback nearest centroid (dalam radius meter)
        ex, ny = wgs_to_utm49s.transform(lon, lat)
        # hitung jarak euklidian ke centroid UTM
        dists = [math.hypot(ex-cx, ny-cy) for (cx,cy) in centers]
        idx = int(np.argmin(dists))
        if dists[idx] <= FALLBACK_RADIUS_M:
            g = geoms[idx]
            return g["kode"], g["desa"], g["kec"], f"nearest<= {int(FALLBACK_RADIUS_M)}m"
        # gagal
        return "", "", "", "unmatched"

    # 4) Proses tiap titik
    out_rows = []
    for i, r in df.iterrows():
        try:
            lon = float(r[col_lon]); lat = float(r[col_lat])
            cnt = int(r[col_cnt])
        except Exception:
            continue
        kode, desa, kec, how = find_admin(lon, lat)
        out_rows.append({
            "lon": lon, "lat": lat, "count": cnt,
            "CC_4": kode, "NAME_4": desa, "KECAMATAN": kec,
            "_match": how
        })

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("Saved:", OUT_CSV)
    print("Ringkasan match:")
    print(out["_match"].value_counts(dropna=False))

if __name__ == "__main__":
    import numpy as np
    main()
