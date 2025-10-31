# scripts/12_standardize_and_enrich_cases.py
from pathlib import Path
import json, math
import pandas as pd
import numpy as np
from shapely.geometry import shape, Point
from shapely.prepared import prep
from pyproj import Transformer

BASE = Path(__file__).resolve().parents[1]
MASTER = BASE / "data" / "raw" / "kejadian" / "data_titik_kasus_2024.csv"
DESA_GJ = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"

KEY_DESA = ["NAME_4","DESA","DESA_KEL","NAMDESA","nama_desa","nama","desa_kel"]
KEY_KODE = ["CC_4","KODE","KODE_DESA","KD_DESA","kode_desa","cc_4"]
KEY_KEC  = ["NAME_3","KECAMATAN","kecamatan","NAMKEC","nama_kecamatan"]

FALLBACK_RADIUS_M = 200.0

def pick(props, keys):
    for k in keys:
        if k in props and props[k] not in (None,"","None",0):
            return props[k]
    return None

def load_desa():
    gj = json.loads(DESA_GJ.read_text(encoding="utf-8"))
    items = []
    for ft in gj.get("features", []):
        geom = shape(ft["geometry"])
        props = ft.get("properties", {})
        items.append({
            "prep": prep(geom),
            "geom": geom,
            "desa": str(pick(props, KEY_DESA) or ""),
            "kode": str(pick(props, KEY_KODE) or ""),
            "kec":  str(pick(props, KEY_KEC)  or "")
        })
    return items

def enrich_row(lon, lat, desa_items, to_utm49s, centers_EN):
    pt = Point(lon, lat)
    # covers
    for it in desa_items:
        if it["prep"].covers(pt):
            return it["kode"], it["desa"], it["kec"], "covers"
    # intersects (di tepi)
    for it in desa_items:
        if it["prep"].intersects(pt):
            return it["kode"], it["desa"], it["kec"], "intersects"
    # fallback nearest centroid <= 200 m (UTM)
    ex, ny = to_utm49s.transform(lon, lat)
    d = [math.hypot(ex-cx, ny-cy) for (cx,cy) in centers_EN]
    i = int(np.argmin(d))
    if d[i] <= FALLBACK_RADIUS_M:
        it = desa_items[i]
        return it["kode"], it["desa"], it["kec"], f"nearest<={int(FALLBACK_RADIUS_M)}m"
    return "", "", "", "unmatched"

def main():
    assert MASTER.exists(), f"Tidak ketemu: {MASTER}"
    assert DESA_GJ.exists(), f"Tidak ketemu boundary: {DESA_GJ}"

    # 1) baca CSV finalmu dan deteksi kolom
    df = pd.read_csv(MASTER)
    cols = {c.lower().strip(): c for c in df.columns}

    lon_col = cols.get("lon") or cols.get("longitude") or cols.get("lng") or cols.get("x")
    lat_col = cols.get("lat") or cols.get("latitude") or cols.get("y")
    desa_raw_col = cols.get("desa_raw")
    kec_raw_col  = cols.get("kec_raw")
    # variasi "jumlah kejadian"
    count_col = None
    for k in ["count","jumlah_kejadian","jumlah kejadian","jumlah"]:
        if k in cols: count_col = cols[k]; break

    assert lon_col and lat_col, f"Kolom lon/lat tidak ditemukan. Ada: {df.columns.tolist()}"

    # 2) standardisasi kolom
    out = pd.DataFrame()
    out["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    out["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    out = out.dropna(subset=["lon","lat"]).copy()

    if count_col:
        out["count"] = pd.to_numeric(df[count_col], errors="coerce").fillna(1).astype(int)
    else:
        out["count"] = 1

    if desa_raw_col:
        out["DESA_RAW"] = df[desa_raw_col].astype(str)
    else:
        out["DESA_RAW"] = ""

    # pakai kec_raw sebagai KECAMATAN jika ada
    out["KECAMATAN"] = df[kec_raw_col].astype(str) if kec_raw_col else ""

    # 3) lengkapi CC_4 & NAME_4 dari boundary desa
    desa_items = load_desa()
    to_utm49s = Transformer.from_crs("EPSG:4326","EPSG:32749",always_xy=True)
    centers_EN = []
    for it in desa_items:
        cx, cy = it["geom"].centroid.x, it["geom"].centroid.y
        centers_EN.append(to_utm49s.transform(cx, cy))

    cc4_list, nm_list, kec_list, how_list = [], [], [], []
    for lon, lat, kec0 in zip(out["lon"], out["lat"], out["KECAMATAN"]):
        kode, desa, kec, how = enrich_row(float(lon), float(lat), desa_items, to_utm49s, centers_EN)
        # kalau KECAMATAN masih kosong, isi dari boundary jika ada
        if not kec0 and kec: kec0 = kec
        cc4_list.append(kode); nm_list.append(desa); kec_list.append(kec0); how_list.append(how)

    out["CC_4"] = cc4_list
    out["NAME_4"] = nm_list
    out["KECAMATAN"] = kec_list
    out["_match"] = how_list

    # 4) urut kolom standar (pipeline akan pakai ini)
    cols_order = ["lon","lat","count","CC_4","NAME_4","KECAMATAN","DESA_RAW","_match"]
    out = out[cols_order]

    # 5) simpan balik MENIMPA file master
    out.to_csv(MASTER, index=False, encoding="utf-8")
    print("OK → file distandarisasi & dilengkapi admin:", MASTER)
    print(out.head(8).to_string(index=False))
    print("Ringkasan match:")
    print(out["_match"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
