# scripts/02_generate_maps.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep

import folium
from folium.plugins import HeatMap, MarkerCluster


# ========== PATHS ==========
BASE = Path(__file__).resolve().parents[1]

GEOJSON_PATH   = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"
KEJADIAN_CSV   = BASE / "data" / "raw" / "kejadian" / "data_titik_kasus_2024.csv"
PENDUDUK_CLEAN = BASE / "data" / "processed" / "penduduk_clean_for_join.csv"

OUT_DIR  = BASE / "outputs"
OUT_HTML = OUT_DIR / "html"
OUT_DATA = OUT_DIR / "data"
for p in (OUT_DIR, OUT_HTML, OUT_DATA):
    p.mkdir(parents=True, exist_ok=True)


# ========== UTIL ==========
NAME_KEYS = ["NAME_4","NAME","DESA","DESA_KEL","NAMDESA","nama","nama_desa","desa_kel"]
CODE_KEYS = ["CC_4","KODE","KODE_DESA","KD_DESA","kode_desa","cc_4"]

def get_prop(props: dict, keys: list):
    for k in keys:
        if k in props and props[k] not in (None, "", 0, "None"):
            return props[k]
    return None

def detect_lat_lon(columns):
    # exact first
    lat_candidates = ["latitude","lat","y","Latitude","Lat","Y"]
    lon_candidates = ["longitude","lon","lng","x","Longitude","Lon","Lng","X"]
    lat = next((c for c in columns if c in lat_candidates), None)
    lon = next((c for c in columns if c in lon_candidates), None)
    if lat and lon:
        return lat, lon
    # fallback by substring
    lat = next((c for c in columns if "lat" in c.lower()), None)
    lon = next((c for c in columns if ("lon" in c.lower()) or ("lng" in c.lower())), None)
    return lat, lon


# ========== MAIN ==========
def main():
    # --- checks
    assert GEOJSON_PATH.exists(), f"Tidak ketemu: {GEOJSON_PATH}"
    assert KEJADIAN_CSV.exists(), f"Tidak ketemu: {KEJADIAN_CSV}"
    assert PENDUDUK_CLEAN.exists(), f"Tidak ketemu: {PENDUDUK_CLEAN}"

    # 1) Boundary & center
    gj = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    feats = gj.get("features", [])
    if not feats:
        raise ValueError("GeoJSON tidak punya 'features'. Cek file boundary.")
    geoms = [shape(ft["geometry"]) for ft in feats]
    center = unary_union(geoms).centroid
    clat, clon = float(center.y), float(center.x)

    # 2) Siapkan polygon prepared
    recs = []
    for ft in feats:
        props = ft.get("properties", {})
        cc4 = str(get_prop(props, CODE_KEYS))
        recs.append({"poly": prep(shape(ft["geometry"])), "CC_4": cc4, "props": props})

    # 3) Baca CSV kejadian & cari kolom lon/lat
    inc = pd.read_csv(KEJADIAN_CSV)
    lat_col, lon_col = detect_lat_lon(list(inc.columns))
    if not lat_col or not lon_col:
        raise ValueError(
            f"Tidak menemukan kolom latitude/longitude pada CSV kejadian.\n"
            f"Kolom yang ada: {inc.columns.tolist()}"
        )

    # pastikan numeric
    inc[lat_col] = pd.to_numeric(inc[lat_col], errors="coerce")
    inc[lon_col] = pd.to_numeric(inc[lon_col], errors="coerce")
    inc = inc.dropna(subset=[lat_col, lon_col]).copy()

    # jika ada 'count'/'jumlah_kejadian', pakai itu; jika tidak, anggap 1
    count_col = None
    for k in ["count", "jumlah_kejadian", "jumlah kejadian", "jumlah", "kejadian"]:
        if k in [c.lower() for c in inc.columns]:
            # temukan nama asli dengan case aslinya
            for c in inc.columns:
                if c.lower() == k:
                    count_col = c
                    break
            break
    if count_col is None:
        inc["count"] = 1
        count_col = "count"
    else:
        inc["count"] = pd.to_numeric(inc[count_col], errors="coerce").fillna(1).astype(int)

    # 4) Map titik → CC_4
    def locate_cc4(pt: Point):
        # contains dulu
        for r in recs:
            if r["poly"].contains(pt):
                return r["CC_4"]
        # lalu intersects (titik di tepi batas)
        for r in recs:
            if r["poly"].intersects(pt):
                return r["CC_4"]
        return None

    # koordinat & CC_4
    points = [Point(float(r[lon_col]), float(r[lat_col])) for _, r in inc.iterrows()]
    inc["CC_4"] = [locate_cc4(pt) for pt in points]
    inc["CC_4"] = inc["CC_4"].astype(str)

    # QA total raw
    total_raw = int(inc["count"].sum())

    # group kejadian per CC_4 (pakai jumlah 'count')
    tmp = inc.dropna(subset=["CC_4"]).copy()
    tmp["CC_4"] = tmp["CC_4"].astype(str).str.strip()
    counts = (
        tmp.groupby("CC_4", as_index=False)["count"]
           .sum()
           .rename(columns={"count": "kejadian_count"})
    )

    # 5) Join penduduk & laju/100k
    pop = pd.read_csv(PENDUDUK_CLEAN, dtype={"CC_4": str})
    pop["CC_4"] = pop["CC_4"].astype(str).str.strip()

    # deteksi nama kolom penduduk
    pop_cols_lower = {c.lower(): c for c in pop.columns}
    pop_key = None
    for k in ["penduduk", "jumlah_penduduk", "jml_penduduk", "jmlpenduduk", "population", "populasi", "pop"]:
        if k in pop_cols_lower:
            pop_key = pop_cols_lower[k]
            break
    if pop_key is None:
        raise ValueError(
            f"Kolom penduduk tidak ditemukan di {PENDUDUK_CLEAN}. "
            f"Kolom tersedia: {list(pop.columns)}"
        )

    pop = pop[["CC_4", pop_key]].rename(columns={pop_key: "penduduk"})
    pop["penduduk"] = pd.to_numeric(pop["penduduk"], errors="coerce").fillna(0).astype(int)

    # PENTING: semua CC_4 yang punya kejadian harus ikut → LEFT dari counts
    agg = counts.merge(pop, on="CC_4", how="left")
    agg["penduduk"] = pd.to_numeric(agg["penduduk"], errors="coerce").fillna(0).astype(int)
    agg["kejadian_count"] = pd.to_numeric(agg["kejadian_count"], errors="coerce").fillna(0).astype(int)

    # laju per 100k (penduduk 0 → laju 0)
    agg["laju_100k"] = np.where(
        agg["penduduk"] > 0,
        (agg["kejadian_count"] / agg["penduduk"]) * 100000.0,
        0.0
    )

    # simpan agregat
    agg_path = OUT_DATA / "aggregated_laju_100k.csv"
    agg.to_csv(agg_path, index=False, encoding="utf-8")

    # QA: total aggregated (harus = total_raw)
    total_agg = int(agg["kejadian_count"].sum())
    print(f"[QA] Total kejadian (raw): {total_raw}")
    print(f"[QA] Total kejadian (agg): {total_agg}")

    # 6) Peta 1 — Centroid tiap desa
    m1 = folium.Map(location=[clat, clon], zoom_start=11, tiles="cartodbpositron")
    folium.GeoJson(gj, name="Batas", style_function=lambda x: {"fillOpacity": 0.05, "weight": 1}).add_to(m1)
    for ft in feats:
        g = shape(ft["geometry"]).centroid
        nm = get_prop(ft.get("properties", {}), NAME_KEYS) or get_prop(ft.get("properties", {}), CODE_KEYS) or "(tanpa nama)"
        folium.CircleMarker([g.y, g.x], radius=3, popup=str(nm)).add_to(m1)
    m1.save(str(OUT_HTML / "01_centroid_desa.html"))

    # 7) Peta 2 — Titik kejadian (cluster)
    m2 = folium.Map(location=[clat, clon], zoom_start=11, tiles="cartodbpositron")
    folium.GeoJson(gj, name="Batas", style_function=lambda x: {"fillOpacity": 0.05, "weight": 1}).add_to(m2)
    cl = MarkerCluster(name="Titik Kejadian").add_to(m2)
    for (pt, w) in zip(points, inc["count"].tolist()):
        folium.CircleMarker([pt.y, pt.x], radius=3 + min(4, int(w)), popup=f"count={w}").add_to(cl)
    m2.save(str(OUT_HTML / "02_titik_kejadian.html"))

    # 8) Peta 3 — Heatmap (berbobot count)
    m3 = folium.Map(location=[clat, clon], zoom_start=11, tiles="cartodbpositron")
    folium.GeoJson(gj, name="Batas", style_function=lambda x: {"fillOpacity": 0.0, "weight": 1}).add_to(m3)
    hm_data = [[float(r[lat_col]), float(r[lon_col]), float(w)] for (_, r), w in zip(inc.iterrows(), inc["count"].tolist())]
    HeatMap(hm_data, radius=18, blur=24, max_zoom=13).add_to(m3)
    m3.save(str(OUT_HTML / "03_heatmap_kejadian.html"))

    # 9) Peta 4 — Choropleth laju/100k
    m4 = folium.Map(location=[clat, clon], zoom_start=11, tiles="cartodbpositron")
    folium.GeoJson(gj, name="Batas").add_to(m4)
    folium.Choropleth(
        geo_data=gj,
        data=agg.rename(columns={"CC_4": "key"}),
        columns=["key", "laju_100k"],
        key_on="feature.properties.CC_4",
        legend_name="Laju per 100k penduduk",
        fill_opacity=0.6,
        line_opacity=0.7,
        nan_fill_opacity=0.15
    ).add_to(m4)
    m4.save(str(OUT_HTML / "04_choropleth_laju_100k.html"))

    print("\nSelesai ✔")
    print(f"- Data  : {agg_path}")
    print(f"- HTML  : {OUT_HTML / '01_centroid_desa.html'}")
    print(f"- HTML  : {OUT_HTML / '02_titik_kejadian.html'}")
    print(f"- HTML  : {OUT_HTML / '03_heatmap_kejadian.html'}")
    print(f"- HTML  : {OUT_HTML / '04_choropleth_laju_100k.html'}")


if __name__ == "__main__":
    main()
