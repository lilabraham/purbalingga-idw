# scripts/04_choropleth_with_tooltip.py
import json
from pathlib import Path
import pandas as pd
from shapely.geometry import shape
from shapely.ops import unary_union
import folium
from folium.features import GeoJson, GeoJsonTooltip, GeoJsonPopup

BASE = Path(__file__).resolve().parents[1]

GEOJSON_PATH = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"
AGG_PATH     = BASE / "outputs" / "data" / "aggregated_laju_100k.csv"
OUT_HTML     = BASE / "outputs" / "html" / "05_choropleth_laju_100k_tooltip.html"

# --- Load data ---
assert GEOJSON_PATH.exists(), f"Tidak ketemu: {GEOJSON_PATH}"
assert AGG_PATH.exists(),     f"Tidak ketemu: {AGG_PATH} (jalankan 02_generate_maps.py dulu)"

gj  = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
agg = pd.read_csv(AGG_PATH, dtype={"CC_4": str}).fillna(0)

# --- Pusat peta ---
geoms  = [shape(ft["geometry"]) for ft in gj["features"]]
center = unary_union(geoms).centroid
clat, clon = float(center.y), float(center.x)

# --- Lookup nilai per CC_4 ---
lookup = agg.set_index("CC_4")[["NAME_4","penduduk","kejadian_count","laju_100k"]].to_dict("index")

# --- Enrich properties di GeoJSON agar bisa dipakai tooltip/popup ---
for ft in gj["features"]:
    props = ft.get("properties", {})
    cc4   = str(props.get("CC_4") or props.get("cc_4") or "")
    m = lookup.get(cc4)
    if m:
        props["_penduduk"] = int(m["penduduk"])
        props["_kejadian"] = int(m["kejadian_count"])
        props["_laju"]     = round(float(m["laju_100k"]), 2)
    else:
        props["_penduduk"] = 0
        props["_kejadian"] = 0
        props["_laju"]     = 0.0

# --- Map ---
m = folium.Map(location=[clat, clon], zoom_start=11, tiles="cartodbpositron")

# Warna area (choropleth) dengan laju/100k
folium.Choropleth(
    geo_data=gj,
    data=agg.rename(columns={"CC_4": "key"}),
    columns=["key", "laju_100k"],
    key_on="feature.properties.CC_4",
    legend_name="Laju per 100k penduduk",
    fill_opacity=0.6,
    line_opacity=0.7,
    nan_fill_opacity=0.15
).add_to(m)

# Outline + tooltip/popup informatif
GeoJson(
    gj,
    name="Info Desa",
    style_function=lambda x: {"fillOpacity": 0, "weight": 0.8, "color": "#555"},
    highlight_function=lambda x: {"weight": 2, "color": "#000"},
    tooltip=GeoJsonTooltip(
        fields=["NAME_4","CC_4","_penduduk","_kejadian","_laju"],
        aliases=["Desa","Kode","Penduduk","Kejadian","Laju/100k"],
        localize=True,
        sticky=False,
    ),
    popup=GeoJsonPopup(
        fields=["NAME_4","CC_4","_penduduk","_kejadian","_laju"],
        aliases=["Desa","Kode","Penduduk","Kejadian","Laju/100k"],
        localize=True,
    ),
).add_to(m)

folium.LayerControl().add_to(m)

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
m.save(str(OUT_HTML))
print("Saved:", OUT_HTML)
