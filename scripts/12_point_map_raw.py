# scripts/12_point_map_raw.py
from pathlib import Path
import json
import pandas as pd
import folium
from shapely.geometry import shape
from shapely.ops import unary_union

BASE = Path(__file__).resolve().parents[1]
GEOJSON_PATH = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"
KEJADIAN_CSV = BASE / "data" / "raw" / "kejadian" / "data_titik_kasus_2024.csv"
OUT_HTML_DIR = BASE / "outputs" / "html"
OUT_HTML_DIR.mkdir(parents=True, exist_ok=True)
HTML_OUT = OUT_HTML_DIR / "12_sebaran_titik_kejadian.html"

# ---- cek file
assert GEOJSON_PATH.exists(), f"Tidak ketemu: {GEOJSON_PATH}"
assert KEJADIAN_CSV.exists(), f"Tidak ketemu: {KEJADIAN_CSV}"

# ---- baca boundary & dapatkan pusat peta
gj = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
geoms = [shape(ft["geometry"]) for ft in gj.get("features", [])]
union = unary_union(geoms)
cy, cx = union.centroid.y, union.centroid.x

# ---- baca data kejadian (deteksi kolom lat/lon)
df = pd.read_csv(KEJADIAN_CSV)
lat_candidates = ["lat","latitude","Lat","Latitude","y","Y"]
lon_candidates = ["lon","longitude","Lon","Longitude","x","X","lng","Lng"]

lat_col = next((c for c in lat_candidates if c in df.columns), None)
lon_col = next((c for c in lon_candidates if c in df.columns), None)
if not lat_col or not lon_col:
    raise ValueError(f"Tidak menemukan kolom lat/lon. Kolom tersedia: {df.columns.tolist()}")

df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
df = df.dropna(subset=[lat_col, lon_col])

# ---- buat peta (ada scale bar)
m = folium.Map(location=[cy, cx], zoom_start=11, tiles=None, control_scale=True)
folium.TileLayer("cartodbpositron", name="Carto Light", control=True).add_to(m)

# batas desa tipis netral
folium.GeoJson(
    gj,
    name="Batas Desa",
    style_function=lambda x: {"fillOpacity": 0.0, "weight": 1.0, "color": "#666"}
).add_to(m)

# titik kejadian individual (tanpa cluster)
fg = folium.FeatureGroup(name="Titik Kejadian 2024", show=True)
for _, r in df.iterrows():
    folium.CircleMarker(
        location=[float(r[lat_col]), float(r[lon_col])],
        radius=3,
        color="#FF5722",      # oranye kemerahan
        weight=0.5,
        fill=True,
        fill_color="#FF5722",
        fill_opacity=0.9,
    ).add_to(fg)
fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
m.save(str(HTML_OUT))
print(f"OK. Buka: {HTML_OUT}")
