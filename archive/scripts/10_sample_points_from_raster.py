# scripts/10_sample_points_from_raster.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.sample import sample_gen
from shapely.geometry import shape, Point, mapping

BASE = Path(__file__).resolve().parents[1]
GJ_PATH   = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"
TIF_PATH  = BASE / "outputs" / "data" / "rasters" / "idw_surface_wgs84.tif"
OUT_CSV   = BASE / "outputs" / "data" / "centroid_laju100k.csv"
OUT_GEOJ  = BASE / "outputs" / "data" / "centroid_laju100k.geojson"

assert GJ_PATH.exists(), "Boundary tidak ketemu"
assert TIF_PATH.exists(), "GeoTIFF tidak ketemu (jalankan 08_export_geotiff.py)"

gj = json.loads(GJ_PATH.read_text(encoding="utf-8"))
features = gj.get("features", [])
rows = []

# bikin centroid (WGS84)
for ft in features:
    geom = shape(ft["geometry"])
    c = geom.centroid
    props = ft.get("properties", {})
    cc4 = props.get("CC_4") or props.get("cc_4") or ""
    name = props.get("NAME_4") or props.get("DESA") or props.get("nama_desa") or ""
    rows.append({"CC_4": str(cc4), "NAME_4": str(name), "lon": float(c.x), "lat": float(c.y)})

df = pd.DataFrame(rows)

# sample raster di titik centroid
with rasterio.open(TIF_PATH) as ds:
    pts = [(r["lon"], r["lat"]) for _, r in df.iterrows()]
    vals = list(sample_gen(ds, pts, indexes=1))
    vals = [float(v[0]) if np.isfinite(v[0]) else None for v in vals]

df["laju100k_idw"] = vals
df.to_csv(OUT_CSV, index=False, encoding="utf-8")

# export juga sebagai GeoJSON (buat dipakai di QGIS/leaflet)
geo_features = []
for _, r in df.iterrows():
    geo_features.append({
        "type": "Feature",
        "properties": {
            "CC_4": r["CC_4"],
            "NAME_4": r["NAME_4"],
            "laju100k_idw": r["laju100k_idw"]
        },
        "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]}
    })

out = {"type":"FeatureCollection","features":geo_features}
OUT_GEOJ.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", OUT_CSV)
print("Saved:", OUT_GEOJ)
