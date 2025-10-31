# scripts/05_make_idw_samples.py
import json
from pathlib import Path
import pandas as pd
from shapely.geometry import shape
from pyproj import Transformer

BASE = Path(__file__).resolve().parents[1]
GEOJSON = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"
AGGCSV  = BASE / "outputs" / "data" / "aggregated_laju_100k.csv"
OUTCSV  = BASE / "outputs" / "data" / "idw_samples.csv"

assert GEOJSON.exists(), f"Boundary tidak ditemukan: {GEOJSON}"
assert AGGCSV.exists(),  f"Agregat tidak ditemukan: {AGGCSV} (jalankan 02_generate_maps.py dulu)"

# baca data
gj   = json.loads(GEOJSON.read_text(encoding="utf-8"))
agg  = pd.read_csv(AGGCSV, dtype={"CC_4": str})

# converter WGS84 (lon,lat) -> UTM 49S (meter)
to_utm49s = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)

rows = []
for ft in gj["features"]:
    props = ft.get("properties", {})
    cc4   = str(props.get("CC_4") or props.get("cc_4") or "")
    name  = props.get("NAME_4") or props.get("NAME") or ""
    geom  = shape(ft["geometry"])
    c     = geom.centroid
    # cari laju_100k untuk CC_4 ini
    rec = agg.loc[agg["CC_4"] == cc4]
    if len(rec) != 1:
        continue
    value = float(rec.iloc[0]["laju_100k"])
    lon, lat = float(c.x), float(c.y)
    east, north = to_utm49s.transform(lon, lat)  # meter
    rows.append({
        "CC_4": cc4,
        "NAME_4": name,
        "lat": lat,
        "lon": lon,
        "east": east,
        "north": north,
        "value": value,  # laju per 100.000
    })

out = pd.DataFrame(rows)
OUTCSV.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTCSV, index=False, encoding="utf-8")

print("Saved:", OUTCSV)
print("Rows:", len(out), "| Kolom:", list(out.columns))
print(out.head(5))
