# scripts/09_check_geotiff.py
from pathlib import Path
import json, math
import numpy as np
import rasterio
from rasterio.crs import CRS
from shapely.geometry import shape, box
from shapely.ops import unary_union

BASE = Path(__file__).resolve().parents[1]
TIF = BASE / "outputs" / "data" / "rasters" / "idw_surface_wgs84.tif"
META = BASE / "outputs" / "data" / "rasters" / "idw_surface_meta.json"
GJ   = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"

assert TIF.exists(), f"Tidak ketemu: {TIF}"

with rasterio.open(TIF) as ds:
    print("CRS             :", ds.crs)
    print("Size (W x H)    :", ds.width, "x", ds.height)
    print("Bounds (lon/lat):", ds.bounds)
    print("Transform       :", ds.transform)
    print("NoData          :", ds.nodata)

    band = ds.read(1, masked=True)   # masked dengan nodata
    valid = band.compressed()
    nodata_count = int(band.mask.sum())
    valid_count = band.size - nodata_count
    print("Valid/Nodata    :", valid_count, "/", nodata_count)
    if valid.size:
        p5, p50, p95 = np.percentile(valid, [5, 50, 95])
        print("Min/Med/P95     :", float(valid.min()), float(p50), float(p95))

    # Perkiraan ukuran piksel (meter)
    xdeg = ds.transform.a
    ydeg = -ds.transform.e
    latc = (ds.bounds.top + ds.bounds.bottom) / 2
    m_per_deg = 111320.0
    x_m = xdeg * m_per_deg * math.cos(math.radians(latc))
    y_m = ydeg * m_per_deg
    print("Pixel size ~    : %.1f m x %.1f m" % (x_m, y_m))

# Cek overlay terhadap batas (opsional, tapi meyakinkan)
if GJ.exists():
    gj = json.loads(GJ.read_text(encoding="utf-8"))
    uni = unary_union([shape(ft["geometry"]) for ft in gj.get("features", [])])
    rb = box(*ds.bounds)  # kotak raster
    print("Boundary ⊂ Raster bounds:",
          rb.contains(uni.buffer(1e-9)))
# Bandingkan vmin/vmax meta (jika ada)
if META.exists():
    meta = json.loads(META.read_text(encoding="utf-8"))
    vmin = meta.get("vmin"); vmax = meta.get("vmax") or meta.get("vmax_p95")
    print("Meta vmin/vmax  :", vmin, vmax)
