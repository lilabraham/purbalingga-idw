# scripts/08_export_geotiff.py
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.mask import mask
from shapely.geometry import shape
from matplotlib import colormaps
from matplotlib.colors import Normalize
from PIL import Image

BASE = Path(__file__).resolve().parents[1]

# ---------- INPUTS ----------
META_PATH = BASE / "outputs" / "data" / "rasters" / "idw_surface_meta.json"
NPY_PATH = BASE / "outputs" / "data" / "rasters" / "idw_surface.npy"
GEOJSON_PATH = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson" # Untuk kliping

# ---------- OUTPUTS ----------
TIF_PATH = BASE / "outputs" / "data" / "rasters" / "idw_surface_wgs84.tif" # TIF final (ter-klip)
PNG_PATH = BASE / "outputs" / "data" / "rasters" / "idw_surface_rgba.png" # PNG final (dari TIF ter-klip)

# ---------- PARAMS ----------
TARGET_CRS = "EPSG:4326"
NODATA_VALUE = -9999.0
CMAP_NAME = "inferno" # Sesuai permintaan: hot=merah/kuning, cold=ungu/gelap
USE_META_SCALE = True
# =========================
# 1. Cek Input
# =========================
print("1. Memeriksa file input...")
assert META_PATH.exists(), f"Tidak ketemu meta: {META_PATH} (jalankan 07)"
assert NPY_PATH.exists(), f"Tidak ketemu array: {NPY_PATH} (jalankan 07)"
assert GEOJSON_PATH.exists(), f"Tidak ketemu boundary: {GEOJSON_PATH}"

meta = json.loads(META_PATH.read_text(encoding="utf-8"))
Z = np.load(NPY_PATH) # (ny, nx)
Z = Z.astype("float32")
Z = np.where(np.isfinite(Z), Z, NODATA_VALUE)
ny, nx = Z.shape

# =========================
# 2. Ambil Info Spasial dari Meta (dari 07)
# =========================
print("2. Membaca metadata spasial...")
b_pad = meta.get("bounds_wgs84_padded")
assert b_pad, "Meta tidak punya 'bounds_wgs84_padded'. Jalankan 07 dulu."

west = float(b_pad["lon_min"])
east = float(b_pad["lon_max"])
south = float(b_pad["lat_min"])
north = float(b_pad["lat_max"])

dx = (east - west) / nx
dy = (north - south) / ny # positif

# Transform (origin=kiri atas, y-axis ke bawah)
transform = from_origin(west, north, dx, dy)

profile = {
    "driver": "GTiff",
    "dtype": "float32",
    "count": 1,
    "crs": TARGET_CRS,
    "transform": transform,
    "height": ny,
    "width": nx,
    "nodata": NODATA_VALUE,
}

# =========================
# 3. Klip Raster ke Batas GeoJSON
# =========================
print(f"3. Meng-klip raster ({(ny, nx)}) ke batas {GEOJSON_PATH.name}...")
gj = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
geoms = [shape(ft["geometry"]) for ft in gj.get("features")]
assert len(geoms) > 0, "GeoJSON tidak mengandung fitur geometri."

# Tulis NPY ke TIF (dalam memori) untuk di-klip
with rasterio.MemoryFile() as memfile:
    with memfile.open(**profile) as ds:
        ds.write(Z, 1)
    
    # Buka kembali TIF in-memory dan klip
    with memfile.open() as ds:
        out_image, out_transform = mask(
            ds, 
            geoms, 
            crop=True,        # Potong raster ke bbox geometri
            all_touched=True, # Bakar piksel jika menyentuh batas
            nodata=NODATA_VALUE
        )
        out_meta = ds.meta.copy()

out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform,
    "compress": "deflate",
    "predictor": 2,
    "tiled": True,
})

# =========================
# 4. Tulis GeoTIFF Final (yang sudah ter-klip)
# =========================
with rasterio.open(TIF_PATH, "w", **out_meta) as dst:
    dst.write(out_image)

print(f"4. Sukses menyimpan GeoTIFF ter-klip: {TIF_PATH}")
print(f"   - Dimensi baru: {(out_meta['height'], out_meta['width'])}")
with rasterio.open(TIF_PATH) as ds:
    print(f"   - Bounds final (W,S,E,N): {ds.bounds}")

# =========================
# 5. Render PNG dari TIF Ter-klip
# =========================
print(f"5. Merender {PNG_PATH.name} dari TIF ter-klip...")
# Ambil vmin/vmax dari meta (dihasilkan 07)
scaling = meta.get("scaling", {})
vmin = scaling.get("vmin", 0.0)
vmax = scaling.get("vmax")
if vmax is None:
    # Fallback jika meta tidak ada vmax
    data_clipped = out_image[0]
    valid_data = data_clipped[data_clipped != NODATA_VALUE]
    if valid_data.size > 0:
        vmax = np.percentile(valid_data, 95)
    else:
        vmax = vmin + 1.0
print(f"   - Colormap: '{CMAP_NAME}' (vmin={vmin:.2f}, vmax={vmax:.2f})")

# Siapkan data: 1D array, 0..1, nodata
data_clipped = out_image[0]
is_nodata = (data_clipped == NODATA_VALUE) | (~np.isfinite(data_clipped))
data_clipped[data_clipped < vmin] = vmin
data_clipped[is_nodata] = np.nan

# Normalisasi 0-1
norm = Normalize(vmin=vmin, vmax=vmax)
data_norm = norm(data_clipped) # (H, W) float 0-1, NaN where nodata

# Terapkan colormap
cmap = colormaps.get_cmap(CMAP_NAME)
rgba = cmap(data_norm) # (H, W, 4) float 0-1

# Set NoData menjadi transparan
rgba[is_nodata, 3] = 0.0 # Set Alpha channel to 0

# Konversi ke 8-bit RGBA
rgba_u8 = (rgba * 255).astype(np.uint8)

# Simpan PNG
img = Image.fromarray(rgba_u8, 'RGBA')
img.save(PNG_PATH)
print(f"   - Sukses menyimpan PNG: {PNG_PATH}")
print("\nProses 08 selesai.")