# scripts/07_idw_surface.py
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import branca.colormap as cm

from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep

from pyproj import Transformer
import folium
from folium.raster_layers import ImageOverlay

# NEW: untuk rasterize batas supaya nempel ke garis
from rasterio.features import rasterize
from rasterio.transform import from_origin

# =========================
# --- Konfigurasi (LOCKED) ---
USE_FIXED_VMAX: bool = True     # << kunci skala legend
FIXED_VMIN, FIXED_VMAX = 0.0, 50.0

# Resolusi grid & padding (meter)
GRID_STEP_M: float = 100.0      # << resolusi tetap
PAD_M: float = 1000.0           # << padding tetap

# Parameter layer "hotspots"
ALPHA_MAX: float = 0.95
ALPHA_GAMMA: float = 0.60
ALPHA_CUTOFF_REL: float = 0.05

# =========================
# Paths
# =========================
BASE = Path(__file__).resolve().parents[1]
GEOJSON_PATH = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"
SAMPLES_CSV = BASE / "outputs" / "data" / "idw_samples.csv"
BEST_JSON = BASE / "outputs" / "data" / "idw_best_params.json"

OUT_HTML_DIR = BASE / "outputs" / "html"
OUT_RASTER_DIR = BASE / "outputs" / "data" / "rasters"
OUT_HTML_DIR.mkdir(parents=True, exist_ok=True)
OUT_RASTER_DIR.mkdir(parents=True, exist_ok=True)

HTML_OUT = OUT_HTML_DIR / "06_idw_surface.html"
PNG_FILLED = OUT_RASTER_DIR / "idw_surface_filled.png"
PNG_HOTSPOT= OUT_RASTER_DIR / "idw_surface_hotspot.png"
NPY_OUT = OUT_RASTER_DIR / "idw_surface.npy"
META_OUT = OUT_RASTER_DIR / "idw_surface_meta.json"

assert GEOJSON_PATH.exists(), f"Boundary tidak ditemukan: {GEOJSON_PATH}"
assert SAMPLES_CSV.exists(), f"Sampel IDW tidak ditemukan: {SAMPLES_CSV} (jalankan 05_make_idw_samples.py)"
assert BEST_JSON.exists(), f"Params terbaik tidak ditemukan: {BEST_JSON} (jalankan 06_idw_loocv.py)"

# =========================
# Fungsi IDW 
# =========================
def idw_predict_for_points(target_pts_utm, sample_pts_utm, sample_values,
                           p=2, k=12, radius=None, chunk_size=20000):
    """
    target_pts_utm: (M,2) [E,N]
    sample_pts_utm: (n,2) [E,N]
    sample_values : (n,)
    return : (M,) prediksi
    """
    M = target_pts_utm.shape[0]
    n = sample_pts_utm.shape[0]
    out = np.full(M, np.nan, dtype=float)

    for s in range(0, M, chunk_size):
        e = min(M, s + chunk_size)
        P = target_pts_utm[s:e]  # (m,2)

        dx = P[:, [0]] - sample_pts_utm[:, 0][None, :]
        dy = P[:, [1]] - sample_pts_utm[:, 1][None, :]
        d = np.sqrt(dx*dx + dy*dy)

        take = min(k, n)
        idx = np.argpartition(d, take - 1, axis=1)[:, :take]

        for i in range(P.shape[0]):
            sel = idx[i]
            di = d[i, sel]
            if np.any(di == 0):
                out[s+i] = float(sample_values[sel[di == 0]][0]); continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                wi = 1.0 / np.power(di, p)
                out[s+i] = float(np.dot(wi, sample_values[sel]) / wi.sum())
    return out

# =========================
# 1) Load data & parameter
# =========================
print("1. Memuat data...")
samples = pd.read_csv(SAMPLES_CSV, dtype={"CC_4": str})
best = json.loads(BEST_JSON.read_text(encoding="utf-8"))
P = int(best["p"])
K = int(best["k"])
RADIUS = None if best["radius"] in (None, "None", 0) else float(best["radius"])

S_UTM = samples[["east","north"]].to_numpy(dtype=float)
V = samples["value"].to_numpy(dtype=float)

# =========================
# 2) Union boundary (WGS84)
# =========================
print("2. Menyiapkan batas wilayah (WGS84)...")
gj = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
geoms = [shape(ft["geometry"]) for ft in gj["features"]]
union_wgs = unary_union(geoms) # WGS84
union_wgs_prep = prep(union_wgs)

# =========================
# 3) Grid WGS84 + padding + rasterize mask
# =========================
print(f"3. Membuat grid WGS84 (resolusi ~{GRID_STEP_M} m) dengan padding {PAD_M} m...")
lon_min, lat_min, lon_max, lat_max = union_wgs.bounds
lat_center = (lat_min + lat_max) / 2.0

M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * np.cos(np.radians(lat_center))

dlat = GRID_STEP_M / M_PER_DEG_LAT
dlon = GRID_STEP_M / M_PER_DEG_LON
pad_lat = PAD_M / M_PER_DEG_LAT
pad_lon = PAD_M / M_PER_DEG_LON

# bounds dipadding (WGS84)
lon_min_p = lon_min - pad_lon
lon_max_p = lon_max + pad_lon
lat_min_p = lat_min - pad_lat
lat_max_p = lat_max + pad_lat

# ukuran grid (pix) dan transform georeferensi (origin di kiri-atas)
nx = int(np.ceil((lon_max_p - lon_min_p) / dlon))
ny = int(np.ceil((lat_max_p - lat_min_p) / dlat))
transform = from_origin(lon_min_p, lat_max_p, dlon, dlat)

# koordinat pusat piksel
xs = lon_min_p + (np.arange(nx) + 0.5) * dlon
ys = lat_max_p - (np.arange(ny) + 0.5) * dlat
XX_wgs, YY_wgs = np.meshgrid(xs, ys)
print(f"   - Dimensi grid: {nx} x {ny} piksel.")

# mask via rasterize (kunci: all_touched=True supaya nempel ke batas)
mask_grid = rasterize(
    [(union_wgs, 1)],
    out_shape=(ny, nx),
    transform=transform,
    fill=0,
    all_touched=True,
    dtype="uint8"
).astype(bool)

# titik pusat piksel yang akan diprediksi
flat_pts_wgs = np.stack([XX_wgs.ravel(), YY_wgs.ravel()], axis=1)
mask_inside_flat = mask_grid.ravel()
pts_to_predict_wgs = flat_pts_wgs[mask_inside_flat]

# =========================
# 4) Prediksi IDW (di UTM)
# =========================
print("4. Membuat mask dan menjalankan prediksi IDW...")
to_utm49s = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)
E, N = to_utm49s.transform(pts_to_predict_wgs[:,0], pts_to_predict_wgs[:,1])
pts_to_predict_utm = np.stack([E, N], axis=1)

pred = idw_predict_for_points(pts_to_predict_utm, S_UTM, V, p=P, k=K, radius=RADIUS)

Z = np.full(flat_pts_wgs.shape[0], np.nan, dtype=float)
Z[mask_inside_flat] = pred
Z = Z.reshape((ny, nx))
np.save(NPY_OUT, Z)

# =========================
# 5) Skala nilai & colormap
# =========================
print("5. Menskalakan nilai dan menyiapkan colormap...")
Z = np.where(np.isfinite(Z), Z, np.nan)
Z[Z < 0] = 0.0

if USE_FIXED_VMAX:
    vmin, vmax = FIXED_VMIN, FIXED_VMAX
else:
    vmin = 0.0
    finite = Z[np.isfinite(Z)]
    if finite.size > 0:
        vmax = float(np.percentile(finite, 95))
        if vmax <= vmin: vmax = float(np.nanmax(finite)) if np.isfinite(finite).any() else 1.0
    else:
        vmax = 1.0

print(f"   - Rentang nilai (vmin, vmax): ({vmin:.2f}, {vmax:.2f})")
Zn = (Z - vmin) / (vmax - vmin)
Zn[~np.isfinite(Zn)] = np.nan

# =========================
# 6) Bangun RGBA & simpan PNG
# =========================
print("6. Membuat gambar RGBA untuk overlay...")
vir = mpl.colormaps.get_cmap("viridis")
rgb = vir(np.nan_to_num(np.clip(Zn, 0, 1)))[:, :, :3]
rgb_u8 = (rgb * 255).astype(np.uint8)

alpha_filled = np.where(mask_grid & np.isfinite(Zn), int(255*0.9), 0).astype(np.uint8)
rgba_filled = np.dstack((rgb_u8, alpha_filled))
plt.imsave(PNG_FILLED, rgba_filled)
print(f"   - Gambar 'filled' disimpan ke: {PNG_FILLED}")

val_norm = np.nan_to_num(np.clip(Zn, 0, 1))
alpha_hot = np.power(val_norm, ALPHA_GAMMA)
alpha_hot[val_norm < ALPHA_CUTOFF_REL] = 0.0
alpha_hot = (alpha_hot * ALPHA_MAX) * mask_grid.astype(float)
alpha_hot_u8 = (alpha_hot * 255).astype(np.uint8)
rgba_hot = np.dstack((rgb_u8, alpha_hot_u8))
plt.imsave(PNG_HOTSPOT, rgba_hot)
print(f"   - Gambar 'hotspot' disimpan ke: {PNG_HOTSPOT}")

# =========================
# 7) Peta Folium
# =========================
print("7. Membuat peta Folium...")
img_bounds = [[lat_max_p, lon_min_p], [lat_min_p, lon_max_p]]
map_center = [(lat_min + lat_max)/2.0, (lon_min + lon_max)/2.0]

m = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB positron")

fg_filled = folium.FeatureGroup(name="IDW surface (filled)", show=False).add_to(m)
fg_hotspot = folium.FeatureGroup(name="IDW surface (hotspots)", show=True).add_to(m)

ImageOverlay(image=str(PNG_FILLED), bounds=img_bounds, opacity=1.0, name="IDW Filled").add_to(fg_filled)
ImageOverlay(image=str(PNG_HOTSPOT), bounds=img_bounds, opacity=1.0, name="IDW Hotspots").add_to(fg_hotspot)

folium.GeoJson(
    gj,
    name="Batas Wilayah",
    style_function=lambda x: {"color": "#3388ff", "weight": 1.2, "fillOpacity": 0.0}
).add_to(m)

colors = [to_hex(vir(i/6)) for i in range(7)]
legend = cm.LinearColormap(colors, vmin=vmin, vmax=vmax)
legend.caption = "Laju Kriminalitas per 100.000 Penduduk (IDW)"
legend.add_to(m)

folium.LayerControl().add_to(m)
m.save(str(HTML_OUT))
print(f"   - Peta disimpan ke: {HTML_OUT}")

# =========================
# 8) Metadata
# =========================
print("8. Menyimpan metadata...")
meta = {
    "grid_resolution_m": GRID_STEP_M,
    "padding_m": PAD_M,
    "grid_dimensions": {"nx": nx, "ny": ny},
    "bounds_wgs84": {
        "lat_min": float(lat_min), "lon_min": float(lon_min),
        "lat_max": float(lat_max), "lon_max": float(lon_max)
    },
    "bounds_wgs84_padded": {
        "lat_min": float(lat_min_p), "lon_min": float(lon_min_p),
        "lat_max": float(lat_max_p), "lon_max": float(lon_max_p)
    },
    "idw_params": {"p": P, "k": K, "radius": RADIUS},
    "scaling": {"vmin": float(vmin), "vmax": float(vmax)},
    "hotspot_params": {"alpha_max": ALPHA_MAX, "alpha_gamma": ALPHA_GAMMA, "alpha_cutoff_rel": ALPHA_CUTOFF_REL},
    "output_paths": {
        "filled_png": str(PNG_FILLED.relative_to(BASE)),
        "hotspot_png": str(PNG_HOTSPOT.relative_to(BASE)),
        "numpy_array": str(NPY_OUT.relative_to(BASE)),
        "html_map": str(HTML_OUT.relative_to(BASE))
    }
}
META_OUT.write_text(json.dumps(meta, indent=4), encoding="utf-8")
print(f"   - Metadata disimpan ke: {META_OUT}")
print("\nProses selesai.")