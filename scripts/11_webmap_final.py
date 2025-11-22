# scripts/11_webmap_final.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium.raster_layers import ImageOverlay
from folium.plugins import Search
import branca.colormap as cm
from matplotlib.colors import to_hex, LinearSegmentedColormap
import rasterio  # masih digunakan untuk cek TIF, walau bounds ambil dari meta

BASE = Path(__file__).resolve().parents[1]

# ---------- INPUT ----------
GEOJSON_PATH = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"
IDW_SAMPLES  = BASE / "outputs" / "data" / "idw_samples.csv"                    # titik/centroid per desa
AGG_CSV      = BASE / "outputs" / "data" / "aggregated_laju_100k.csv"           # penduduk & kasus (per desa)
META_PATH    = BASE / "outputs" / "data" / "rasters" / "idw_surface_meta.json"
TIF_PATH     = BASE / "outputs" / "data" / "rasters" / "idw_surface_wgs84.tif"  # utk cek saja
PNG_PATH     = BASE / "outputs" / "data" / "rasters" / "idw_surface_filled.png" # overlay indeks 0–1 (merah–putih)

# ---- HOTSPOT candidates ----
HOTSPOT_CANDIDATES = [
    BASE / "outputs" / "data" / "hotspots" / "hotspot_patches_p95.geojson",
    BASE / "outputs" / "data" / "hotspots" / "hotspots_p95.geojson",
    BASE / "outputs" / "data" / "hotspots_p95.geojson",
    BASE / "outputs" / "data" / "hotspot_p95.geojson",
]

# ---------- OUTPUT ----------
OUT_HTML_DIR = BASE / "outputs" / "html"
OUT_HTML_DIR.mkdir(parents=True, exist_ok=True)
HTML_OUT = OUT_HTML_DIR / "06_idw_surface.html"

# ---------- PARAM VISUAL ----------
OPACITY = 0.85

# kolom yang dipakai
CSV_KEY       = "CC_4"
CSV_NAME_COL  = "NAME_4"
CSV_VALUE_COL = "value"
CSV_LAT_COLS  = ["lat", "latitude", "Lat", "Latitude", "y", "Y"]
CSV_LON_COLS  = ["lon", "longitude", "Lon", "Longitude", "x", "X", "lng", "Lng"]

# properti kode desa umum pada GeoJSON
GJ_CODE_KEYS = ["CC_4", "cc_4", "KD_DESA", "KODE_DESA", "KODE", "kode_desa"]


def pick_lat_lon_cols(df: pd.DataFrame):
    lat = next((c for c in CSV_LAT_COLS if c in df.columns), None)
    lon = next((c for c in CSV_LON_COLS if c in df.columns), None)
    if not lat or not lon:
        raise KeyError(
            f"Tidak menemukan kolom lat/lon pada {IDW_SAMPLES.name}. "
            f"Kolom ada: {df.columns.tolist()}"
        )
    return lat, lon


def get_code_from_props(props: dict):
    for k in GJ_CODE_KEYS:
        if k in props and props[k] not in (None, "", "None", 0):
            return str(props[k]).strip()
    return None


def fmt_int_id(n):
    try:
        return f"{int(float(n)):,}".replace(",", ".")
    except Exception:
        return "—"


def fmt_float2(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "—"


def scale_radius(values, r_min=3, r_max=16):
    v = pd.to_numeric(pd.Series(values, dtype="float64"), errors="coerce").fillna(0.0)
    if (v > 0).sum() == 0:
        return [0 for _ in v]
    lo = np.percentile(v[v > 0], 5)
    hi = np.percentile(v[v > 0], 95)
    hi = max(hi, lo + 1e-9)
    norm = np.clip((v - lo) / (hi - lo), 0, 1)
    rad = r_min + np.sqrt(norm) * (r_max - r_min)
    rad[v <= 0] = 0
    return rad.tolist()


# ---------- 1) Cek file ----------
print("1. Memeriksa file input...")
assert GEOJSON_PATH.exists(), f"Boundary tidak ketemu: {GEOJSON_PATH}"
assert IDW_SAMPLES.exists(), f"Sampel CSV tidak ketemu: {IDW_SAMPLES}"
assert AGG_CSV.exists(), f"Aggregat tidak ketemu: {AGG_CSV}"
assert META_PATH.exists(), f"Meta raster tidak ketemu: {META_PATH}"
assert TIF_PATH.exists(), f"TIF tidak ketemu: {TIF_PATH} (jalankan 08)"
assert PNG_PATH.exists(), f"PNG render tidak ketemu: {PNG_PATH} (jalankan 07_idw_surface.py)"

# ---------- 2) Bounds dari META (harus sama dengan 07_idw_surface) ----------
print(f"2. Membaca bounds & skala dari Meta: {META_PATH.name}")
meta = json.loads(META_PATH.read_text(encoding="utf-8"))

# bounds WGS84 yang dipakai 07_idw_surface.py (padded)
bpad = meta.get("bounds_wgs84_padded", {})
lat_min_p = float(bpad.get("lat_min"))
lon_min_p = float(bpad.get("lon_min"))
lat_max_p = float(bpad.get("lat_max"))
lon_max_p = float(bpad.get("lon_max"))

# ini yang dipakai untuk ImageOverlay → harus persis dengan 07_idw_surface.py
img_bounds = [[lat_max_p, lon_min_p], [lat_min_p, lon_max_p]]
print(f"   - Bounds (padded, dari meta): {img_bounds}")

# Skala laju per 100k (informasi saja)
scaling = meta.get("scaling", {})
vmin_laju = float(scaling.get("vmin_laju", scaling.get("vmin", 0.0)))
vmax_laju = float(scaling.get("vmax_laju", scaling.get("vmax", 50.0)))

# Skala indeks 0–1 untuk legend
idx = meta.get("index_range", {})
idx_min = float(idx.get("min", 0.0))
idx_max = float(idx.get("max", 1.0))

print(f"   - Laju/100k: vmin={vmin_laju:.2f}, vmax={vmax_laju:.2f}")
print(f"   - Indeks kerawanan: min={idx_min:.2f}, max={idx_max:.2f}")

# ---------- 3) Data: samples (IDW), agregat (penduduk & kasus), GeoJSON ----------
print(f"3. Menggabungkan data GeoJSON dengan nilai dari {IDW_SAMPLES.name} & {AGG_CSV.name}")
gj = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))

df_samp = pd.read_csv(IDW_SAMPLES, dtype={CSV_KEY: str})
df_samp[CSV_KEY] = df_samp[CSV_KEY].astype(str).str.strip()

df_agg = pd.read_csv(AGG_CSV, dtype={CSV_KEY: str})
df_agg[CSV_KEY] = df_agg[CSV_KEY].astype(str).str.strip()

# lookups
val_lookup  = df_samp.set_index(CSV_KEY)[CSV_VALUE_COL].to_dict()
name_lookup = (
    df_samp.set_index(CSV_KEY)[CSV_NAME_COL].to_dict()
    if CSV_NAME_COL in df_samp.columns
    else {}
)
pop_lookup  = df_agg.set_index(CSV_KEY)["penduduk"].to_dict() if "penduduk" in df_agg.columns else {}
cnt_lookup  = df_agg.set_index(CSV_KEY)["kejadian_count"].to_dict() if "kejadian_count" in df_agg.columns else {}

# sisipkan properti untuk tooltip batas
features_joined = 0
for ft in gj.get("features", []):
    props = ft.get("properties", {}) or {}
    code = get_code_from_props(props)
    nm  = name_lookup.get(code, props.get("NAME_4") or props.get("DESA") or code or "N/A")
    val = fmt_float2(val_lookup.get(code)) if code in val_lookup else "—"
    ppl = fmt_int_id(pop_lookup.get(code)) if code in pop_lookup else "—"
    cnt = fmt_int_id(cnt_lookup.get(code)) if code in cnt_lookup else "—"
    props["display_name"]      = nm
    props["display_value"]     = val
    props["display_penduduk"]  = ppl
    props["display_kasus"]     = cnt
    if code in val_lookup:
        features_joined += 1
print(f"   - {features_joined} fitur GeoJSON berhasil dipasangkan nilai.")

# ---------- 4) Buat peta ----------
print("4. Membuat peta Folium...")
center = [
    (lat_min_p + lat_max_p) / 2.0,
    (lon_min_p + lon_max_p) / 2.0,
]
m = folium.Map(location=center, zoom_start=11, control_scale=True, tiles=None)

# Basemap
folium.TileLayer("cartodbpositron", name="Carto Light", control=True).add_to(m)
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google", name="Google Satellite", control=True
).add_to(m)
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google", name="Google Hybrid", control=True, show=False
).add_to(m)

# Raster overlay (harus align dengan batas karena pakai bounds meta)
ImageOverlay(
    image=str(PNG_PATH),
    bounds=img_bounds,
    opacity=OPACITY,
    name="Permukaan kerawanan (indeks 0–1)",
    zindex=2,
).add_to(m)

# ---------- HOTSPOT PATCH (opsional, robust) ----------
def pick_hotspot_path(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


HOTSPOT_GJ = pick_hotspot_path(HOTSPOT_CANDIDATES)
if HOTSPOT_GJ is None:
    print("   - Hotspot GeoJSON: TIDAK DITEMUKAN (lewati overlay patch).")
else:
    try:
        gj_hot = json.loads(HOTSPOT_GJ.read_text(encoding="utf-8"))
        feats = gj_hot.get("features", [])
        if not feats:
            print(f"   - Hotspot GeoJSON ada tetapi TANPA fitur: {HOTSPOT_GJ}")
        else:
            props0 = next(
                (ft.get("properties", {}) or {} for ft in feats if isinstance(ft, dict)),
                {},
            )
            thr_display = None
            for k in ("threshold_laju", "threshold", "cutoff"):
                if k in props0:
                    try:
                        thr_display = float(props0[k])
                    except Exception:
                        pass
                    break
            layer_name = (
                f"Hotspot P95 (≥{thr_display:.2f})"
                if isinstance(thr_display, (int, float))
                else "Hotspot P95 (patch)"
            )

            prop_keys = set(props0.keys())

            def pick_key(cands):
                for k in cands:
                    if k in prop_keys:
                        return k
                return None

            key_id   = pick_key(["patch_id", "id", "patch", "cluster_id"])
            key_area = pick_key(["area_km2", "luas_km2", "area_km", "area", "luas"])
            key_mean = pick_key(["value_mean", "mean", "mean_laju", "mean_val", "avg", "avg_laju"])
            key_max  = pick_key(["value_max", "max", "max_laju", "max_val", "puncak"])
            key_nv   = pick_key(["n_villages", "n_desa", "count_villages", "desa", "n"])

            fields, aliases = [], []
            for k, a in [
                (key_id,   "Patch:"),
                (key_area, "Luas (km²):"),
                (key_mean, "Mean laju:"),
                (key_max,  "Maks laju:"),
                (key_nv,   "Jumlah desa:"),
            ]:
                if k:
                    fields.append(k)
                    aliases.append(a)

            fg_hot = folium.FeatureGroup(name=layer_name, show=True).add_to(m)

            def _style_hot(_):
                return {
                    "color": "#ff4d4d",
                    "weight": 2,
                    "fillColor": "#ffcccb",
                    "fillOpacity": 0.22,
                }

            hot_geo = folium.GeoJson(
                gj_hot,
                style_function=_style_hot,
                highlight_function=lambda x: {
                    "weight": 3,
                    "color": "#ff0000",
                    "fillOpacity": 0.35,
                },
            ).add_to(fg_hot)

            if fields:
                folium.GeoJsonTooltip(
                    fields=fields,
                    aliases=aliases,
                    sticky=True,
                ).add_to(hot_geo)

            print(
                f"   - Hotspot GeoJSON: {HOTSPOT_GJ.name} | fitur={len(feats)} | layer='{layer_name}'"
            )
    except Exception as e:
        print(f"   - Gagal memuat hotspot: {e} (lewati overlay patch)")

# Layer batas desa + tooltip (dengan penduduk & kasus)
gj_layer = folium.GeoJson(
    gj,
    name="Batas Desa (hover)",
    style_function=lambda x: {
        "fillOpacity": 0.0,
        "weight": 1.2,
        "color": "#3BA3FF",
    },
    highlight_function=lambda x: {
        "weight": 3,
        "color": "#FFFF00",
        "fillOpacity": 0.1,
    },
).add_to(m)

folium.GeoJsonTooltip(
    fields=["display_name", "display_value", "display_kasus", "display_penduduk"],
    aliases=[
        "Desa/Kelurahan:",
        "Laju (per 100k):",
        "Kasus (2024):",
        "Penduduk (2024):",
    ],
    sticky=True,
    style=(
        "background-color: rgba(0,0,0,0.75);"
        "border: 1px solid black; border-radius: 3px; box-shadow: 3px;"
        "color: white; padding: 6px;"
    ),
    localize=True,
    labels=True,
).add_to(gj_layer)

# Pencarian by nama desa
Search(
    layer=gj_layer,
    geom_type="Polygon",
    search_label="display_name",
    placeholder="Cari desa…",
    collapsed=False,
).add_to(m)

# ---------- 5) Layer Titik & Nilai (hover) ----------
lat_col, lon_col = pick_lat_lon_cols(df_samp)
df_samp["_lat"] = pd.to_numeric(df_samp[lat_col], errors="coerce")
df_samp["_lon"] = pd.to_numeric(df_samp[lon_col], errors="coerce")
df_samp["_val"] = pd.to_numeric(df_samp[CSV_VALUE_COL], errors="coerce")

fg_points = folium.FeatureGroup(
    name="Titik & Nilai (hover)",
    show=True  # <-- default langsung tampil
).add_to(m)

for _, r in df_samp.dropna(subset=["_lat", "_lon", "_val"]).iterrows():
    code = str(r[CSV_KEY]).strip()
    nm   = (
        str(r.get(CSV_NAME_COL))
        if CSV_NAME_COL in df_samp.columns and pd.notna(r.get(CSV_NAME_COL))
        else code
    )
    tip_lines = [
        f"<b>{nm}</b>",
        f"Laju: {fmt_float2(r['_val'])} per 100k",
        f"Kasus (2024): {fmt_int_id(cnt_lookup.get(code)) if code in cnt_lookup else '—'}",
        f"Penduduk (2024): {fmt_int_id(pop_lookup.get(code)) if code in pop_lookup else '—'}",
    ]
    folium.CircleMarker(
        location=[float(r["_lat"]), float(r["_lon"])],
        radius=4,                 # sedikit lebih besar supaya kelihatan
        color="#111111",          # garis hitam pekat
        weight=1,
        fill=True,
        fill_color="#111111",     # isi hitam juga
        fill_opacity=0.95,
        tooltip=folium.Tooltip("<br>".join(tip_lines), sticky=True),
    ).add_to(fg_points)


# ---------- 6) Lingkaran Proporsional (opsional) ----------
df_samp["_penduduk"] = df_samp[CSV_KEY].map(pop_lookup).fillna(0).astype(float)
df_samp["_kasus"]    = df_samp[CSV_KEY].map(cnt_lookup).fillna(0).astype(float)

r_pop  = scale_radius(df_samp["_penduduk"], r_min=3, r_max=18)
r_case = scale_radius(df_samp["_kasus"],    r_min=3, r_max=18)

fg_pop = folium.FeatureGroup(name="Penduduk 2024 (lingkaran)", show=False).add_to(m)
for (_, r), rad in zip(df_samp.iterrows(), r_pop):
    if pd.isna(r["_lat"]) or pd.isna(r["_lon"]) or rad <= 0:
        continue
    nm  = name_lookup.get(str(r[CSV_KEY]).strip(), r.get(CSV_NAME_COL, r[CSV_KEY]))
    ppl = fmt_int_id(r["_penduduk"])
    tip = f"<b>{nm}</b><br>Penduduk (2024): {ppl}"
    folium.CircleMarker(
        location=[float(r["_lat"]), float(r["_lon"])],
        radius=float(rad),
        color="#1d6fb8",
        weight=1,
        fill=True,
        fill_color="#6aaed6",
        fill_opacity=0.55,
        tooltip=folium.Tooltip(tip, sticky=True),
    ).add_to(fg_pop)

fg_case = folium.FeatureGroup(name="Kasus 2024 (lingkaran)", show=False).add_to(m)
for (_, r), rad in zip(df_samp.iterrows(), r_case):
    if pd.isna(r["_lat"]) or pd.isna(r["_lon"]) or rad <= 0:
        continue
    nm  = name_lookup.get(str(r[CSV_KEY]).strip(), r.get(CSV_NAME_COL, r[CSV_KEY]))
    kss = fmt_int_id(r["_kasus"])
    tip = f"<b>{nm}</b><br>Kasus (2024): {kss}"
    folium.CircleMarker(
        location=[float(r["_lat"]), float(r["_lon"])],
        radius=float(rad),
        color="#b30000",
        weight=1,
        fill=True,
        fill_color="#ef3b2c",
        fill_opacity=0.55,
        tooltip=folium.Tooltip(tip, sticky=True),
    ).add_to(fg_case)

# ---------- Legend utama (kanan bawah) ----------
RISK_COLORS = ["#ffffff", "#fee5d9", "#cb181d"]
risk_cmap = LinearSegmentedColormap.from_list("risk_purbalingga", RISK_COLORS)
colors = [to_hex(risk_cmap(i / 4)) for i in range(5)]

legend = cm.LinearColormap(colors, vmin=idx_min, vmax=idx_max)
legend.caption  = "Indeks Kerawanan Kriminalitas (0–1)"
legend.position = "bottomright"
legend.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
m.save(str(HTML_OUT))

print(f"\nProses 11 selesai. Peta disimpan ke: {HTML_OUT}")
print("Tip: jalankan `python -m http.server` lalu buka http://localhost:8000/outputs/html/06_idw_surface.html")
