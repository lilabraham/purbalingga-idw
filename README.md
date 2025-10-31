# Peta Kerawanan Kriminalitas Purbalingga (IDW)
- Data titik kasus: data/raw/kejadian/data_titik_kasus_2024.csv
- Batas desa: data/raw/boundary/desa_kasus.geojson
- Raster final: outputs/data/rasters/idw_surface_wgs84.tif
- Webmap final: outputs/html/07_webmap_final.html

Cara lihat peta:
1) python -m http.server
2) Buka http://localhost:8000/outputs/html/07_webmap_final.html

Regenerate raster & peta (jika data kasus di-update):
- python scripts\02_generate_maps.py
- python scripts\07_idw_surface.py
- python scripts\08_export_geotiff.py
- python scripts\11_webmap_final.py

1) python scripts\02_generate_maps.py
2) python scripts\05_make_idw_samples.py
3) python scripts\06_idw_loocv.py
4) python scripts\07_idw_surface.py
5) python scripts\08_export_geotiff.py
6) python scripts\11_webmap_final.py && python -m http.server

Urutan jalan:
aktifkakn venv : .\.venv\Scripts\Activate.ps1
python scripts\07_idw_surface.py
python scripts\08_export_geotiff.py
python scripts\11_webmap_final.py
python -m http.server
# buka http://localhost:8000/outputs/html/06_idw_surface.html



# Bersihkan hanya file turunan (aman). Jalankan: powershell -ExecutionPolicy Bypass -File .\clean_outputs.ps1
$ErrorActionPreference = "SilentlyContinue"

$paths = @(
  "outputs\data\aggregated_laju_100k.csv",
  "outputs\data\idw_samples.csv",
  "outputs\data\penduduk_fill_log.csv",
  "outputs\data\rasters\*",
  "outputs\figures\*",
  "outputs\html\06_idw_surface.html",
  "outputs\tables\*"
)

foreach ($p in $paths) { Remove-Item $p -Recurse }

Write-Host "Clean done."



# Jalankan end-to-end. Pakai: powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
$ErrorActionPreference = "Stop"

# 0) Env (opsional jika belum aktif)
# . .\.venv\Scripts\Activate.ps1

# 1) Ingest titik -> agregat & sampel IDW
python scripts/02_ingest_points_centroid.py

# 2) LOOCV (memastikan p, k, radius; hasil ringkasan & best params)
python scripts/06_idw_loocv.py

# 3) Bangun permukaan IDW + overlay untuk web
python scripts/07_idw_surface.py

# 4) Ekspor GeoTIFF (WGS84; NoData=-9999; clip batas; render PNG)
python scripts/08_export_geotiff.py

# 5) Webmap final (overlay inferno 0–50, batas, search, layer penduduk/kasus)
python scripts/11_webmap_final.py

# 6) Figur & tabel Bab 4 (scatter iso-kasus, Top-10, QC laju)
python scripts/15_plots_laju_pop.py

# 7) (opsional) Patch hotspot P95 jika dipakai di Bab 4.5
# python scripts/16_hotspot_patches.py

Write-Host "`nDONE. Buka webmap: outputs\html\06_idw_surface.html"
