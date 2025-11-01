@'
# Purbalingga IDW 2024

Repositori ini memuat pipeline pemetaan kerawanan berbasis **Inverse Distance Weighting (IDW)** untuk Kabupaten Purbalingga, fokus tahun **2024** dengan indikator **laju per 100.000 penduduk**.

## Parameter & Metodologi
- Metode interpolasi: **IDW**
- Power: **p = 2**
- Tetangga terdekat: **k = 12** (tanpa radius)
- Validasi: **LOOCV (Leave-One-Out Cross-Validation)**
- Sistem koordinat jarak: **UTM Zona 49S (meter)**
- Jarak: **Euclidean**
- Visualisasi raster: rentang dikunci **0–50** (nilai >50 disaturasi ke kelas tertinggi)

## Struktur Folder (ringkas)
- `data/raw/` : boundary & data mentah
- `data/processed/` : hasil join/normalisasi (jika ada)
- `outputs/data/` : agregat, sampel IDW, raster
- `outputs/html/` : webmap
- `scripts/` : skrip pemrosesan (mis. 02_ingest_points_centroid.py, 07_idw_surface.py)
- `reports/` : dokumen pendukung (opsional)

## Menjalankan Pipeline (contoh)
.\.venv\Scripts\Activate.ps1

> Jalankan dari root repo (aktifkan environment Python lebih dulu).
```bash
# 1) Ingest titik → join ke desa, hitung laju/100k per desa
python scripts/02_ingest_points_centroid.py

# 2) Penalaan & Interpolasi (p=2, k=12 dari LOOCV) → raster IDW
python scripts/07_idw_surface.py

# 3) Render webmap (overlay PNG inferno 0–50, klip batas, ekspor WGS84)
python scripts/08_render_webmap.py
