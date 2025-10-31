# scripts/15_plots_laju_pop.py
# Membuat:
# - outputs/figures/Gambar_4_5_scatter_laju_vs_penduduk.png
# - outputs/figures/Gambar_4_6_bar_top10_laju.png
# - outputs/tables/top10_laju_100k.csv
# - outputs/tables/laju_qc_diff.csv  (audit selisih laju)

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ------------------------- Konfigurasi Path -------------------------
BASE = Path(__file__).resolve().parents[1]
AGG_CSV = BASE / "outputs" / "data" / "aggregated_laju_100k.csv"
GJ_PATH  = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"

FIG_DIR = BASE / "outputs" / "figures"
TAB_DIR = BASE / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# --------------------- 1) Baca & siapkan data ----------------------
df = pd.read_csv(AGG_CSV, dtype={"CC_4": str})

needed_cols = {"CC_4", "kejadian_count", "penduduk", "laju_100k"}
missing = needed_cols - set(df.columns)
if missing:
    raise ValueError(f"Kolom hilang di {AGG_CSV.name}: {missing}")

# Ambil nama desa dari GeoJSON (fallback ke CC_4)
name_map = {}
if GJ_PATH.exists():
    try:
        gj = json.loads(GJ_PATH.read_text(encoding="utf-8"))
        for f in gj.get("features", []):
            props = f.get("properties", {}) or {}
            cc = str(props.get("CC_4", "")).strip()
            nm = (
                str(props.get("NAME_4", "")).strip()
                or str(props.get("DESA", "")).strip()
                or str(props.get("NAMDESA", "")).strip()
            )
            if cc and nm:
                name_map[cc] = nm
    except Exception as e:
        print(f"Peringatan: gagal memproses GeoJSON {GJ_PATH.name}: {e}")

df["nama_desa"] = df["CC_4"].map(name_map).fillna(df["CC_4"])

# Pastikan numerik
df["kejadian_count"] = pd.to_numeric(df["kejadian_count"], errors="coerce").fillna(0).astype(int)
df["penduduk"]      = pd.to_numeric(df["penduduk"], errors="coerce").fillna(0).astype(int)
df["laju_100k"]     = pd.to_numeric(df["laju_100k"], errors="coerce").fillna(0.0)

# Filter valid
dplot = df[(df["penduduk"] > 0)].copy()
if dplot.empty:
    raise ValueError("Tidak ada data valid (penduduk > 0) untuk di-plot.")

# ----------------- 2) Gambar 4.5 — Scatter + iso-kasus -----------------
# Hitung ulang laju agar titik presisi berada di kurva iso-kasus
dplot["laju_calc"] = 100000.0 * dplot["kejadian_count"] / dplot["penduduk"]

# --- SINGLE SOURCE OF TRUTH + QC ---
dplot["laju_use"] = dplot["laju_calc"]  # dipakai untuk semua grafik & tabel

# QC: selisih laju_100k (file) vs laju hitungan ulang
dplot["laju_diff"] = (dplot["laju_100k"] - dplot["laju_use"]).abs()
qc_csv = TAB_DIR / "laju_qc_diff.csv"
dplot.sort_values("laju_diff", ascending=False)[
    ["nama_desa", "CC_4", "kejadian_count", "penduduk",
     "laju_100k", "laju_use", "laju_diff"]
].to_csv(qc_csv, index=False, encoding="utf-8", float_format="%.6f")

max_diff = float(dplot["laju_diff"].max()) if not dplot["laju_diff"].empty else 0.0
if max_diff > 0.15:
    print(f"PERINGATAN QC: selisih laju_100k vs hitungan > 0.15 (maks={max_diff:.3f}). Lihat {qc_csv}.")
else:
    print(f"QC OK: selisih maksimum {max_diff:.3f}. Lihat {qc_csv}.")

# ----------------- Plot scatter -----------------
fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=150)

x_pop = dplot["penduduk"].to_numpy(dtype=float)
y_laju = dplot["laju_use"].to_numpy(dtype=float)
cases  = dplot["kejadian_count"].to_numpy(dtype=int)

# Plot titik
sizes = 15 + 25 * np.sqrt(np.clip(cases, 0, None))
ax.scatter(x_pop, y_laju, s=sizes, alpha=0.85, edgecolor="none", zorder=5)

# Skala log + batas paksa agar kurva membentang penuh sumbu-X
ax.set_xscale("log")
x_min_raw = float(np.nanmin(x_pop))
x_max_raw = float(np.nanmax(x_pop))
xmin = max(10.0, x_min_raw * 0.90)    # margin kiri 10%
xmax = x_max_raw * 1.10               # margin kanan 10%
ax.set_xlim(xmin, xmax)

# Domain kurva pada sumbu log (pakai geomspace agar mulus)
xx = np.geomspace(xmin, xmax, 600)

# Tentukan set garis iso-kasus (dinamis)
max_cases_actual = int(dplot["kejadian_count"].max())
iso_counts = list(range(1, min(max_cases_actual, 5) + 1))
if max_cases_actual > 5:
    extras = []
    if 6 <= max_cases_actual <= 9:
        extras = [max_cases_actual]
    elif 10 <= max_cases_actual <= 15:
        extras = [10, max_cases_actual]
    elif max_cases_actual > 15:
        extras = [10, 20, max_cases_actual]
    iso_counts = sorted(set(iso_counts + [c for c in extras if c <= max_cases_actual]))

# Posisi label stabil di ruang log
def geomed(a, b, t):
    return float(np.exp(np.log(a) * (1 - t) + np.log(b) * t))

for i, C in enumerate(iso_counts):
    yy = 100000.0 * C / xx
    ax.plot(xx, yy, linewidth=1.6, color="gray", linestyle="--", zorder=3)
    t = max(0.72, 0.90 - 0.02 * i)   # posisi label dekat kanan, tidak menumpuk
    xlab = geomed(xmin, xmax, t)
    ylab = 100000.0 * C / xlab
    ax.text(xlab, ylab, f"{C} kasus", fontsize=9, color="dimgray",
            ha="left", va="center", zorder=4)

# Anotasi titik yang kasusnya di atas garis tertinggi yang digambar
top_shown = max(iso_counts) if iso_counts else 0
if top_shown > 0 and max_cases_actual > top_shown:
    for xp, yp, kc in zip(x_pop, y_laju, cases):
        if kc > top_shown:
            xoff = min(xp * 1.03, xmax * 0.98)
            ha = "left" if xoff >= xp else "right"
            ax.text(xoff, yp, f"{kc} kasus", fontsize=9, color="black",
                    ha=ha, va="center", zorder=6)

ax.set_xlabel("Jumlah penduduk (skala log)")
ax.set_ylabel("Laju per 100.000 penduduk")
ax.set_title("Penduduk vs Laju/100k\n(dengan kurva referensi iso-kasus)")
ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.55)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(mticker.NullFormatter())

fig.tight_layout()
out1 = FIG_DIR / "Gambar_4_5_scatter_laju_vs_penduduk.png"
fig.savefig(out1, bbox_inches="tight")
plt.close(fig)

# ----------------- 3) Gambar 4.6 — Bar Top-10 Laju/100k -----------------
top10 = dplot.sort_values("laju_use", ascending=False).head(10).copy()
if top10.empty:
    print("Peringatan: Tidak ada data Top-10 untuk di-plot.")
else:
    top10 = top10.iloc[::-1]  # agar yang tertinggi di atas

    fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=150)
    y_pos = np.arange(len(top10))
    bars = ax.barh(y_pos, top10["laju_use"], height=0.6, zorder=5)

    labels = top10["nama_desa"].fillna(top10["CC_4"]).astype(str)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Laju per 100.000 penduduk")
    ax.set_title("Top-10 Desa/Kelurahan berdasarkan Laju/100k (2024)")
    ax.grid(axis="x", linestyle="--", alpha=0.6, zorder=0)

    max_laju = float(top10["laju_use"].max())
    ax.set_xlim(left=0, right=max_laju * 1.40)
    horizontal_padding = max_laju * 0.015

    for i, bar in enumerate(bars):
        laju = float(bar.get_width())
        row = top10.iloc[i]
        kasus = int(row["kejadian_count"])
        penduduk_fmt = f'{int(row["penduduk"]):,}'.replace(",", ".")
        label_text = f"{laju:.1f} | {kasus} kasus | {penduduk_fmt} jiwa"
        x_pos = laju + horizontal_padding
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                label_text, va="center", ha="left",
                fontsize=9.5, weight="bold", zorder=10)

    fig.tight_layout()
    out2 = FIG_DIR / "Gambar_4_6_bar_top10_laju.png"
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)
    print(f"- File bar chart   : {out2}")

# ----------------- 4) Simpan tabel Top-10 ke CSV -----------------
top10_to_save = dplot.sort_values("laju_use", ascending=False).head(10).copy()
out_tab = TAB_DIR / "top10_laju_100k.csv"
top10_to_save[["nama_desa", "CC_4", "kejadian_count", "penduduk", "laju_use"]].rename(
    columns={"laju_use": "laju_100k"}
).to_csv(out_tab, index=False, encoding="utf-8", float_format="%.2f")

# ------------------------------- Selesai ------------------------------
print("Proses selesai.")
print(f"- File scatter plot: {out1}")
if 'out2' not in locals():
    print("- File bar chart   : (Dilewati, tidak ada data)")
print(f"- File tabel CSV   : {out_tab}")
print(f"- File QC          : {qc_csv}")
