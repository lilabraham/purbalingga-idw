# scripts/14_make_summary_charts.py
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
AGG_CSV = BASE / "outputs" / "data" / "aggregated_laju_100k.csv"
OUT_DIR = BASE / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

assert AGG_CSV.exists(), (
    f"Tidak ketemu: {AGG_CSV} "
    f"(jalankan 02_ingest_points_centroid.py terlebih dahulu)"
)

# --- Baca data agregat desa
df = pd.read_csv(AGG_CSV)

# Deteksi kolom secara robust
def pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

col_rate  = pick(df.columns, ["laju_100k", "laju", "rate", "rate_100k", "laju_per_100k"])
col_count = pick(df.columns, ["kejadian_count", "count", "cases", "n_kasus"])
col_pop   = pick(df.columns, ["penduduk", "population", "pop"])

if col_rate is None or col_count is None:
    raise ValueError(f"Kolom wajib tidak ditemukan. Ada kolom: {df.columns.tolist()}")

# Casting aman
df[col_rate]  = pd.to_numeric(df[col_rate], errors="coerce")
df[col_count] = pd.to_numeric(df[col_count], errors="coerce")
if col_pop:
    df[col_pop] = pd.to_numeric(df[col_pop], errors="coerce")

df_valid = df.dropna(subset=[col_rate, col_count]).copy()

# --- Ringkasan ke terminal
rate = df_valid[col_rate].to_numpy()
cnt  = df_valid[col_count].to_numpy()
print(f"[INFO] Desa valid: {len(df_valid)}")
print(f"[INFO] Ringkasan laju/100k → min={np.nanmin(rate):.2f}, "
      f"median={np.nanmedian(rate):.2f}, "
      f"p95={np.nanpercentile(rate, 95):.2f}, "
      f"max={np.nanmax(rate):.2f}")
print(f"[INFO] Ringkasan jumlah kasus/desa → min={np.nanmin(cnt):.0f}, "
      f"median={np.nanmedian(cnt):.0f}, max={np.nanmax(cnt):.0f}")

# --- 1) Histogram laju/100k (untuk Gambar 4.2)
# Kunci vmin/vmax agar selaras dengan webmap (0–50)
vmin, vmax = 0, 50
bins = np.linspace(vmin, vmax, 21)  # lebar bin ~2.5

n_hi = int(np.sum(rate > vmax))  # berapa desa di atas 50 (akan terpotong visualnya)

plt.figure(figsize=(10, 5), dpi=200)
plt.hist(np.clip(rate, vmin, vmax), bins=bins, edgecolor="black")
plt.axvline(np.nanmedian(rate), linestyle="--", linewidth=1.5,
            label=f"Median = {np.nanmedian(rate):.2f}")
plt.axvline(np.nanmean(rate),   linestyle=":",  linewidth=1.5,
            label=f"Mean = {np.nanmean(rate):.2f}")
plt.title("Distribusi Laju Kriminalitas per 100.000 Penduduk (Desa, 2024)")
plt.xlabel("Laju per 100.000 (dikunci 0–50)")
plt.ylabel("Jumlah Desa")
if n_hi > 0:
    # anotasi kecil agar pembaca tahu ada nilai yang terpotong
    ymax = plt.gca().get_ylim()[1]
    plt.text(vmax - 1, ymax*0.92, f">{vmax}: {n_hi} desa",
             ha="right", va="top", fontsize=9)
plt.legend()
plt.tight_layout()
out_hist_rate = OUT_DIR / "Gambar_4_2_hist_laju_100k.png"
plt.savefig(out_hist_rate)
plt.close()

# --- 2) Boxplot laju/100k (alternatif Gambar 4.2)
plt.figure(figsize=(4.5, 5), dpi=200)
# Matplotlib 3.9+: gunakan tick_labels (bukan labels) untuk hindari warning
plt.boxplot(rate[~np.isnan(rate)], vert=True, showfliers=True, tick_labels=["Laju/100k"])
plt.ylim(vmin, vmax)
plt.title("Boxplot Laju Kriminalitas per 100.000 (Desa, 2024)")
plt.ylabel("Laju per 100.000 (0–50)")
plt.tight_layout()
out_box_rate = OUT_DIR / "Gambar_4_2_box_laju_100k.png"
plt.savefig(out_box_rate)
plt.close()

# --- 3) (Opsional) Histogram jumlah kasus per desa
bins_cnt = np.arange(0, max(1, math.ceil(np.nanmax(cnt))) + 1) - 0.5  # bin per integer
plt.figure(figsize=(10, 5), dpi=200)
plt.hist(cnt, bins=bins_cnt, edgecolor="black")
plt.title("Distribusi Jumlah Kasus per Desa (2024)")
plt.xlabel("Jumlah Kasus (desa)")
plt.ylabel("Jumlah Desa")
plt.xticks(np.arange(0, max(1, int(np.nanmax(cnt))) + 1, 1))
plt.tight_layout()
out_hist_cnt = OUT_DIR / "Gambar_4_2_hist_kejadian_count.png"
plt.savefig(out_hist_cnt)
plt.close()

print("[OK] Tersimpan:")
print(f"- {out_hist_rate}")
print(f"- {out_box_rate}")
print(f"- {out_hist_cnt} (opsional)")
