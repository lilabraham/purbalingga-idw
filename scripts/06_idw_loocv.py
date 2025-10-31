# scripts/06_idw_loocv.py
from pathlib import Path
import json, math, time
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SAMPLES_CSV = BASE / "outputs" / "data" / "idw_samples.csv"
OUT_DIR     = BASE / "outputs" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

assert SAMPLES_CSV.exists(), f"Tidak ketemu: {SAMPLES_CSV} (jalankan 05_make_idw_samples.py dulu)"

df = pd.read_csv(SAMPLES_CSV)
df = df.dropna(subset=["east","north","value"]).reset_index(drop=True)

X = df[["east","north"]].to_numpy(float)  # meter (UTM 49S)
y = df["value"].to_numpy(float)           # laju/100k

# Grid parameter (boleh kamu ubah setelah run pertama)
POWERS    = [2, 3, 4, 5]
NEIGHBORS = [6, 8, 10, 12]
RADII     = [None, 2000, 4000]  # meter

def idw_predict_one(x0, X, y, p=2, k=8, radius=None):
    # hitung jarak euclidean (meter)
    d = np.sqrt(((X - x0)**2).sum(axis=1))
    # jika radius dipakai, pilih titik di dlm radius; kalau kurang, fallback ke k terdekat
    if radius is not None:
        mask = d <= radius
        idx_in = np.where(mask)[0]
        if len(idx_in) >= 3:  # minimal 3 titik
            dsub = d[idx_in]
            ysub = y[idx_in]
            # ambil sampai k terdekat di dalam radius
            take = min(k, len(idx_in))
            order = np.argsort(dsub)[:take]
            dsel, ysel = dsub[order], ysub[order]
        else:
            # fallback: k terdekat global
            order = np.argsort(d)[:k]
            dsel, ysel = d[order], y[order]
    else:
        order = np.argsort(d)[:k]
        dsel, ysel = d[order], y[order]

    # jika ada jarak 0 (seharusnya tidak untuk LOOCV), langsung kembalikan nilai aslinya
    if np.any(dsel == 0):
        return float(ysel[dsel == 0][0])

    w = 1.0 / np.power(dsel, p)
    ws = w.sum()
    if ws == 0 or np.isnan(ws):
        return np.nan
    return float(np.dot(w, ysel) / ws)

def loocv(p=2, k=8, radius=None):
    n = len(X)
    preds = np.empty(n, dtype=float)
    for i in range(n):
        # keluarkan titik i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pred = idw_predict_one(X[i], X[mask], y[mask], p=p, k=k, radius=radius)
        preds[i] = pred
    # metrik
    err = y - preds
    mae  = float(np.nanmean(np.abs(err)))
    rmse = float(math.sqrt(np.nanmean(err**2)))
    return rmse, mae, preds, err

rows = []
best = {"rmse": float("inf"), "mae": float("inf"), "p": None, "k": None, "radius": None}
t0 = time.time()

for p in POWERS:
    for k in NEIGHBORS:
        for radius in RADII:
            rmse, mae, preds, err = loocv(p=p, k=k, radius=radius)
            rows.append({"power": p, "neighbors": k, "radius_m": 0 if radius is None else radius,
                         "RMSE": rmse, "MAE": mae})
            # pilih terbaik by RMSE lalu MAE
            if (rmse < best["rmse"]) or (rmse == best["rmse"] and mae < best["mae"]):
                best.update({"rmse": rmse, "mae": mae, "p": p, "k": k, "radius": radius})
                # simpan error per titik utk kombo sementara terbaik
                best_err = pd.DataFrame({
                    "CC_4": df["CC_4"],
                    "NAME_4": df["NAME_4"],
                    "value_true": y,
                    "value_pred": preds,
                    "error": y - preds
                })

summary = pd.DataFrame(rows).sort_values(["RMSE","MAE","power","neighbors","radius_m"])
summary_path = OUT_DIR / "idw_loocv_summary.csv"
summary.to_csv(summary_path, index=False, encoding="utf-8")

best_path = OUT_DIR / "idw_best_params.json"
with open(best_path, "w", encoding="utf-8") as f:
    json.dump(best, f, ensure_ascii=False, indent=2)

best_err_path = OUT_DIR / "idw_loocv_errors_best.csv"
best_err.to_csv(best_err_path, index=False, encoding="utf-8")

elapsed = time.time() - t0
print("Selesai LOOCV.")
print("Ringkasan:", summary_path)
print("Best params:", best_path)
print("Errors (best):", best_err_path)
print(f"Waktu: {elapsed:.1f}s")
print("Best =", best)
