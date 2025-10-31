# scripts/14_make_param_table.py
from __future__ import annotations
import json
from pathlib import Path
import math
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

# ---- Inputs
BEST_JSON         = BASE / "outputs" / "data" / "idw_best_params.json"
LOOCV_ERRORS_CSV  = BASE / "outputs" / "data" / "idw_loocv_errors_best.csv"
META_JSON         = BASE / "outputs" / "data" / "rasters" / "idw_surface_meta.json"  # optional

# ---- Outputs
OUT_DIR           = BASE / "outputs" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT           = OUT_DIR / "idw_param_table.csv"
MD_OUT            = OUT_DIR / "idw_param_table.md"

# ---- Helpers
OBS_CANDS   = ["obs", "observed", "y_true", "true", "actual", "nilai_asli", "nilai_aktual"]
PRED_CANDS  = ["pred", "predicted", "y_pred", "preds", "estimate", "estimasi", "fitted", "hat", "nilai_prediksi"]
RES_CANDS   = ["res", "residual", "resid", "error", "err", "e"]

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lc_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lc_map:
            return lc_map[c.lower()]
    return None

def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer()):
        return str(int(x))
    return f"{float(x):.{nd}f}"

# ---- Load best params
best = {}
if BEST_JSON.exists():
    try:
        best = json.loads(BEST_JSON.read_text(encoding="utf-8"))
    except Exception:
        best = {}
p       = best.get("p")
k       = best.get("k")
radius  = best.get("radius")
rmse_b  = best.get("rmse")
mae_b   = best.get("mae")

# ---- Load legend/meta (optional)
vmin = vmax = None
if META_JSON.exists():
    try:
        meta = json.loads(META_JSON.read_text(encoding="utf-8"))
        sc = meta.get("scaling", {})
        vmin = sc.get("vmin", None)
        vmax = sc.get("vmax", None)
    except Exception:
        pass

# ---- Read LOOCV errors and auto-detect columns
if not LOOCV_ERRORS_CSV.exists():
    raise FileNotFoundError(f"Tidak menemukan {LOOCV_ERRORS_CSV}")

df = pd.read_csv(LOOCV_ERRORS_CSV)

obs_col  = pick_col(df, OBS_CANDS)
pred_col = pick_col(df, PRED_CANDS)

# Try to reconstruct from residuals if one of the two is missing
res_col  = pick_col(df, RES_CANDS)

if obs_col is None and pred_col is None:
    # Last resort: pick two most-numeric columns
    numeric_score = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        frac_num = s.notna().mean()
        numeric_score.append((frac_num, c))
    numeric_score.sort(reverse=True)
    numeric_cols = [c for _, c in numeric_score if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
    if len(numeric_cols) >= 2:
        obs_col, pred_col = numeric_cols[:2]
elif obs_col is None and pred_col is not None and res_col is not None:
    # obs = pred - residual
    df["_pred_tmp"] = pd.to_numeric(df[pred_col], errors="coerce")
    df["_res_tmp"]  = pd.to_numeric(df[res_col],  errors="coerce")
    df["_obs_tmp"]  = df["_pred_tmp"] - df["_res_tmp"]
    obs_col = "_obs_tmp"
elif pred_col is None and obs_col is not None and res_col is not None:
    # pred = obs + residual
    df["_obs_tmp"]  = pd.to_numeric(df[obs_col],  errors="coerce")
    df["_res_tmp"]  = pd.to_numeric(df[res_col],  errors="coerce")
    df["_pred_tmp"] = df["_obs_tmp"] + df["_res_tmp"]
    pred_col = "_pred_tmp"

if obs_col is None or pred_col is None:
    raise ValueError(
        f"Tidak menemukan kolom observed/predicted di {LOOCV_ERRORS_CSV.name}. "
        f"Cari salah satu dari {OBS_CANDS} dan {PRED_CANDS}, atau sertakan residual ({RES_CANDS})."
    )

# ---- Compute metrics from CSV
obs  = pd.to_numeric(df[obs_col],  errors="coerce")
pred = pd.to_numeric(df[pred_col], errors="coerce")
mask = obs.notna() & pred.notna()
n    = int(mask.sum())
if n == 0:
    raise ValueError("Tidak ada baris numerik yang dapat dihitung untuk RMSE/MAE.")

err  = (pred - obs)[mask]
rmse = float((err**2).mean()**0.5)
mae  = float(err.abs().mean())

# Optional MAPE & coverage
nz_mask = mask & (obs != 0)
mape = float((err[nz_mask].abs() / obs[nz_mask].abs()).mean() * 100) if int(nz_mask.sum()) > 0 else None
mape_cov = float(nz_mask.sum() / n * 100)

# ---- Compose table
rows = [
    ("Metode",                    "Inverse Distance Weighting (IDW)"),
    ("Power (p)",                 fmt(p)),
    ("Tetangga (k)",              fmt(k)),
    ("Radius",                    "-" if radius in (None, "None", 0, "0") else fmt(radius)),
    ("Sampel LOOCV (n)",          fmt(n, 0)),
    ("RMSE (LOOCV)",              fmt(rmse)),
    ("MAE (LOOCV)",               fmt(mae)),
]

# If JSON had metrics, show them as reference
if rmse_b is not None or mae_b is not None:
    rows += [
        ("RMSE (best_params.json)", fmt(rmse_b)),
        ("MAE  (best_params.json)", fmt(mae_b)),
    ]

if mape is not None:
    rows += [("MAPE (%, pada desa dengan laju>0)", f"{fmt(mape)} (coverage {fmt(mape_cov)}%)")]

if vmin is not None or vmax is not None:
    rows += [("Legenda kontinu (vmin–vmax)", f"{fmt(vmin)} – {fmt(vmax)}")]

# Save CSV & Markdown
pd.DataFrame(rows, columns=["Komponen", "Nilai"]).to_csv(CSV_OUT, index=False, encoding="utf-8")

md_lines = [
    "| Komponen | Nilai |",
    "|---|---|",
]
for k_, v_ in rows:
    md_lines.append(f"| {k_} | {v_} |")

MD_OUT.write_text("\n".join(md_lines), encoding="utf-8")

# Pretty print to console
print("\nRingkasan Parameter & Kinerja IDW")
print("-" * 38)
for k_, v_ in rows:
    print(f"{k_:30s}: {v_}")

print(f"\nTersimpan:")
print(f" - {CSV_OUT}")
print(f" - {MD_OUT}")
