# scripts/01_validate_and_clean_population.py
import json, re
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

GEOJSON_PATH = BASE / "data" / "raw" / "boundary" / "desa_kasus.geojson"
PENDUDUK_XLSX = BASE / "data" / "raw" / "penduduk" / "penduduk_bersih.xlsx"

OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)
OUT_CLEAN       = BASE / "data" / "processed" / "penduduk_clean_for_join.csv"
OUT_DUP         = OUT_DIR / "penduduk_duplicates.csv"
OUT_UNMAPPED    = OUT_DIR / "penduduk_unmapped.csv"
OUT_REPORT_TXT  = OUT_DIR / "penduduk_validasi.txt"

NAME_KEYS = ["NAME_4","NAME","DESA","DESA_KEL","NAMDESA","nama","nama_desa","desa_kel"]
CODE_KEYS = ["CC_4","KODE","KODE_DESA","KD_DESA","kode_desa","cc_4"]

def norm_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.replace("desa ", "").replace("kelurahan ", "")

def get(props: dict, keys: list):
    for k in keys:
        if k in props and props[k] not in (None, "", 0, "None"):
            return props[k]
    return None

def main():
    assert GEOJSON_PATH.exists(), f"Tidak ketemu: {GEOJSON_PATH}"
    assert PENDUDUK_XLSX.exists(), f"Tidak ketemu: {PENDUDUK_XLSX}"

    # 1) Boundary → kamus CC_4 <-> NAME_4
    gj = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    cc4_to_name, name_to_cc4 = {}, {}
    for ft in gj.get("features", []):
        props = ft.get("properties", {})
        cc4 = str(get(props, CODE_KEYS))
        nm  = str(get(props, NAME_KEYS) or cc4)
        if cc4 and cc4 != "None":
            cc4_to_name[cc4] = nm
            name_to_cc4[norm_name(nm)] = cc4
    boundary_cc4 = set(cc4_to_name.keys())

    # 2) Baca Excel penduduk (pilih sheet paling relevan)
    xl = pd.ExcelFile(PENDUDUK_XLSX)
    sheet = xl.sheet_names[0]
    for s in xl.sheet_names:
        if any(k in s.lower() for k in ["penduduk","population","kependudukan","desa","kelurahan","data"]):
            sheet = s; break
    df_raw = xl.parse(sheet)

    # 3) Deteksi kolom
    kode_col = next((c for c in df_raw.columns if c in CODE_KEYS or str(c).lower() in ["cc_4","kode desa","kode_desa","kd desa","kd_desa"]), None)
    name_col = next((c for c in df_raw.columns if str(c).lower() in ["name_4","name","desa","nama_desa","kelurahan","desa_kel","nama"]), None)
    pend_col = next((c for c in df_raw.columns if any(k in str(c).lower() for k in ["penduduk","population","populasi","jumlah"])), None)
    if pend_col is None:
        raise ValueError("Kolom jumlah penduduk tidak ditemukan. Harus ada 'penduduk'/'population'/'jumlah'.")

    # 4) Bersihkan angka penduduk → integer
    df = df_raw.copy()
    df["penduduk"] = (
        df[pend_col].astype(str)
        .str.replace(r"[^\d\-]", "", regex=True)
        .replace("", "0").astype(int)
    )

    # 5) Bangun kolom CC_4
    if kode_col is not None:
        df["CC_4"] = df[kode_col].astype(str).str.extract(r"(\d{10,13})", expand=False)
    else:
        df["CC_4"] = None

    if df["CC_4"].isna().all():
        if name_col is None:
            raise ValueError("Tidak ada CC_4 dan kolom nama desa untuk pemetaan.")
        df["CC_4"] = df[name_col].astype(str).map(lambda x: name_to_cc4.get(norm_name(x)))

    # 6) Laporan masalah
    dups = df[df["CC_4"].duplicated(keep=False)].sort_values("CC_4")
    unmapped = df[df["CC_4"].isna()]
    if not dups.empty: dups.to_csv(OUT_DUP, index=False, encoding="utf-8")
    if not unmapped.empty: unmapped.to_csv(OUT_UNMAPPED, index=False, encoding="utf-8")

    # 7) Satukan per CC_4 (jumlahkan penduduk)
    clean = (df.dropna(subset=["CC_4"])
               .groupby("CC_4", as_index=False)["penduduk"].sum())
    clean["NAME_4"] = clean["CC_4"].map(cc4_to_name)
    clean.to_csv(OUT_CLEAN, index=False, encoding="utf-8")

    # 8) Report ringkas
    coverage = (clean["CC_4"].isin(boundary_cc4)).mean()*100 if not clean.empty else 0.0
    report = [
        f"Sheet dipakai: {sheet}",
        f"Deteksi kolom -> kode: {kode_col!r}, nama: {name_col!r}, penduduk: {pend_col!r}",
        f"Cakupan CC_4 (match boundary): {coverage:.2f}%",
        f"Duplikat CC_4 (baris asli): {dups['CC_4'].nunique()} kode; total baris: {len(dups)}",
        f"Baris tanpa CC_4 (asli): {len(unmapped)}",
        f"Output clean: {OUT_CLEAN}",
        f"Issue files: {OUT_DUP if OUT_DUP.exists() else '-'}, {OUT_UNMAPPED if OUT_UNMAPPED.exists() else '-'}"
    ]
    OUT_REPORT_TXT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))

if __name__ == "__main__":
    main()
