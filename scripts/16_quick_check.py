from pathlib import Path
import pandas as pd, time, os

BASE = Path(__file__).resolve().parents[1]
agg_p = BASE/"outputs/data/aggregated_laju_100k.csv"
smp_p = BASE/"outputs/data/idw_samples.csv"
html_p= BASE/"outputs/html/06_idw_surface.html"

def ts(p): return time.ctime(os.path.getmtime(p))

agg = pd.read_csv(agg_p, dtype={'CC_4':str})
smp = pd.read_csv(smp_p, dtype={'CC_4':str})

cc = "3303010011"  # Senon
print("== TIMESTAMP ==")
print("AGG :", ts(agg_p))
print("SAMP:", ts(smp_p))
print("HTML:", ts(html_p))

print("\n== AGG Senon ==")
print(agg.loc[agg['CC_4']==cc])

print("\n== SAMP Senon ==")
print(smp.loc[smp['CC_4']==cc, ['CC_4','NAME_4','value','lat','lon']])
