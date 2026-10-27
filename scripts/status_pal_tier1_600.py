"""Progress and running rates of the published pipeline on the Tier-1 (COSMOS-source) lenses at 600 s.

    python scripts/status_pal_tier1_600.py
"""
import json
import math
import shutil
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results/pyautolens_slam_tier1_600"


def rows(pop):
    f = R / pop / "scan_results.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


clean, sub_all = rows("no_subhalo"), rows("fixed60")
sub = [r for r in sub_all if r.get("log10_M200_true") is not None]
clean = clean + [r for r in sub_all if r.get("log10_M200_true") is None]   # the 10% of c=60 lenses without a subhalo are clean
hours = [r["seconds"] / 3600 for r in rows("no_subhalo") + sub_all]
print(f"done: {len(rows('no_subhalo'))}/50 clean, {len(sub_all)}/50 c=60 ({len(sub_all)-len(sub)} of them without a subhalo); median {sorted(hours)[len(hours) // 2]:.1f} h per lens" if hours else "none done")
if clean:
    fp = sum(r["delta_lnZ"] > 10 for r in clean)
    print(f"clean lenses flagged (dlnZ > 10): {fp}/{len(clean)}")
for lo, hi in ((9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0), (8.0, 9.0)):
    s = [r for r in sub if lo <= r["log10_M200_true"] < hi]
    if s:
        det = [r for r in s if r["delta_lnZ"] > 10]
        loc = sum(math.hypot(r["best_x"] - r["x_true"], r["best_y"] - r["y_true"]) <= 0.16 for r in det)
        print(f"c=60 10^{lo}-10^{hi}: detected {len(det)}/{len(s)}, of which {loc} within 2 px")
left = 100 - len(rows("no_subhalo")) - len(sub_all)
if hours and left:
    print(f"remaining {left} lenses; at 3 chains about {left * sorted(hours)[len(hours) // 2] / 3 / 24:.1f} days")
print(f"disk free: {shutil.disk_usage('/').free / 1e9:.0f} GB")
