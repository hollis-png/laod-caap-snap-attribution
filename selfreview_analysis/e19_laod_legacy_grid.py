"""E19: re-summarise our per-IoU CAAP with the threshold grids of the LAOD v2 repository (laod/metrics/grids.py, 2026-10-06).

That file documents that the original paper's LO/MI/HI were built with np.arange(0.50, 0.65, 0.05) and similar calls. Floating point makes
(0.65 - 0.50) / 0.05 evaluate to 3.0000000000000004, so arange emits four values:
  legacy  LO = 0.50, 0.55, 0.60, 0.65   MI = 0.65, 0.70, 0.75, 0.80   HI = 0.85, 0.90, 0.95, 1.00
  v2      LO = 0.50, 0.55, 0.60         MI = 0.65 ... 0.80           HI = 0.85, 0.90, 0.95
Nothing can reach IoU 1.00 (LAOD's own table: CAAP@1.00 = 0.0001), so the legacy HI is deflated. We take AP@1.00 as 0 and as 0.0001 (bounds).
Input: e10_result.json, key 'crowd=1|nms=None|interp=all' (our baseline, per-threshold CAAP at the ten standard thresholds).
Usage: python e19_laod_legacy_grid.py e10_result.json e19_result.json
"""
import json, sys
import numpy as np

e10_p, out_p = sys.argv[1:3]
base = json.load(open(e10_p))["crowd=1|nms=None|interp=all"]["per_threshold"]
ap = {round(float(k), 4): v for k, v in base.items()}

PAPER_V1 = {"LO": 0.25, "MI": 0.22, "HI": 0.08}
GRIDS = {
    "v2": dict(LO=(0.50, 0.55, 0.60), MI=(0.65, 0.70, 0.75, 0.80), HI=(0.85, 0.90, 0.95)),
    "legacy": dict(LO=(0.50, 0.55, 0.60, 0.65), MI=(0.65, 0.70, 0.75, 0.80), HI=(0.85, 0.90, 0.95, 1.00)),
}

def summarise(grid, ap_at_1):
    a = {**ap, 1.0: ap_at_1}
    return {k: round(float(np.mean([a[t] for t in ts])), 4) for k, ts in grid.items()}

res = {"macro_ten_thresholds": round(float(np.mean(list(ap.values()))), 4), "paper_v1_reported": PAPER_V1}
for name, grid in GRIDS.items():
    for ap_at_1 in (0.0, 0.0001):
        res[f"{name}|AP@1.00={ap_at_1}"] = summarise(grid, ap_at_1)
legacy = res["legacy|AP@1.00=0.0"]
res["legacy_minus_paper_v1"] = {k: round(legacy[k] - v, 4) for k, v in PAPER_V1.items()}
for k, v in res.items():
    print(k, v)
json.dump(res, open(out_p, "w"), indent=1)
