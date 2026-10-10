"""E16: does LAOD's Eq. 7 definition of CAAP explain our HI interval (0.113) being above LAOD's reported 0.08?

LAOD (arXiv:2507.10844, section 4.1) writes CAAP as the area under the precision-recall curve (Eq. 5) but defines the per-IoU value as
  CAAP(theta_iou) = 1/|Theta_s| * sum_{theta_s in Theta_s} P(theta_iou, theta_s)          (Eq. 7)
i.e. the mean of the precision of the detections with confidence >= theta_s, over a set of confidence thresholds Theta_s that the text does not
give. That is a different number from AP (no recall weighting). This script computes the Eq. 7 version for several guesses of Theta_s, with the
same one-to-one greedy matching as e10_hi_interval.py, and the LAOD interval boundaries stated in the paper (LO 0.50-0.60, MI 0.65-0.80, HI 0.85-0.95).
Two ways to treat a theta_s above every confidence (no detection left): precision 0 ("zero") or leave it out of the mean ("skip").
Baseline AP (interp 'all') is also recomputed and must reproduce 0.2682 / 0.2309 / 0.1132.
Usage: python e16_eq7_caap.py <preds.jsonl> <instances_val2017.json> <out.json>
"""
import json, sys
import numpy as np

pred_p, coco_p, out_p = sys.argv[1:4]
coco = json.load(open(coco_p))
preds = {}
for l in open(pred_p):
    if l.strip():
        r = json.loads(l); preds[r["image_id"]] = r["detections"]
ids = sorted(i["id"] for i in coco["images"] if i["id"] in preds)
G = {i: [] for i in ids}
for a in coco["annotations"]:
    if a["image_id"] in G:
        x, y, w, h = a["bbox"]; G[a["image_id"]].append([x, y, x + w, y + h])
G = {k: np.array(v, float).reshape(-1, 4) for k, v in G.items()}

def iou_vec(b, g):
    if len(g) == 0: return np.zeros(0)
    ix = np.clip(np.minimum(b[2], g[:, 2]) - np.maximum(b[0], g[:, 0]), 0, None); iy = np.clip(np.minimum(b[3], g[:, 3]) - np.maximum(b[1], g[:, 1]), 0, None)
    I = ix * iy; U = (b[2]-b[0])*(b[3]-b[1]) + (g[:, 2]-g[:, 0])*(g[:, 3]-g[:, 1]) - I
    return np.where(U > 0, I / U, 0)

def match(th):
    """(confidence, tp) for all detections of all images, sorted by confidence descending; greedy one-to-one per image."""
    pairs = []; ngt = 0
    for i in ids:
        g = G[i]; ngt += len(g); m = np.zeros(len(g), bool)
        for d in sorted(preds[i], key=lambda d: -d["confidence"]):
            v = iou_vec(np.array(d["box"], float), g)
            if len(g): v = np.where(m, 0, v)
            j = int(np.argmax(v)) if len(v) else -1
            tp = j >= 0 and v[j] > 0 and v[j] >= th
            if tp: m[j] = True
            pairs.append((d["confidence"], int(tp)))
    pairs.sort(key=lambda x: -x[0])
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]), ngt

def ap_all(conf, tp, ngt):
    ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp); prec = ctp / (ctp + cfp); rec = ctp / ngt
    ip = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(ip * np.diff(np.concatenate(([0.0], rec)))))

def eq7(conf, tp, thetas, empty):
    vals = []
    for s in thetas:
        k = int(np.sum(conf >= s))
        if k == 0:
            if empty == "zero": vals.append(0.0)
            continue
        vals.append(float(tp[:k].sum()) / k)
    return float(np.mean(vals)) if vals else 0.0

ALL = [round(0.5 + 0.05 * k, 2) for k in range(10)]
INT = {"LO": [.5, .55, .6], "MI": [.65, .7, .75, .8], "HI": [.85, .9, .95]}
THETAS = {
    "s=0.1..0.9 step 0.1": [round(0.1 * k, 1) for k in range(1, 10)],
    "s=0.05..0.95 step 0.05": [round(0.05 * k, 2) for k in range(1, 20)],
    "s=0.0..0.9 step 0.1": [round(0.1 * k, 1) for k in range(0, 10)],
    "s=0.01..0.99 step 0.01": [round(0.01 * k, 2) for k in range(1, 100)],
}
m = {t: match(t) for t in ALL}
conf0 = m[0.5][0]
res = {"confidence_quantiles(0,5,25,50,75,95,100%)": [round(float(q), 4) for q in np.percentile(conf0, [0, 5, 25, 50, 75, 95, 100])],
       "detections": int(len(conf0)), "n_gt": int(m[0.5][2])}
def row(per):
    return {**{k: round(float(np.mean([per[t] for t in v])), 4) for k, v in INT.items()}, "per_threshold": {str(t): round(v, 4) for t, v in per.items()}}
res["AP_all_point (our baseline)"] = row({t: ap_all(*m[t]) for t in ALL})
for name, th in THETAS.items():
    for empty in ("zero", "skip"):
        res[f"Eq7 {name} | empty={empty}"] = row({t: eq7(m[t][0], m[t][1], th, empty) for t in ALL})
for k, v in res.items():
    if isinstance(v, dict): print(f"{k:55s} LO {v['LO']}  MI {v['MI']}  HI {v['HI']}")
    else: print(k, v)
json.dump(res, open(out_p, "w"), indent=1)
