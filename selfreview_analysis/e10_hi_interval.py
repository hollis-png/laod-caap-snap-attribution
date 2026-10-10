"""E10: which implementation choice could explain our HI-interval CAAP (0.113) being above LAOD's reported 0.08?

LAOD ships no evaluation code, so its AP details are unknown. We vary the choices that are free in a re-implementation and see
whether any of them moves HI towards 0.08 while LO and MI stay near LAOD's 0.25 / 0.22:
  interp   'all'  : all-point interpolation (our baseline, VOC2010 style)
           '101'  : 101-point recall interpolation (COCO style)
  nms      class-agnostic NMS on the detections of each image before scoring (none / IoU 0.5 / 0.7)
  gt       with or without crowd boxes (iscrowd=1)
Also prints the per-threshold AP (10 values), so any other interval convention can be read off.
Baseline must reproduce CAAP LO/MI/HI = 0.2682 / 0.2309 / 0.1132.
Usage: python e10_hi_interval.py <preds.jsonl> <instances_val2017.json> <out.json>   (laod-exp venv, cwd ~/laod-exp)
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

def gt_boxes(with_crowd):
    g = {i: [] for i in ids}
    for a in coco["annotations"]:
        if a["image_id"] in g and (with_crowd or not a.get("iscrowd", 0)):
            x, y, w, h = a["bbox"]; g[a["image_id"]].append([x, y, x + w, y + h])
    return {k: np.array(v, float).reshape(-1, 4) for k, v in g.items()}

def iou_mat(b, g):
    if len(g) == 0: return np.zeros(0)
    ix = np.clip(np.minimum(b[2], g[:, 2]) - np.maximum(b[0], g[:, 0]), 0, None); iy = np.clip(np.minimum(b[3], g[:, 3]) - np.maximum(b[1], g[:, 1]), 0, None)
    I = ix * iy; U = (b[2]-b[0])*(b[3]-b[1]) + (g[:, 2]-g[:, 0])*(g[:, 3]-g[:, 1]) - I
    return np.where(U > 0, I / U, 0)

def nms(dets, thr):
    if thr is None: return dets
    dets = sorted(dets, key=lambda d: -d[4]); keep = []
    for d in dets:
        if all(iou_mat(np.array(d[:4]), np.array([k[:4]]))[0] < thr for k in keep): keep.append(d)
    return keep

def prepare(nms_thr):
    return {i: nms([d["box"] + [d["confidence"]] for d in preds[i]], nms_thr) for i in ids}

def ap_at(P, G, th, interp):
    pairs = []; ngt = 0
    for i in ids:
        g = G[i]; ngt += len(g); m = np.zeros(len(g), bool)
        for d in sorted(P[i], key=lambda d: -d[4]):
            v = iou_mat(np.array(d[:4]), g)
            if len(g): v = np.where(m, 0, v)
            j = int(np.argmax(v)) if len(v) else -1
            tp = j >= 0 and v[j] > 0 and v[j] >= th
            if tp: m[j] = True
            pairs.append((d[4], int(tp)))
    if ngt == 0 or not pairs: return 0.0
    pairs.sort(key=lambda x: -x[0]); tp = np.cumsum([p[1] for p in pairs]); fp = np.cumsum([1 - p[1] for p in pairs])
    prec = tp / (tp + fp); rec = tp / ngt; ip = np.maximum.accumulate(prec[::-1])[::-1]
    if interp == "all": return float(np.sum(ip * np.diff(np.concatenate(([0.0], rec)))))
    pts = np.linspace(0, 1, 101); out = []
    for r in pts:
        k = np.searchsorted(rec, r, side="left"); out.append(ip[k] if k < len(ip) else 0.0)
    return float(np.mean(out))

ALL = [round(0.5 + 0.05 * k, 2) for k in range(10)]
INT = {"LO": [.5, .55, .6], "MI": [.65, .7, .75, .8], "HI": [.85, .9, .95]}
res = {}
for crowd in (True, False):
    G = gt_boxes(crowd)
    for nms_thr in (None, 0.7, 0.5):
        P = prepare(nms_thr)
        for interp in ("all", "101"):
            per = {t: ap_at(P, G, t, interp) for t in ALL}
            key = f"crowd={int(crowd)}|nms={nms_thr}|interp={interp}"
            res[key] = {"per_threshold": {str(t): round(v, 4) for t, v in per.items()},
                        **{k: round(float(np.mean([per[t] for t in v])), 4) for k, v in INT.items()},
                        "mean": round(float(np.mean(list(per.values()))), 4),
                        "detections": sum(len(P[i]) for i in ids)}
            print(key, res[key]["LO"], res[key]["MI"], res[key]["HI"], res[key]["mean"], flush=True)
json.dump(res, open(out_p, "w"), indent=1)
