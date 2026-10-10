"""Class-agnostic AP check. (a) replicates compute_ca_ap's matching, which iterates over ALL ground-truth boxes of ALL images;
(b) standard per-image matching (a detection can only match a GT box in its own image).
Same predictions as the main run. All-point interpolated AP, pooled over images, one IoU threshold at a time."""
import json, sys, time
import numpy as np
t0 = time.time()
coco = json.load(open("instances_val2017.json")); preds = {}
for l in open("coco_val_subset_predictions.jsonl"):
    if l.strip(): r = json.loads(l); preds[r["image_id"]] = r["detections"]
ids = sorted(i["id"] for i in coco["images"] if i["id"] in preds)
gt = {i: [] for i in ids}
for a in coco["annotations"]:
    if a["image_id"] in gt:
        x, y, w, h = a["bbox"]; gt[a["image_id"]].append((x, y, x + w, y + h, a["iscrowd"]))
D = [(d["confidence"], iid, d["box"]) for iid in ids for d in preds[iid]]
D.sort(key=lambda x: -x[0])
box = np.array([d[2] for d in D]); conf = np.array([d[0] for d in D]); img = np.array([d[1] for d in D])
G = {i: np.array(gt[i], dtype=float).reshape(-1, 5) for i in ids}
print("detections", len(D), "GT boxes", sum(len(v) for v in G.values()), "crowd", int(sum(v[:, 4].sum() for v in G.values())), flush=True)
def iou_row(b, g):
    ix = np.clip(np.minimum(b[2], g[:, 2]) - np.maximum(b[0], g[:, 0]), 0, None); iy = np.clip(np.minimum(b[3], g[:, 3]) - np.maximum(b[1], g[:, 1]), 0, None)
    I = ix * iy; U = (b[2]-b[0])*(b[3]-b[1]) + (g[:, 2]-g[:, 0])*(g[:, 3]-g[:, 1]) - I
    return np.where(U > 0, I / U, 0)
def ap_from_tp(tp, ngt):
    c = np.cumsum(tp); f = np.cumsum(1 - tp); p = c / (c + f); r = c / ngt
    ip = np.maximum.accumulate(p[::-1])[::-1]; return float(np.sum(ip * np.diff(np.concatenate(([0.0], r)))))
def per_image(th, use_crowd):
    matched = {i: np.zeros(len(G[i]), bool) for i in ids}; tp = np.zeros(len(D))
    for k in range(len(D)):
        g = G[img[k]]
        if len(g) == 0: continue
        ious = iou_row(box[k], g); ious[matched[img[k]]] = 0
        if not use_crowd: ious[g[:, 4] == 1] = 0
        j = int(np.argmax(ious))
        if ious[j] >= th and ious[j] > 0: tp[k] = 1; matched[img[k]][j] = True
    ngt = sum(len(v) if use_crowd else int((v[:, 4] == 0).sum()) for v in G.values()); return ap_from_tp(tp, ngt), tp.sum()
allg = np.concatenate([G[i] for i in ids])
def cross_image(th):
    matched = np.zeros(len(allg), bool); tp = np.zeros(len(D))
    for k in range(len(D)):
        ious = iou_row(box[k], allg); ious[matched] = 0; j = int(np.argmax(ious))
        if ious[j] >= th and ious[j] > 0: tp[k] = 1; matched[j] = True
    return ap_from_tp(tp, len(allg)), tp.sum()
groups = {"LO": [0.5, 0.55, 0.6], "MI": [0.65, 0.7, 0.75, 0.8], "HI": [0.85, 0.9, 0.95]}
res = {}
for name, fn in (("per_image_with_crowd_as_GT", lambda th: per_image(th, True)), ("per_image_crowd_excluded", lambda th: per_image(th, False)), ("cross_image (as in compute_ca_ap)", cross_image)):
    out = {}
    for g, ths in groups.items():
        vals = [fn(th) for th in ths]; out[g] = float(np.mean([v[0] for v in vals])); 
        print(f"[{time.time()-t0:5.0f}s] {name:36s} {g}: {out[g]:.4f}   (TP at first thr: {vals[0][1]:.0f})", flush=True)
    out["mean .5:.95"] = (3 * out["LO"] + 4 * out["MI"] + 3 * out["HI"]) / 10; res[name] = out
    print(f"    {name}: mean over .5:.95 = {out['mean .5:.95']:.4f}", flush=True)
print("LAOD reported: LO 0.25 MI 0.22 HI 0.08"); json.dump(res, open("caap_check_result.json", "w"), indent=1)
