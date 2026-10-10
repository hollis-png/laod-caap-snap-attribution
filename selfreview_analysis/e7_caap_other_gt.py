"""E7: corpus CAAP of the main run (Gemma-3 + YOLO-World X) scored against other ground-truth sets.

GT sets, all class-agnostic, per-image matching (the corrected compute_ca_ap):
  coco   instances_val2017.json (baseline; must reproduce CAAP mean 0.2068)
  rem    COCO-ReM val (refined COCO boxes + 7,217 added instances of the same 80 classes), all 5,000 images
  lvis   LVIS v1 val boxes (1,203 categories, federated), only the images that LVIS covers (4,809 of 5,000)
  union  COCO-ReM boxes plus every LVIS box whose IoU with all COCO-ReM boxes of that image is < 0.5 (LVIS images only)
For like-for-like comparison COCO and COCO-ReM are also scored on exactly the LVIS images ('*_on_lvis_images').
Adding ground truth cannot raise AP by itself: boxes the pipeline does not find become false negatives. The numbers therefore show
how much of the CAAP shortfall is annotation coverage, not that AP is the right metric for it (see E1 in REPORT.md).
Usage: python e7_caap_other_gt.py <preds.jsonl> <coco.json> <rem.json> <lvis.json> <out.json>   (run in the laod-exp venv, cwd ~/laod-exp)
"""
import json, sys, collections
import numpy as np
from eval.caap import compute_ca_ap

pred_path, coco_p, rem_p, lvis_p, out_p = sys.argv[1:6]
preds = {}
for l in open(pred_path):
    if l.strip():
        r = json.loads(l); preds[r["image_id"]] = r["detections"]

def load_gt(path):
    d = json.load(open(path)); g = collections.defaultdict(list)
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]; g[a["image_id"]].append([x, y, x + w, y + h, "obj"])
    return g, {i["id"] for i in d["images"]}

coco, coco_ids = load_gt(coco_p); rem, rem_ids = load_gt(rem_p); lvis, lvis_ids = load_gt(lvis_p)
ids_all = sorted(i for i in preds if i in coco_ids)
ids_lvis = [i for i in ids_all if i in lvis_ids]

def iou_vec(b, G):
    if not G: return np.zeros(0)
    g = np.array([x[:4] for x in G], float)
    ix = np.clip(np.minimum(b[2], g[:, 2]) - np.maximum(b[0], g[:, 0]), 0, None); iy = np.clip(np.minimum(b[3], g[:, 3]) - np.maximum(b[1], g[:, 1]), 0, None)
    I = ix * iy; U = (b[2]-b[0])*(b[3]-b[1]) + (g[:, 2]-g[:, 0])*(g[:, 3]-g[:, 1]) - I
    return np.where(U > 0, I / U, 0)

union = {}
for i in ids_lvis:
    base = list(rem.get(i, []))
    extra = [b for b in lvis.get(i, []) if not (len(iou_vec(b[:4], base)) and iou_vec(b[:4], base).max() >= 0.5)]
    union[i] = base + extra

THS = {"LO": [.5, .55, .6], "MI": [.65, .7, .75, .8], "HI": [.85, .9, .95], "mean": [0.5 + 0.05 * k for k in range(10)]}
def score(ids, gt):
    P = [[d["box"] + [d["confidence"]] for d in preds[i]] for i in ids]; G = [gt.get(i, []) for i in ids]
    out = {"images": len(ids), "detections": sum(len(p) for p in P), "gt_boxes": sum(len(g) for g in G)}
    for k, th in THS.items(): out[f"CAAP_{k}"] = round(float(compute_ca_ap(P, G, th)), 4)
    # detection-level hit rate at IoU 0.5 (any GT box of the same image), as in section 3.2
    hit = 0
    for i, g in zip(ids, G):
        for d in preds[i]:
            v = iou_vec(d["box"], g); hit += bool(len(v) and v.max() >= 0.5)
    out["detection_hit_rate_iou05"] = round(hit / max(1, out["detections"]), 4)
    return out

res = {"coco": score(ids_all, coco), "rem": score(ids_all, rem),
       "coco_on_lvis_images": score(ids_lvis, coco), "rem_on_lvis_images": score(ids_lvis, rem),
       "lvis": score(ids_lvis, lvis), "union_rem_lvis": score(ids_lvis, union)}
json.dump(res, open(out_p, "w"), indent=1); print(json.dumps(res, indent=1))
