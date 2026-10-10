"""E3 step 3: evaluate a prediction file. Corrected corpus CAAP (per-image matching) and the section-3.6 vocabulary table
(detection-level CAAP-TP rate by vocabulary category; denominator = all detections, and also restricted to detections that overlap some GT)."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.expanduser("~/laod-exp"))
from eval.caap import compute_ca_ap
from eval.phase2_stage2a import vocabulary_category
pred_path, out_path = sys.argv[1], sys.argv[2]
coco = json.load(open(os.path.expanduser("~/laod-caap-check/instances_val2017.json"))); cat = {c["id"]: c["name"] for c in coco["categories"]}
preds = {}
for l in open(pred_path):
    if l.strip(): r = json.loads(l); preds[r["image_id"]] = r["detections"]
ids = sorted(i["id"] for i in coco["images"] if i["id"] in preds)
gts = {i: [] for i in ids}
for a in coco["annotations"]:
    if a["image_id"] in gts: x, y, w, h = a["bbox"]; gts[a["image_id"]].append([x, y, x + w, y + h, cat[a["category_id"]]])
P = [[d["box"] + [d["confidence"]] for d in preds[i]] for i in ids]; G = [gts[i] for i in ids]
out = {"images": len(ids), "detections": sum(len(p) for p in P)}
for name, th in (("CAAP_LO", [.5, .55, .6]), ("CAAP_MI", [.65, .7, .75, .8]), ("CAAP_HI", [.85, .9, .95]), ("CAAP_mean", [0.5 + 0.05 * k for k in range(10)])): out[name] = round(float(compute_ca_ap(P, G, th)), 4)
def iou(b, g):
    if len(g) == 0: return np.zeros(0)
    ix = np.clip(np.minimum(b[2], g[:, 2]) - np.maximum(b[0], g[:, 0]), 0, None); iy = np.clip(np.minimum(b[3], g[:, 3]) - np.maximum(b[1], g[:, 1]), 0, None)
    I = ix * iy; U = (b[2]-b[0])*(b[3]-b[1]) + (g[:, 2]-g[:, 0])*(g[:, 3]-g[:, 1]) - I; return np.where(U > 0, I / U, 0)
cnt = {k: [0, 0, 0] for k in ("native", "plural_of_native", "out_of_vocabulary")}   # n, tp, n_overlapping_some_gt
for i in ids:
    g = np.array([x[:4] for x in gts[i]], float).reshape(-1, 4)
    for d in preds[i]:
        v = iou(np.array(d["box"]), g); best = float(v.max()) if len(v) else 0.0
        c = cnt[vocabulary_category(d["label"].lower().strip())]; c[0] += 1; c[1] += best >= 0.5; c[2] += best > 0
out["vocab"] = {k: dict(n=v[0], tp=int(v[1]), rate=round(v[1] / v[0], 4), n_with_gt_match=int(v[2]), rate_matched_only=round(v[1] / max(1, v[2]), 4)) for k, v in cnt.items()}
nat, oov = out["vocab"]["native"], out["vocab"]["out_of_vocabulary"]
out["gap_all_denominator_pp"] = round(100 * (nat["rate"] - oov["rate"]), 1); out["gap_matched_denominator_pp"] = round(100 * (nat["rate_matched_only"] - oov["rate_matched_only"]), 1)
json.dump(out, open(out_path, "w"), indent=1); print(json.dumps(out, indent=1))
