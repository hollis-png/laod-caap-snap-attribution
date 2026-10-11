"""E21: recompute the image-level CAAP/SNAP correlation and extremes quoted in §3.1, and SNAP at tau = 1.00 quoted in §2.2.

Uses the same functions as the original run (eval/run_batch_eval.py: build_eval_inputs, per_image_caap_snap) on the stored Gemma-3 + YOLO-World predictions.
  full split : 5,000 images (4,895 with detections enter the corpus; images without detections give CAAP = SNAP = 0 as in the original run)
  pilot      : gt.subset(200, seed=0), the 200 images of the pilot
  extremes   : largest per-image (SNAP - CAAP) and (CAAP - SNAP)
  snap_tau1  : corpus SNAP at tau = 1.00 with CLIP (compute_snap), the value the legacy HI interval of LAOD's paper would include
Usage (NDHU, cwd = folder containing eval/): python e21_perimage_r.py <predictions.jsonl> <instances_val2017.json> <images_dir> <out.json>
"""
import json, sys
import numpy as np

sys.path.insert(0, ".")
from eval.coco_loader import CocoGroundTruth
from eval.run_batch_eval import build_eval_inputs, per_image_caap_snap
from eval.snap import compute_snap

pred_p, ann_p, img_dir, out_p = sys.argv[1:5]
gt = CocoGroundTruth(ann_p, img_dir)
preds = {}
for l in open(pred_p, encoding="utf-8"):
    if l.strip():
        r = json.loads(l); preds[r["image_id"]] = r

def stats(rows):
    c = np.array([r["caap"] for r in rows]); s = np.array([r["snap"] for r in rows])
    return dict(n=len(rows), pearson_r=float(np.corrcoef(c, s)[0, 1]), max_snap_minus_caap=float((s - c).max()), max_caap_minus_snap=float((c - s).max()),
                mean_caap=float(c.mean()), mean_snap=float(s.mean()))

res = {}
ids, cp, cg, sp, sg = build_eval_inputs(gt, preds)
per = per_image_caap_snap(ids, cp, cg, sp, sg)
res["full"] = stats(per)
print("full", res["full"], flush=True)
pilot_ids = set(gt.subset(200, seed=0).image_ids)
res["pilot"] = stats([r for r in per if r["image_id"] in pilot_ids])
print("pilot", res["pilot"], flush=True)
res["snap_tau_1.00_corpus"] = float(compute_snap(sp, sg, [1.0]))
print("SNAP@1.00", res["snap_tau_1.00_corpus"], flush=True)
json.dump(res, open(out_p, "w"), indent=1)
