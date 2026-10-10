"""E11: recompute the 200-image pilot (seed 0; ids parsed from run_200_log.txt) with the predictions of the full 5,000-image run.

The pilot of 2026-08 reported CAAP 0.2521 and SNAP 0.4382 (CLIP ViT-B/32, mean over tau 0.6-0.9) from its own LLM generation. The full run used a
second generation of the same pipeline. Scoring the full run's predictions on the same 200 images shows how much of the pilot-vs-full difference
is the choice of images and how much is run-to-run variation of the LLM.
Usage: python e11_pilot_subset.py <pilot_ids.txt> <full_preds.jsonl> <instances_val2017.json> <out.json>
(laod-exp venv, cwd ~/snapchk so that eval.snap_checkpoint_compare and eval.caap are importable)
"""
import json, sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from eval.caap import compute_ca_ap
from eval.coco_loader import CocoGroundTruth
import eval.snap_checkpoint_compare as sc

ids_p, pred_p, coco_p, out_p = sys.argv[1:5]
ids = [int(x) for x in open(ids_p).read().split()]
preds = {}
for l in open(pred_p):
    if l.strip():
        r = json.loads(l); preds[r["image_id"]] = r["detections"]
gt = CocoGroundTruth(coco_p, "/home/ndhu/laod-exp/coco/val2017")
ids = [i for i in ids if i in preds]
P = [[d["box"] + [d["confidence"]] for d in preds[i]] for i in ids]; G = [gt.gt_boxes_caap(i) for i in ids]
res = {"n_images": len(ids), "n_detections": sum(len(p) for p in P)}
res["CAAP_mean"] = round(float(compute_ca_ap(P, G, [0.5 + 0.05 * k for k in range(10)])), 4)
res["CAAP_LO"] = round(float(compute_ca_ap(P, G, [.5, .55, .6])), 4)
sp = [[[d["label"], d["confidence"]] for d in preds[i]] for i in ids]; sg = [gt.gt_labels_snap(i) for i in ids]
res["SNAP_B32_mean"] = round(sc.compute_snap(sp, sg, "openai/clip-vit-base-patch32"), 4)
res["pilot_reported"] = {"CAAP_mean": 0.2521, "SNAP_B32_mean": 0.4382}
json.dump(res, open(out_p, "w"), indent=1); print(json.dumps(res, indent=1))
