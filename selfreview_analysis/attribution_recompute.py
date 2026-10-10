"""Self-review recompute: decompose CAAP false positives with a non-vacuous semantic criterion.
Read-only on existing data; writes only to the --out dir given."""
import argparse, json, os, sys, csv
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.coco_loader import CocoGroundTruth
from eval.caap import calculate_iou
from eval.snap import embed_labels, cosine_similarity_matrix
from eval.phase2_stage2a import normalize_plural, vocabulary_category

ap = argparse.ArgumentParser()
ap.add_argument("--annotations", required=True); ap.add_argument("--images", required=True)
ap.add_argument("--predictions", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
gt = CocoGroundTruth(a.annotations, a.images)
preds = [json.loads(l) for l in open(a.predictions) if l.strip()]
labels = sorted({d["label"] for p in preds for d in p["detections"]} | set(gt.category_name_by_id.values()))
emb = dict(zip(labels, embed_labels(labels)))
rows = []
for p in preds:
    iid = p["image_id"]; boxes = gt.gt_boxes_caap(iid); gl = [b[4] for b in boxes]
    for d in p["detections"]:
        best, bi = 0.0, -1
        for i, b in enumerate(boxes):
            v = calculate_iou(d["box"], b[:4])
            if v > best: best, bi = v, i
        sims_img = [float(emb[d["label"]] @ emb[g]) for g in gl]
        max_sim_img = max(sims_img) if sims_img else 0.0
        near_cls = boxes[bi][4] if bi >= 0 else ""
        near_sim = float(emb[d["label"]] @ emb[near_cls]) if near_cls else 0.0
        norm = normalize_plural(d["label"].lower().strip())
        exact = int(bool(near_cls) and norm == near_cls)
        rows.append(dict(image_id=iid, label=d["label"], conf=d["confidence"], box=d["box"], best_iou=best,
                         caap_tp=int(best >= 0.5), max_sim_img=max_sim_img, near_cls=near_cls, near_sim=near_sim,
                         exact=exact, vocab=vocabulary_category(d["label"].lower().strip())))
N = len(rows); fp = [r for r in rows if not r["caap_tp"]]
out = {"N": N, "caap_tp": N - len(fp), "caap_fp": len(fp)}
# vacuity of SNAP threshold: share of ALL detections / FPs that count as SNAP-TP
for t in (0.6, 0.7, 0.8, 0.85, 0.9):
    out[f"snap_tp_rate_all@{t}"] = sum(r["max_sim_img"] >= t for r in rows) / N
    out[f"snap_tp_in_fp@{t}"] = sum(r["max_sim_img"] >= t for r in fp)
# decomposition of CAAP FPs
def bucket(r, t):
    if r["best_iou"] < 0.02: return "no_overlap"
    if r["exact"]: return "exact_class_loose_box"
    return "other_overlap_nearest_sim>=%.2f" % t if r["near_sim"] >= t else "other_overlap_nearest_sim<%.2f" % t
for t in (0.7, 0.8, 0.9):
    out[f"fp_buckets@{t}"] = dict(Counter(bucket(r, t) for r in fp))
out["legacy_rq3_near_miss_def(near_sim>=0.7)"] = sum(1 for r in fp if r["best_iou"] >= 0.02 and r["near_sim"] >= 0.7)
# which (label, nearest class) pairs dominate the legacy near-miss that are NOT exact class matches
pairs = Counter((r["label"], r["near_cls"]) for r in fp if r["best_iou"] >= 0.02 and r["near_sim"] >= 0.7 and not r["exact"])
out["top_non_exact_pairs"] = [[k[0], k[1], v] for k, v in pairs.most_common(40)]
json.dump(out, open(os.path.join(a.out, "recompute_report.json"), "w"), indent=1, ensure_ascii=False)
with open(os.path.join(a.out, "detections.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: r = dict(r); r["box"] = json.dumps(r["box"]); w.writerow(r)
print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])
