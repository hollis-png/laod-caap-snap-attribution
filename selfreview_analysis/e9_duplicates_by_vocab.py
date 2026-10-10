"""E9: are the duplicate detections (detection-level hits that the one-to-one AP matcher does not count) concentrated in one vocabulary category?

Section 3.2 uses a detection-level rule (IoU >= 0.5 with any ground-truth box of the image: 15,134 hits) while AP matches one-to-one
(14,250 true positives); the 884 difference are duplicates of an already matched box. This script reproduces both counts and splits them
by vocabulary category (native / plural of native / out of vocabulary), then recomputes the native-vs-OOV hit-rate gap under one-to-one
matching, to show whether the main finding depends on the detection-level rule.
Matching replicates e6_soft_ap.py: all detections of the corpus in global descending confidence, per image the unmatched ground-truth
box with the highest IoU, threshold 0.5, crowd boxes included.
Usage: python e9_duplicates_by_vocab.py <detections.csv> <instances_val2017.json> <out.json>
"""
import csv, json, sys, collections
import numpy as np

det_p, coco_p, out_p = sys.argv[1:4]
rows = list(csv.DictReader(open(det_p, encoding="utf-8")))
coco = json.load(open(coco_p)); G = collections.defaultdict(list)
for a in coco["annotations"]:
    x, y, w, h = a["bbox"]; G[a["image_id"]].append((x, y, x + w, y + h))
G = {k: np.array(v, float) for k, v in G.items()}; E = np.zeros((0, 4))

def iou(b, g):
    if len(g) == 0: return np.zeros(0)
    ix = np.clip(np.minimum(b[2], g[:, 2]) - np.maximum(b[0], g[:, 0]), 0, None); iy = np.clip(np.minimum(b[3], g[:, 3]) - np.maximum(b[1], g[:, 1]), 0, None)
    I = ix * iy; U = (b[2]-b[0])*(b[3]-b[1]) + (g[:, 2]-g[:, 0])*(g[:, 3]-g[:, 1]) - I
    return np.where(U > 0, I / U, 0)

order = sorted(range(len(rows)), key=lambda k: -float(rows[k]["conf"]))
matched = {}; one_to_one = np.zeros(len(rows), bool)
for k in order:
    r = rows[k]; iid = int(r["image_id"]); g = G.get(iid, E)
    if len(g) == 0: continue
    m = matched.setdefault(iid, np.zeros(len(g), bool)); v = iou(np.array(json.loads(r["box"])), g); v[m] = 0; j = int(np.argmax(v))
    if v[j] >= 0.5: one_to_one[k] = True; m[j] = True

stats = collections.defaultdict(lambda: collections.Counter())
dup_labels = collections.Counter()
for k, r in enumerate(rows):
    c = r["vocab"]; hit = r["caap_tp"] == "1"; s = stats[c]
    s["n"] += 1; s["hit_detection_level"] += hit; s["tp_one_to_one"] += bool(one_to_one[k]); s["dup"] += int(hit and not one_to_one[k])
    if hit and not one_to_one[k]: dup_labels[r["label"].lower()] += 1

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - h) / d, (c + h) / d)

tot = collections.Counter()
for c, s in stats.items(): tot.update(s)
res = {"vocab_values_seen": sorted(stats), "total": dict(tot), "by_vocab": {}}
for c, s in stats.items():
    lo, hi = wilson(s["dup"], s["hit_detection_level"])
    res["by_vocab"][c] = dict(dict(s), detection_hit_rate=s["hit_detection_level"] / s["n"], one_to_one_rate=s["tp_one_to_one"] / s["n"],
                              dup_share_of_hits=s["dup"] / max(1, s["hit_detection_level"]), dup_share_of_hits_ci=[lo, hi],
                              share_of_all_detections=s["n"] / tot["n"], share_of_all_dups=s["dup"] / max(1, tot["dup"]))
nat, oov = res["by_vocab"].get("native"), res["by_vocab"].get("out_of_vocabulary")
if nat and oov:
    res["gap_pp_detection_level"] = 100 * (nat["detection_hit_rate"] - oov["detection_hit_rate"])
    res["gap_pp_one_to_one"] = 100 * (nat["one_to_one_rate"] - oov["one_to_one_rate"])
res["top_duplicate_labels"] = dup_labels.most_common(15)
json.dump(res, open(out_p, "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
