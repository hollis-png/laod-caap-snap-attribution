"""E14: recompute section 3.2 (RQ1/RQ2) numbers from detections.csv so that they have a stored source.

Detection-box size = box area / image area, bins <2%, 2-8%, 8-25%, >25%. CAAP-TP = detection-level rule of section 3.2 (IoU >= 0.5 with
any ground-truth box of the same image; column caap_tp). Prints Table 1 (size x vocabulary), the size marginals, the vocabulary
marginals and the aspect-ratio split (elongated = longer side / shorter side >= 2.5).
Usage: python e14_size_by_vocab.py <detections.csv> <instances_val2017.json> <out.json>
"""
import csv, json, sys, collections

det_p, coco_p, out_p = sys.argv[1:4]
imgs = {i["id"]: (i["width"], i["height"]) for i in json.load(open(coco_p))["images"]}
BINS = [("tiny (<2%)", 0, 0.02), ("small (2-8%)", 0.02, 0.08), ("medium (8-25%)", 0.08, 0.25), ("large (>25%)", 0.25, 9)]
cell = collections.defaultdict(lambda: [0, 0]); size = collections.defaultdict(lambda: [0, 0]); vocab = collections.defaultdict(lambda: [0, 0])
aspect = collections.defaultdict(lambda: [0, 0])
for r in csv.DictReader(open(det_p, encoding="utf-8")):
    w, h = imgs[int(r["image_id"])]; x1, y1, x2, y2 = json.loads(r["box"]); bw, bh = x2 - x1, y2 - y1
    frac = (bw * bh) / (w * h); tp = int(r["caap_tp"] == "1")
    b = next(n for n, lo, hi in BINS if lo <= frac < hi)
    for d, k in ((cell, f"{b} | {r['vocab']}"), (size, b), (vocab, r["vocab"]), (aspect, "elongated" if max(bw, bh) / max(1e-9, min(bw, bh)) >= 2.5 else "compact")):
        d[k][0] += 1; d[k][1] += tp
fmt = lambda d: {k: dict(n=v[0], tp=v[1], rate=round(v[1] / v[0], 4)) for k, v in d.items()}
res = dict(table1=fmt(cell), size=fmt(size), vocab=fmt(vocab), aspect=fmt(aspect))
json.dump(res, open(out_p, "w"), indent=1); print(json.dumps(res, indent=1))
