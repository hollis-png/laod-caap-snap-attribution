"""
Independent, from-scratch verification of gdino_rq2_report.json.
Deliberately re-implements (does not import) vocabulary_category/normalize_plural/
COCO80_NAMES and the IoU/best-match logic, typed out fresh by hand from reading
eval/phase2_stage2a.py, to catch any bug that a copy-paste re-run of the same
script would not catch. Also independently recomputes corpus CAAP using
eval.caap.compute_ca_ap (imported, since re-deriving mAP-style logic from scratch
is out of scope and caap.py is a small, previously-reviewed utility) and reports
total detection counts / vocab category counts / per-category CAAP-TP rate.

Run in the laod conda env on the Windows box (needs eval.coco_loader, eval.caap).
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, r"C:\LAOD")

from eval.coco_loader import CocoGroundTruth
from eval.caap import compute_ca_ap

ANNOTATIONS = r"./datasets/coco/annotations/instances_val2017.json"
IMAGES = r"./datasets/coco/val2017"
PREDICTIONS = r"C:\LAOD\eval_out\gdino_predictions.jsonl"

# --- independently retyped COCO-80 name set (not imported) ---
COCO80 = """person bicycle car motorcycle airplane bus train truck boat
traffic_light fire_hydrant stop_sign parking_meter bench bird cat
dog horse sheep cow elephant bear zebra giraffe backpack
umbrella handbag tie suitcase frisbee skis snowboard sports_ball
kite baseball_bat baseball_glove skateboard surfboard tennis_racket
bottle wine_glass cup fork knife spoon bowl banana apple
sandwich orange broccoli carrot hot_dog pizza donut cake chair
couch potted_plant bed dining_table toilet tv laptop mouse
remote keyboard cell_phone microwave oven toaster sink
refrigerator book clock vase scissors teddy_bear hair_drier
toothbrush""".split()
COCO80_NAMES = {w.replace("_", " ") for w in COCO80}
assert len(COCO80_NAMES) == 80, f"expected 80 names, got {len(COCO80_NAMES)}"


def my_singularize(w):
    """Independent re-derivation of a simple pluralization stripper."""
    w = w.lower().strip()
    irregular = {"people": "person", "mice": "mouse", "knives": "knife",
                 "loaves": "loaf", "shelves": "shelf", "leaves": "leaf"}
    if w in irregular:
        return irregular[w]
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and (w.endswith("ses") or w.endswith("xes") or w.endswith("zes")
                        or w.endswith("ches") or w.endswith("shes")):
        return w[:-2]
    if len(w) > 1 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


COCO80_SINGULAR = {my_singularize(n) for n in COCO80_NAMES}


def classify(label):
    l = label.lower().strip()
    if l in COCO80_NAMES:
        return "native"
    if my_singularize(l) in COCO80_SINGULAR:
        return "plural_of_native"
    return "out_of_vocabulary"


def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, xb - xa), max(0.0, yb - ya)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def best_match(box, gt_boxes):
    best_i, best_idx = 0.0, -1
    for idx, gt in enumerate(gt_boxes):
        v = iou(box, gt[:4])
        if v > best_i:
            best_i, best_idx = v, idx
    return best_i, best_idx


def main():
    gt = CocoGroundTruth(ANNOTATIONS, IMAGES)

    preds = {}
    with open(PREDICTIONS, "r") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                preds[r["image_id"]] = r

    image_ids = [i for i in gt.image_ids if i in preds]
    print("n_images_matched:", len(image_ids), "n_images_in_file:", len(preds))

    tally = defaultdict(lambda: {"count": 0, "matched": 0, "tp": 0})
    total_det = 0
    label_examples = defaultdict(list)

    caap_preds_list = []
    caap_gts_list = []

    for iid in image_ids:
        pred = preds[iid]
        gtboxes = gt.gt_boxes_caap(iid)
        caap_gts_list.append(gtboxes)
        row_preds = []
        for det in pred.get("detections", []):
            total_det += 1
            cat = classify(det["label"])
            tally[cat]["count"] += 1
            if len(label_examples[cat]) < 5:
                label_examples[cat].append(det["label"])
            bi, gi = best_match(det["box"], gtboxes)
            if gi != -1:
                tally[cat]["matched"] += 1
                if bi >= 0.5:
                    tally[cat]["tp"] += 1
            row_preds.append([det["box"][0], det["box"][1], det["box"][2], det["box"][3], det["confidence"]])
        caap_preds_list.append(row_preds)

    print("\ntotal_detections (independent count):", total_det)
    print("\nvocab category breakdown (independent):")
    for cat, v in tally.items():
        rate = v["tp"] / v["matched"] if v["matched"] else None
        print(f"  {cat}: count={v['count']} matched={v['matched']} tp={v['tp']} caap_tp_rate={rate}")
        print(f"    example labels: {label_examples[cat]}")

    print("\nComputing corpus CAAP (using eval.caap.compute_ca_ap, imported utility)...")
    caap_score = compute_ca_ap(caap_preds_list, caap_gts_list,
                                [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    print("corpus_caap (independent):", float(caap_score))

    out = {
        "n_images": len(image_ids),
        "total_detections": total_det,
        "corpus_caap_independent": float(caap_score),
        "vocab_breakdown": {
            cat: {
                "count": v["count"], "matched": v["matched"], "tp": v["tp"],
                "caap_tp_rate": (v["tp"] / v["matched"] if v["matched"] else None),
            } for cat, v in tally.items()
        },
    }
    with open(r"C:\LAOD\eval_out\gdino_independent_verify.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWritten to eval_out\\gdino_independent_verify.json")


if __name__ == "__main__":
    main()

