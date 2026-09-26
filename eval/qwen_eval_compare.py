"""
Step 3: evaluate qwen_yoloworld_predictions.jsonl the same way the validated
Gemma+YOLO-World run was evaluated (corpus CAAP/SNAP + RQ2 native-vs-OOV
CAAP-TP split, using phase2_stage2a.py's vocabulary_category/normalize_plural
logic verbatim), then compare vocabulary distributions and a few example
labels against the original Gemma-based coco_val_subset_predictions.jsonl.

Usage:
    python eval/qwen_eval_compare.py --annotations D:\\datasets\\coco\\annotations\\instances_val2017.json \
        --images D:\\datasets\\coco\\val2017 \
        --qwen-predictions eval_out/qwen_yoloworld_predictions.jsonl \
        --gemma-predictions eval_out/coco_val_subset_predictions.jsonl \
        --out eval_out/qwen_rq2_report.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.coco_loader import CocoGroundTruth
from eval.caap import compute_ca_ap, calculate_iou
from eval.snap import compute_snap
from eval.phase2_stage2a import vocabulary_category, normalize_plural, COCO80_NAMES

CAAP_IOU_THRESHOLD = 0.5


def best_iou_match(det_box, gt_boxes):
    best_iou, best_idx = 0.0, -1
    for i, gt in enumerate(gt_boxes):
        iou = calculate_iou(det_box, gt[:4])
        if iou > best_iou:
            best_iou, best_idx = iou, i
    return best_iou, best_idx


def load_predictions(path):
    preds = {}
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                preds[rec["image_id"]] = rec
    return preds


def compute_caap_tp_rate_by_vocab(gt, preds_by_image):
    tally = defaultdict(lambda: {"n_with_gt_match": 0, "tp_with_gt_match": 0, "count": 0})
    for image_id, pred in preds_by_image.items():
        gt_boxes_caap = gt.gt_boxes_caap(image_id)
        for det in pred.get("detections", []):
            label = det["label"]
            category = vocabulary_category(label)
            tally[category]["count"] += 1
            best_iou, gt_idx = best_iou_match(det["box"], gt_boxes_caap)
            if gt_idx != -1:
                tally[category]["n_with_gt_match"] += 1
                if best_iou >= CAAP_IOU_THRESHOLD:
                    tally[category]["tp_with_gt_match"] += 1
    result = {}
    for category, entry in tally.items():
        rate = (entry["tp_with_gt_match"] / entry["n_with_gt_match"]) if entry["n_with_gt_match"] > 0 else None
        result[category] = {
            "count": entry["count"],
            "n_with_gt_match": entry["n_with_gt_match"],
            "tp_with_gt_match": entry["tp_with_gt_match"],
            "caap_tp_rate": rate,
        }
    return result


def vocab_label_distribution(preds_by_image):
    """Fraction of LLM-generated labels (not detections) that are native / plural_of_native / OOV."""
    counts = Counter()
    total = 0
    for pred in preds_by_image.values():
        for label in pred.get("llm_labels", []):
            counts[vocabulary_category(label)] += 1
            total += 1
    return {k: {"count": v, "fraction": v / total if total else None} for k, v in counts.items()}, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--qwen-predictions", default="eval_out/qwen_yoloworld_predictions.jsonl")
    parser.add_argument("--gemma-predictions", default="eval_out/coco_val_subset_predictions.jsonl")
    parser.add_argument("--out", default="eval_out/qwen_rq2_report.json")
    args = parser.parse_args()

    gt = CocoGroundTruth(args.annotations, args.images)
    qwen_preds = load_predictions(args.qwen_predictions)
    gemma_preds = load_predictions(args.gemma_predictions) if os.path.exists(args.gemma_predictions) else {}

    image_ids = [iid for iid in gt.image_ids if iid in qwen_preds]
    print(f"Evaluating {len(image_ids)} images with Qwen2.5-VL-generated labels + YOLO-World detections")

    caap_preds, caap_gts = [], []
    snap_preds, snap_gts = [], []
    for image_id in image_ids:
        pred = qwen_preds[image_id]
        caap_preds.append([
            [d["box"][0], d["box"][1], d["box"][2], d["box"][3], d["confidence"]]
            for d in pred["detections"]
        ])
        caap_gts.append(gt.gt_boxes_caap(image_id))
        snap_preds.append([[d["label"], d["confidence"]] for d in pred["detections"]])
        snap_gts.append(gt.gt_labels_snap(image_id))

    print("Computing corpus-level CAAP...")
    caap_score = compute_ca_ap(caap_preds, caap_gts, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    print(f"CAAP = {caap_score:.4f}")

    print("Computing corpus-level SNAP (loads CLIP)...")
    snap_score = compute_snap(snap_preds, snap_gts, [0.6, 0.7, 0.8, 0.9])
    print(f"SNAP = {snap_score:.4f}")

    print("Computing native vs out-of-vocabulary CAAP-TP rate split (Qwen)...")
    qwen_vocab_split = compute_caap_tp_rate_by_vocab(gt, qwen_preds)
    for category, stats in qwen_vocab_split.items():
        print(f"  {category}: n={stats['count']} matched={stats['n_with_gt_match']} caap_tp_rate={stats['caap_tp_rate']}")

    qwen_label_dist, qwen_total_labels = vocab_label_distribution(qwen_preds)
    print(f"\nQwen LLM label vocabulary distribution (n={qwen_total_labels}):")
    for k, v in qwen_label_dist.items():
        print(f"  {k}: {v}")

    gemma_vocab_split = None
    gemma_label_dist = None
    gemma_total_labels = None
    spot_check = []
    if gemma_preds:
        print("\nComputing native vs out-of-vocabulary CAAP-TP rate split (Gemma, for comparison)...")
        gemma_vocab_split = compute_caap_tp_rate_by_vocab(gt, gemma_preds)
        for category, stats in gemma_vocab_split.items():
            print(f"  {category}: n={stats['count']} matched={stats['n_with_gt_match']} caap_tp_rate={stats['caap_tp_rate']}")

        gemma_label_dist, gemma_total_labels = vocab_label_distribution(gemma_preds)
        print(f"\nGemma LLM label vocabulary distribution (n={gemma_total_labels}):")
        for k, v in gemma_label_dist.items():
            print(f"  {k}: {v}")

        # Qualitative spot-check: same image_ids, side-by-side labels
        common_ids = [iid for iid in image_ids if iid in gemma_preds][:15]
        for iid in common_ids:
            spot_check.append({
                "image_id": iid,
                "qwen_llm_labels": qwen_preds[iid].get("llm_labels", []),
                "gemma_llm_labels": gemma_preds[iid].get("llm_labels", []),
            })
        print("\nSpot-check (first 15 common image_ids):")
        for row in spot_check:
            print(f"  id={row['image_id']}: qwen={row['qwen_llm_labels']} | gemma={row['gemma_llm_labels']}")

    report = {
        "n_images": len(image_ids),
        "corpus_caap": float(caap_score),
        "corpus_snap": float(snap_score),
        "qwen_vocab_caap_tp_split": qwen_vocab_split,
        "qwen_label_distribution": qwen_label_dist,
        "qwen_total_labels": qwen_total_labels,
        "gemma_vocab_caap_tp_split": gemma_vocab_split,
        "gemma_label_distribution": gemma_label_dist,
        "gemma_total_labels": gemma_total_labels,
        "spot_check": spot_check,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()

