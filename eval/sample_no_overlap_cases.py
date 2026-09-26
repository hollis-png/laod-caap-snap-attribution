"""
Extract full list of no_overlap cases (SNAP-TP but CAAP-FP, best_iou < 0.02)
from the full 5000-image predictions, then randomly sample 60 for manual
visual inspection.

Reuses the exact same classification logic as eval/phase2_rq3.py.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.coco_loader import CocoGroundTruth
from eval.caap import calculate_iou
from eval.snap import embed_labels, cosine_similarity_matrix

CAAP_IOU_THRESHOLD = 0.5
SNAP_SIM_THRESHOLD = 0.7
NEAR_MISS_IOU_FLOOR = 0.1

ANNOTATIONS = r"./datasets/coco/annotations/instances_val2017.json"
IMAGES = r"./datasets/coco/val2017"
PREDICTIONS = "eval_out/coco_val_subset_predictions.jsonl"
OUT_SAMPLE = r"C:\LAOD\eval_out\no_overlap_sample.json"

SAMPLE_SIZE = 60


def load_predictions(predictions_path):
    preds = {}
    with open(predictions_path, "r") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                preds[rec["image_id"]] = rec
    return preds


def best_iou_match(det_box, gt_boxes):
    best_iou, best_idx = 0.0, -1
    for i, gt in enumerate(gt_boxes):
        iou = calculate_iou(det_box, gt[:4])
        if iou > best_iou:
            best_iou, best_idx = iou, i
    return best_iou, best_idx


def classify_failure(best_iou, matched_gt_label, det_label, matched_gt_sim):
    if best_iou < 0.02:
        return "no_overlap"
    if matched_gt_sim >= SNAP_SIM_THRESHOLD:
        return "near_miss_right_object"
    if best_iou >= NEAR_MISS_IOU_FLOOR:
        return "misplaced_onto_different_object"
    return "no_overlap"


def main():
    gt = CocoGroundTruth(ANNOTATIONS, IMAGES)
    preds_by_image = load_predictions(PREDICTIONS)

    no_overlap_cases = []

    for image_id, pred in preds_by_image.items():
        gt_boxes_caap = gt.gt_boxes_caap(image_id)
        gt_labels_snap = [g[0] for g in gt.gt_labels_snap(image_id)]
        detections = pred["detections"]
        if not detections or not gt_labels_snap:
            continue

        det_labels = [d["label"] for d in detections]
        det_embeddings = embed_labels(det_labels)
        gt_embeddings = embed_labels(gt_labels_snap)
        sim_matrix = cosine_similarity_matrix(det_embeddings, gt_embeddings)

        for i, det in enumerate(detections):
            best_sim = float(sim_matrix[i].max()) if sim_matrix.shape[1] > 0 else 0.0
            snap_tp = best_sim >= SNAP_SIM_THRESHOLD
            if not snap_tp:
                continue

            best_iou, gt_idx = best_iou_match(det["box"], gt_boxes_caap)
            caap_tp = best_iou >= CAAP_IOU_THRESHOLD
            if caap_tp:
                continue

            matched_gt_label = gt_boxes_caap[gt_idx][4] if gt_idx != -1 else None
            matched_gt_sim = 0.0
            if gt_idx != -1 and matched_gt_label in gt_labels_snap:
                matched_gt_sim = float(sim_matrix[i][gt_labels_snap.index(matched_gt_label)])

            failure_type = classify_failure(best_iou, matched_gt_label, det["label"], matched_gt_sim)
            if failure_type != "no_overlap":
                continue

            no_overlap_cases.append({
                "image_id": image_id,
                "det_label": det["label"],
                "det_confidence": det["confidence"],
                "det_box": det["box"],
                "best_iou": best_iou,
            })

    print(f"Total no_overlap cases found: {len(no_overlap_cases)}")
    assert len(no_overlap_cases) == 3825, (
        f"Expected 3825 no_overlap cases, got {len(no_overlap_cases)} -- BLOCKED, inconsistent with Step 1"
    )

    random.seed(0)
    sample = random.sample(no_overlap_cases, SAMPLE_SIZE)

    sample_out = [
        {
            "image_id": c["image_id"],
            "det_label": c["det_label"],
            "det_confidence": c["det_confidence"],
            "det_box": c["det_box"],
        }
        for c in sample
    ]

    os.makedirs(os.path.dirname(OUT_SAMPLE), exist_ok=True)
    with open(OUT_SAMPLE, "w") as f:
        json.dump(sample_out, f, indent=2)

    print(f"Sampled {len(sample_out)} cases written to {OUT_SAMPLE}")


if __name__ == "__main__":
    main()

