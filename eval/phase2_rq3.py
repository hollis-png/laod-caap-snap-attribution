"""
Phase 2, RQ3: for detections that are SNAP-TP (semantically correct label) but
CAAP-FP (localization failed, IoU < 0.5 against the best-matching GT box),
characterize *how* localization failed:

  (a) "near-miss on the right object": the best-IoU-matching GT box's class
      is semantically close to the detection's label (i.e. SNAP would also
      call this box a match) but the box just isn't tight enough (some
      nonzero IoU, under the 0.5 threshold).
  (b) "misplaced onto a different, nearby object": the best-IoU-matching GT
      box's class is a different (or same) category but visually adjacent,
      with some meaningful IoU (>0.1) showing spatial overlap exists.
  (c) "no meaningful overlap with anything": best_iou is near zero,
      i.e. the detection box isn't spatially close to any GT box at all,
      suggesting the localization is essentially detached from real objects
      rather than merely imprecise.

Usage:
	python eval/phase2_rq3.py --annotations ./datasets/coco/annotations/instances_val2017.json \
		--images ./datasets/coco/val2017 --predictions eval_out/coco_val_subset_predictions.jsonl \
		--out eval_out/phase2_rq3_report.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.coco_loader import CocoGroundTruth
from eval.caap import calculate_iou
from eval.snap import embed_labels, cosine_similarity_matrix

CAAP_IOU_THRESHOLD = 0.5
SNAP_SIM_THRESHOLD = 0.7
NEAR_MISS_IOU_FLOOR = 0.1  # any spatial overlap worth calling "nearby"


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
		# The nearest box IS the semantically-correct object, just too loose.
		return "near_miss_right_object"
	if best_iou >= NEAR_MISS_IOU_FLOOR:
		return "misplaced_onto_different_object"
	return "no_overlap"


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--annotations", required=True)
	parser.add_argument("--images", required=True)
	parser.add_argument("--predictions", default="eval_out/coco_val_subset_predictions.jsonl")
	parser.add_argument("--out", default="eval_out/phase2_rq3_report.json")
	args = parser.parse_args()

	gt = CocoGroundTruth(args.annotations, args.images)
	preds_by_image = load_predictions(args.predictions)

	failure_cases = []
	category_counts = {"no_overlap": 0, "near_miss_right_object": 0, "misplaced_onto_different_object": 0}
	total_snap_tp_caap_fp = 0

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
				continue  # only care about SNAP-TP cases here

			best_iou, gt_idx = best_iou_match(det["box"], gt_boxes_caap)
			caap_tp = best_iou >= CAAP_IOU_THRESHOLD
			if caap_tp:
				continue  # only care about the SNAP-TP & CAAP-FP subset

			total_snap_tp_caap_fp += 1

			matched_gt_label = gt_boxes_caap[gt_idx][4] if gt_idx != -1 else None
			# similarity between the detection's label and the *nearest* GT box's
			# class (which may differ from the GT box that actually satisfied SNAP).
			matched_gt_sim = 0.0
			if gt_idx != -1 and matched_gt_label in gt_labels_snap:
				matched_gt_sim = float(sim_matrix[i][gt_labels_snap.index(matched_gt_label)])

			failure_type = classify_failure(best_iou, matched_gt_label, det["label"], matched_gt_sim)
			category_counts[failure_type] += 1

			failure_cases.append({
				"image_id": image_id,
				"det_label": det["label"],
				"det_confidence": det["confidence"],
				"best_iou": best_iou,
				"matched_gt_label": matched_gt_label,
				"failure_type": failure_type,
			})

	report = {
		"total_snap_tp_caap_fp": total_snap_tp_caap_fp,
		"failure_type_counts": category_counts,
		"failure_type_fractions": {
			k: (v / total_snap_tp_caap_fp if total_snap_tp_caap_fp else None)
			for k, v in category_counts.items()
		},
		"sample_cases_per_type": {
			ftype: [c for c in failure_cases if c["failure_type"] == ftype][:5]
			for ftype in category_counts
		},
	}

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	with open(args.out, "w") as f:
		json.dump(report, f, indent=2)

	print(json.dumps(
		{"total_snap_tp_caap_fp": total_snap_tp_caap_fp,
		 "failure_type_counts": category_counts,
		 "failure_type_fractions": report["failure_type_fractions"]},
		indent=2,
	))
	print(f"\nWritten to {args.out}")


if __name__ == "__main__":
	main()

