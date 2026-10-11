"""
Vocab deep-dive extraction (FULL): same as eval/vocab_deepdive_extract.py but
outputs ALL unique out-of-vocabulary labels (no top-N truncation), sorted by
count descending. Logic copied verbatim from eval/vocab_deepdive_extract.py.

Usage:
	python eval/vocab_deepdive_extract_full.py --annotations D:\\datasets\\coco\\annotations\\instances_val2017.json \
		--images D:\\datasets\\coco\\val2017 --predictions eval_out/coco_val_subset_predictions.jsonl \
		--out eval_out/vocab_deepdive_raw_full.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.coco_loader import CocoGroundTruth
from eval.caap import calculate_iou

CAAP_IOU_THRESHOLD = 0.5

# --- copied verbatim from eval/phase2_stage2a.py (via vocab_deepdive_extract.py) ---

from eval.vocab import COCO80_NAMES, COCO80_NAMES_SINGULAR, _IRREGULAR_SINGULARS, normalize_plural, vocabulary_category  # noqa: F401  (single shared definition, see eval/vocab.py)


def best_iou_match(det_box, gt_boxes):
	best_iou, best_idx = 0.0, -1
	for i, gt in enumerate(gt_boxes):
		iou = calculate_iou(det_box, gt[:4])
		if iou > best_iou:
			best_iou, best_idx = iou, i
	return best_iou, best_idx

# --- end copied section ---


def load_predictions(predictions_path):
	preds = {}
	with open(predictions_path, "r") as f:
		for line in f:
			if line.strip():
				rec = json.loads(line)
				preds[rec["image_id"]] = rec
	return preds


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--annotations", required=True)
	parser.add_argument("--images", required=True)
	parser.add_argument("--predictions", default="eval_out/coco_val_subset_predictions.jsonl")
	parser.add_argument("--out", default="eval_out/vocab_deepdive_raw_full.json")
	args = parser.parse_args()

	gt = CocoGroundTruth(args.annotations, args.images)
	preds_by_image = load_predictions(args.predictions)

	# label(lowercased, stripped) -> counts / matched-gt-class tally / caap_tp tally
	by_label = defaultdict(lambda: {
		"count": 0,
		"matched_gt_counter": Counter(),
		"n_with_gt_match": 0,
		"tp_with_gt_match": 0,
	})

	n_images = 0
	n_total_detections = 0
	n_oov_detections = 0

	for image_id, pred in preds_by_image.items():
		n_images += 1
		gt_boxes_caap = gt.gt_boxes_caap(image_id)
		detections = pred.get("detections", [])
		n_total_detections += len(detections)

		for det in detections:
			label = det["label"]
			if vocabulary_category(label) != "out_of_vocabulary":
				continue
			n_oov_detections += 1

			key = label.lower().strip()
			entry = by_label[key]
			entry["count"] += 1

			best_iou, gt_idx = best_iou_match(det["box"], gt_boxes_caap)
			if gt_idx != -1:
				matched_class = gt_boxes_caap[gt_idx][4]
				entry["matched_gt_counter"][matched_class] += 1
				entry["n_with_gt_match"] += 1
				if best_iou >= CAAP_IOU_THRESHOLD:
					entry["tp_with_gt_match"] += 1

	results = []
	for label, entry in by_label.items():
		if entry["n_with_gt_match"] > 0:
			most_common_matched_gt, _ = entry["matched_gt_counter"].most_common(1)[0]
			caap_tp_rate = entry["tp_with_gt_match"] / entry["n_with_gt_match"]
		else:
			most_common_matched_gt = None
			caap_tp_rate = None
		results.append({
			"label": label,
			"count": entry["count"],
			"most_common_matched_gt": most_common_matched_gt,
			"caap_tp_rate": caap_tp_rate,
			"n_with_gt_match": entry["n_with_gt_match"],
		})

	results.sort(key=lambda r: r["count"], reverse=True)

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	with open(args.out, "w") as f:
		json.dump(results, f, indent=2)

	print(f"n_images={n_images} n_total_detections={n_total_detections} n_oov_detections={n_oov_detections}")
	print(f"n_unique_oov_labels={len(results)}  writing all {len(results)} to {args.out}")
	print("\nTop 20 entries:")
	for r in results[:20]:
		print(json.dumps(r))


if __name__ == "__main__":
	main()

