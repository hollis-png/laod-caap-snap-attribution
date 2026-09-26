"""
Evaluates gdino_predictions.jsonl the same way the validated YOLO-World run
was evaluated: corpus-level CAAP/SNAP, plus the RQ2 native-vs-out-of-vocabulary
CAAP-TP-rate comparison (vocabulary_category logic copied from
eval/vocab_deepdive_extract.py).

Usage:
	python eval/gdino_rq2_eval.py --annotations D:\\datasets\\coco\\annotations\\instances_val2017.json \
		--images D:\\datasets\\coco\\val2017 \
		--predictions eval_out/gdino_predictions.jsonl \
		--out eval_out/gdino_rq2_report.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.coco_loader import CocoGroundTruth
from eval.caap import compute_ca_ap, calculate_iou
from eval.snap import compute_snap

CAAP_IOU_THRESHOLD = 0.5

COCO80_NAMES = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
	"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
	"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
	"kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
	"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
	"couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
	"refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
	"toothbrush",
}

_IRREGULAR_SINGULARS = {
	"people": "person",
	"mice": "mouse",
	"knives": "knife",
	"loaves": "loaf",
	"shelves": "shelf",
	"leaves": "leaf",
}


def normalize_plural(word):
	w = word.lower().strip()
	if w in _IRREGULAR_SINGULARS:
		return _IRREGULAR_SINGULARS[w]
	if w.endswith("ies") and len(w) > 3:
		return w[:-3] + "y"
	if w.endswith(("ses", "xes", "zes", "ches", "shes")) and len(w) > 3:
		return w[:-2]
	if w.endswith("s") and not w.endswith("ss") and len(w) > 1:
		return w[:-1]
	return w


COCO80_NAMES_SINGULAR = {normalize_plural(n) for n in COCO80_NAMES}


def vocabulary_category(label):
	raw = label.lower().strip()
	if raw in COCO80_NAMES:
		return "native"
	if normalize_plural(raw) in COCO80_NAMES_SINGULAR:
		return "plural_of_native"
	return "out_of_vocabulary"


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
	"""
	For each detection, classify its label as native/plural_of_native/out_of_vocabulary,
	find its best-IoU-matching GT box (if any), and compute the CAAP-TP rate
	(fraction of GT-matched detections whose IoU >= 0.5) split by vocab category.
	Mirrors the logic in eval/vocab_deepdive_extract.py but aggregated by category
	rather than by individual label string.
	"""
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


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--annotations", required=True)
	parser.add_argument("--images", required=True)
	parser.add_argument("--predictions", default="eval_out/gdino_predictions.jsonl")
	parser.add_argument("--out", default="eval_out/gdino_rq2_report.json")
	args = parser.parse_args()

	gt = CocoGroundTruth(args.annotations, args.images)
	preds_by_image = load_predictions(args.predictions)

	image_ids = [iid for iid in gt.image_ids if iid in preds_by_image]
	print(f"Evaluating {len(image_ids)} images with Grounding DINO predictions")

	caap_preds, caap_gts = [], []
	snap_preds, snap_gts = [], []
	for image_id in image_ids:
		pred = preds_by_image[image_id]
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

	print("Computing native vs out-of-vocabulary CAAP-TP rate split...")
	vocab_split = compute_caap_tp_rate_by_vocab(gt, preds_by_image)
	for category, stats in vocab_split.items():
		print(f"  {category}: n={stats['count']} matched={stats['n_with_gt_match']} "
			  f"caap_tp_rate={stats['caap_tp_rate']}")

	report = {
		"n_images": len(image_ids),
		"corpus_caap": float(caap_score),
		"corpus_snap": float(snap_score),
		"vocab_caap_tp_split": vocab_split,
	}
	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	with open(args.out, "w") as f:
		json.dump(report, f, indent=2)
	print(f"\nReport written to {args.out}")


if __name__ == "__main__":
	main()

