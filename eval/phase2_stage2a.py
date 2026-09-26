"""
Phase 2, stage 2a: fast screening of RQ1/RQ2/RQ4 hypotheses on the existing
200-image COCO-Val predictions from Phase 1, without re-running inference.

For each detection, we tag whether it's a CAAP true positive (IoU >= 0.5
against its best-matching, class-agnostic GT box) and a SNAP true positive
(CLIP cosine similarity >= 0.7 against its best-matching GT label), then
cross-analyze against candidate explanatory factors:

  RQ1: GT box size (relative area) and aspect ratio vs CAAP-TP rate
  RQ2: whether the LLM label is a native COCO-80 category name (proxy for
       "common/in-vocabulary" vs "novel phrasing") vs CAAP-TP rate
  RQ4: detection confidence vs the IoU actually achieved against its best GT match

Usage:
	python eval/phase2_stage2a.py --annotations ./datasets/coco/annotations/instances_val2017.json \
		--images ./datasets/coco/val2017 --predictions eval_out/coco_val_subset_predictions.jsonl \
		--out eval_out/phase2_stage2a_report.json
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

# COCO-80 category names, used as a crude proxy for "common/in-vocabulary"
# phrasing vs whatever novel wording the LLM might generate instead.
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

# Irregular plural/singular pairs among COCO-80 names and common LLM phrasing
# for them; anything not listed here falls back to simple suffix stripping.
_IRREGULAR_SINGULARS = {
	"people": "person",
	"mice": "mouse",
	"knives": "knife",
	"loaves": "loaf",
	"shelves": "shelf",
	"leaves": "leaf",
}


def normalize_plural(word):
	"""
	Crude, dependency-free singularization so that "chairs"/"cows"/"bottles"
	aren't miscounted as novel/out-of-vocabulary wording relative to COCO-80's
	singular "chair"/"cow"/"bottle" -- this is a pure grammatical-form
	difference, not evidence of semantic drift, and conflating the two
	confounds the RQ2 vocabulary-novelty analysis.
	"""
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
	"""
	Classifies a detection label relative to COCO-80 into three buckets:
	  - "native": exact match to a COCO-80 name (singular or plural form)
	  - "plural_of_native": normalizes to a COCO-80 name but isn't an exact
	    string match (pure grammatical-form difference, e.g. "chairs")
	  - "out_of_vocabulary": doesn't match COCO-80 at all, even after
	    normalization (e.g. "plate", "cabinet") -- these are genuinely
	    outside the COCO-80 label space, not just differently-worded synonyms
	"""
	raw = label.lower().strip()
	if raw in COCO80_NAMES:
		return "native"
	if normalize_plural(raw) in COCO80_NAMES_SINGULAR:
		return "plural_of_native"
	return "out_of_vocabulary"


def load_predictions(predictions_path):
	preds = {}
	with open(predictions_path, "r") as f:
		for line in f:
			if line.strip():
				rec = json.loads(line)
				preds[rec["image_id"]] = rec
	return preds


def best_iou_match(det_box, gt_boxes):
	"""Class-agnostic best IoU match, mirroring caap.py's matching logic."""
	best_iou, best_idx = 0.0, -1
	for i, gt in enumerate(gt_boxes):
		iou = calculate_iou(det_box, gt[:4])
		if iou > best_iou:
			best_iou, best_idx = iou, i
	return best_iou, best_idx


def gt_box_area_ratio(gt_box, img_w, img_h):
	x1, y1, x2, y2 = gt_box[:4]
	area = max(0, x2 - x1) * max(0, y2 - y1)
	return area / (img_w * img_h) if img_w and img_h else 0.0


def gt_aspect_ratio(gt_box):
	x1, y1, x2, y2 = gt_box[:4]
	w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
	return max(w, h) / min(w, h)  # always >= 1, elongation regardless of orientation


def size_bucket(area_ratio):
	if area_ratio < 0.02:
		return "tiny(<2%)"
	elif area_ratio < 0.08:
		return "small(2-8%)"
	elif area_ratio < 0.25:
		return "medium(8-25%)"
	return "large(>25%)"


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--annotations", required=True)
	parser.add_argument("--images", required=True)
	parser.add_argument("--predictions", default="eval_out/coco_val_subset_predictions.jsonl")
	parser.add_argument("--out", default="eval_out/phase2_stage2a_report.json")
	args = parser.parse_args()

	gt = CocoGroundTruth(args.annotations, args.images)
	preds_by_image = load_predictions(args.predictions)

	det_records = []  # one row per detection, across all images

	for image_id, pred in preds_by_image.items():
		gt_boxes_caap = gt.gt_boxes_caap(image_id)  # [x1,y1,x2,y2,class_name]
		gt_labels_snap = [g[0] for g in gt.gt_labels_snap(image_id)]
		img_meta = gt.image_meta_by_id[image_id]
		img_w, img_h = img_meta["width"], img_meta["height"]

		detections = pred["detections"]
		if not detections:
			continue

		det_labels = [d["label"] for d in detections]
		det_embeddings = embed_labels(det_labels)
		gt_embeddings = embed_labels(gt_labels_snap) if gt_labels_snap else None
		sim_matrix = (
			cosine_similarity_matrix(det_embeddings, gt_embeddings)
			if gt_labels_snap else None
		)

		for i, det in enumerate(detections):
			best_iou, gt_idx = best_iou_match(det["box"], gt_boxes_caap)
			caap_tp = best_iou >= CAAP_IOU_THRESHOLD

			if sim_matrix is not None and sim_matrix.shape[1] > 0:
				best_sim = float(sim_matrix[i].max())
			else:
				best_sim = 0.0
			snap_tp = best_sim >= SNAP_SIM_THRESHOLD

			matched_gt = gt_boxes_caap[gt_idx] if gt_idx != -1 else None

			det_records.append({
				"image_id": image_id,
				"label": det["label"],
				"confidence": det["confidence"],
				"best_iou": best_iou,
				"best_sim": best_sim,
				"caap_tp": caap_tp,
				"snap_tp": snap_tp,
				# RQ1: size/shape of the GT box this detection best overlaps with
				# (only meaningful if there was any overlap at all).
				"matched_gt_area_ratio": gt_box_area_ratio(matched_gt, img_w, img_h) if matched_gt else None,
				"matched_gt_aspect_ratio": gt_aspect_ratio(matched_gt) if matched_gt else None,
				# RQ2: classify the LLM's wording relative to COCO-80 into
				# native / plural_of_native (grammar-only difference) /
				# out_of_vocabulary (genuinely not a COCO-80 category).
				"vocab_category": vocabulary_category(det["label"]),
			})

	# --- RQ1: size/aspect ratio vs CAAP-TP rate ---
	rq1_by_size = {}
	for r in det_records:
		if r["matched_gt_area_ratio"] is None:
			continue
		bucket = size_bucket(r["matched_gt_area_ratio"])
		rq1_by_size.setdefault(bucket, {"tp": 0, "total": 0})
		rq1_by_size[bucket]["total"] += 1
		rq1_by_size[bucket]["tp"] += int(r["caap_tp"])
	rq1_summary = {
		b: {"tp_rate": v["tp"] / v["total"], "n": v["total"]}
		for b, v in rq1_by_size.items()
	}

	elongated = [r for r in det_records if r["matched_gt_aspect_ratio"] and r["matched_gt_aspect_ratio"] >= 2.5]
	compact = [r for r in det_records if r["matched_gt_aspect_ratio"] and r["matched_gt_aspect_ratio"] < 2.5]
	rq1_aspect_summary = {
		"elongated(aspect>=2.5)": {
			"tp_rate": sum(r["caap_tp"] for r in elongated) / len(elongated) if elongated else None,
			"n": len(elongated),
		},
		"compact(aspect<2.5)": {
			"tp_rate": sum(r["caap_tp"] for r in compact) / len(compact) if compact else None,
			"n": len(compact),
		},
	}

	# --- RQ2: COCO-80 vocabulary category vs CAAP-TP rate (three-way split,
	# separating pure grammatical plural/singular mismatches from genuine
	# out-of-vocabulary wording, since conflating them confounded the
	# original two-way native/novel split) ---
	rq2_summary = {}
	for category in ("native", "plural_of_native", "out_of_vocabulary"):
		subset = [r for r in det_records if r["vocab_category"] == category]
		rq2_summary[category] = {
			"caap_tp_rate": sum(r["caap_tp"] for r in subset) / len(subset) if subset else None,
			"n": len(subset),
		}
	# Convenience rollup: native + plural_of_native = "in COCO-80 vocabulary
	# regardless of grammatical form" vs out_of_vocabulary, for comparison
	# against the original (pre-fix) two-way split.
	in_vocab = [r for r in det_records if r["vocab_category"] in ("native", "plural_of_native")]
	out_vocab = [r for r in det_records if r["vocab_category"] == "out_of_vocabulary"]
	rq2_summary["in_vocabulary_rollup"] = {
		"caap_tp_rate": sum(r["caap_tp"] for r in in_vocab) / len(in_vocab) if in_vocab else None,
		"n": len(in_vocab),
	}
	rq2_summary["out_of_vocabulary_rollup"] = {
		"caap_tp_rate": sum(r["caap_tp"] for r in out_vocab) / len(out_vocab) if out_vocab else None,
		"n": len(out_vocab),
	}

	# --- RQ4: confidence vs achieved IoU (calibration check) ---
	confs = [r["confidence"] for r in det_records]
	ious = [r["best_iou"] for r in det_records]

	def pearson(x, y):
		mx, my = sum(x) / len(x), sum(y) / len(y)
		num = sum((a - mx) * (b - my) for a, b in zip(x, y))
		denx = sum((a - mx) ** 2 for a in x) ** 0.5
		deny = sum((b - my) ** 2 for b in y) ** 0.5
		return num / (denx * deny) if denx and deny else None

	rq4_summary = {
		"confidence_vs_iou_pearson_r": pearson(confs, ious),
		"n": len(det_records),
		"high_conf_low_iou_count": sum(1 for r in det_records if r["confidence"] >= 0.7 and r["best_iou"] < 0.3),
		"high_conf_count": sum(1 for r in det_records if r["confidence"] >= 0.7),
	}

	report = {
		"n_detections": len(det_records),
		"caap_iou_threshold": CAAP_IOU_THRESHOLD,
		"snap_sim_threshold": SNAP_SIM_THRESHOLD,
		"rq1_size_bucket": rq1_summary,
		"rq1_aspect_ratio": rq1_aspect_summary,
		"rq2_vocabulary": rq2_summary,
		"rq4_confidence_calibration": rq4_summary,
	}

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	with open(args.out, "w") as f:
		json.dump(report, f, indent=2)

	print(json.dumps(report, indent=2))
	print(f"\nWritten to {args.out}")


if __name__ == "__main__":
	main()

