"""
Read-only analysis of the full 5000-image LAOD predictions, mirroring the
logic in phase2_stage2a.py and phase2_rq3.py exactly, but writing to a
separate output file so nothing existing gets overwritten.
"""
import json
import os
import sys

sys.path.insert(0, r"C:\LAOD")

from eval.coco_loader import CocoGroundTruth
from eval.caap import calculate_iou
from eval.snap import embed_labels, cosine_similarity_matrix

CAAP_IOU_THRESHOLD = 0.5
SNAP_SIM_THRESHOLD = 0.7
NEAR_MISS_IOU_FLOOR = 0.1

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
	"people": "person", "mice": "mouse", "knives": "knife",
	"loaves": "loaf", "shelves": "shelf", "leaves": "leaf",
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

def load_predictions(path):
	preds = {}
	with open(path, "r") as f:
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

def gt_box_area_ratio(gt_box, img_w, img_h):
	x1, y1, x2, y2 = gt_box[:4]
	area = max(0, x2 - x1) * max(0, y2 - y1)
	return area / (img_w * img_h) if img_w and img_h else 0.0

def gt_aspect_ratio(gt_box):
	x1, y1, x2, y2 = gt_box[:4]
	w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
	return max(w, h) / min(w, h)

def size_bucket(area_ratio):
	if area_ratio < 0.02:
		return "tiny(<2%)"
	elif area_ratio < 0.08:
		return "small(2-8%)"
	elif area_ratio < 0.25:
		return "medium(8-25%)"
	return "large(>25%)"

def classify_failure(best_iou, matched_gt_sim):
	if best_iou < 0.02:
		return "no_overlap"
	if matched_gt_sim >= SNAP_SIM_THRESHOLD:
		return "near_miss_right_object"
	if best_iou >= NEAR_MISS_IOU_FLOOR:
		return "misplaced_onto_different_object"
	return "no_overlap"

def pearson(x, y):
	mx, my = sum(x) / len(x), sum(y) / len(y)
	num = sum((a - mx) * (b - my) for a, b in zip(x, y))
	denx = sum((a - mx) ** 2 for a in x) ** 0.5
	deny = sum((b - my) ** 2 for b in y) ** 0.5
	return num / (denx * deny) if denx and deny else None

def main():
	annotations = r"./datasets/coco/annotations/instances_val2017.json"
	images = r"./datasets/coco/val2017"
	predictions = r"C:\LAOD\eval_out\coco_val_subset_predictions.jsonl"

	gt = CocoGroundTruth(annotations, images)
	preds_by_image = load_predictions(predictions)
	print(f"Loaded {len(preds_by_image)} images of predictions")

	det_records = []
	rq3_failures = []
	rq3_total_snap_tp_caap_fp = 0
	rq3_category_counts = {"no_overlap": 0, "near_miss_right_object": 0, "misplaced_onto_different_object": 0}

	n_processed = 0
	for image_id, pred in preds_by_image.items():
		gt_boxes_caap = gt.gt_boxes_caap(image_id)
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
				"caap_tp": caap_tp,
				"snap_tp": snap_tp,
				"confidence": det["confidence"],
				"best_iou": best_iou,
				"matched_gt_area_ratio": gt_box_area_ratio(matched_gt, img_w, img_h) if matched_gt else None,
				"matched_gt_aspect_ratio": gt_aspect_ratio(matched_gt) if matched_gt else None,
				"vocab_category": vocabulary_category(det["label"]),
			})

			# RQ3: SNAP-TP but CAAP-FP subset
			if snap_tp and not caap_tp and gt_labels_snap:
				rq3_total_snap_tp_caap_fp += 1
				matched_gt_label = matched_gt[4] if matched_gt else None
				matched_gt_sim = 0.0
				if gt_idx != -1 and matched_gt_label in gt_labels_snap:
					matched_gt_sim = float(sim_matrix[i][gt_labels_snap.index(matched_gt_label)])
				ftype = classify_failure(best_iou, matched_gt_sim)
				rq3_category_counts[ftype] += 1
				rq3_failures.append(ftype)

		n_processed += 1
		if n_processed % 500 == 0:
			print(f"  processed {n_processed}/{len(preds_by_image)} images...")

	print(f"\nTotal detections: {len(det_records)}")

	# RQ1: size bucket
	rq1_by_size = {}
	for r in det_records:
		if r["matched_gt_area_ratio"] is None:
			continue
		b = size_bucket(r["matched_gt_area_ratio"])
		rq1_by_size.setdefault(b, {"tp": 0, "total": 0})
		rq1_by_size[b]["total"] += 1
		rq1_by_size[b]["tp"] += int(r["caap_tp"])
	rq1_summary = {b: {"tp_rate": v["tp"] / v["total"], "n": v["total"]} for b, v in rq1_by_size.items()}

	elongated = [r for r in det_records if r["matched_gt_aspect_ratio"] and r["matched_gt_aspect_ratio"] >= 2.5]
	compact = [r for r in det_records if r["matched_gt_aspect_ratio"] and r["matched_gt_aspect_ratio"] < 2.5]
	rq1_aspect = {
		"elongated(aspect>=2.5)": {"tp_rate": sum(r["caap_tp"] for r in elongated) / len(elongated) if elongated else None, "n": len(elongated)},
		"compact(aspect<2.5)": {"tp_rate": sum(r["caap_tp"] for r in compact) / len(compact) if compact else None, "n": len(compact)},
	}

	# RQ2: vocab category (3-way + rollup)
	rq2_summary = {}
	for category in ("native", "plural_of_native", "out_of_vocabulary"):
		subset = [r for r in det_records if r["vocab_category"] == category]
		rq2_summary[category] = {"caap_tp_rate": sum(r["caap_tp"] for r in subset) / len(subset) if subset else None, "n": len(subset)}
	in_vocab = [r for r in det_records if r["vocab_category"] in ("native", "plural_of_native")]
	out_vocab = [r for r in det_records if r["vocab_category"] == "out_of_vocabulary"]
	rq2_summary["in_vocabulary_rollup"] = {"caap_tp_rate": sum(r["caap_tp"] for r in in_vocab) / len(in_vocab) if in_vocab else None, "n": len(in_vocab)}
	rq2_summary["out_of_vocabulary_rollup"] = {"caap_tp_rate": sum(r["caap_tp"] for r in out_vocab) / len(out_vocab) if out_vocab else None, "n": len(out_vocab)}

	# RQ4: confidence vs iou
	confs = [r["confidence"] for r in det_records]
	ious = [r["best_iou"] for r in det_records]
	rq4_summary = {
		"confidence_vs_iou_pearson_r": pearson(confs, ious),
		"n": len(det_records),
		"high_conf_low_iou_count": sum(1 for r in det_records if r["confidence"] >= 0.7 and r["best_iou"] < 0.3),
		"high_conf_count": sum(1 for r in det_records if r["confidence"] >= 0.7),
	}

	# RQ3 fractions
	rq3_fractions = {k: (v / rq3_total_snap_tp_caap_fp if rq3_total_snap_tp_caap_fp else None) for k, v in rq3_category_counts.items()}

	report = {
		"n_detections": len(det_records),
		"rq1_size_bucket": rq1_summary,
		"rq1_aspect_ratio": rq1_aspect,
		"rq2_vocabulary": rq2_summary,
		"rq4_confidence_calibration": rq4_summary,
		"rq3_total_snap_tp_caap_fp": rq3_total_snap_tp_caap_fp,
		"rq3_failure_type_counts": rq3_category_counts,
		"rq3_failure_type_fractions": rq3_fractions,
	}

	out_path = r"C:\LAOD\eval_out\phase2_5000_readonly_analysis.json"
	with open(out_path, "w") as f:
		json.dump(report, f, indent=2)

	print(json.dumps(report, indent=2))
	print(f"\nWritten to {out_path}")

if __name__ == "__main__":
	main()

