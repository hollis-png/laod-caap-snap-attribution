"""
Phase 1 batch evaluation driver: runs the LAOD pipeline over a COCO-Val subset,
caches raw predictions to JSON (so CAAP/SNAP can be recomputed without
re-running inference), and reports both metrics plus their overall correlation
(the go/no-go check from experiment_plan.md Phase 1, checkpoint B).

Usage:
	python eval/run_batch_eval.py --annotations ./datasets/coco/annotations/instances_val2017.json \
		--images ./datasets/coco/val2017 --n 200 --predictions-out eval_out/coco_val_subset_predictions.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from eval.coco_loader import CocoGroundTruth
from eval.laod_infer import laod_yolo_structured
from eval.caap import compute_ca_ap
from eval.snap import compute_snap


def run_inference(gt, predictions_out, limit=None, resume=True):
	"""
	Runs laod_yolo_structured over every image in gt, caching results to
	predictions_out as newline-delimited JSON so a crashed/interrupted run
	can resume without re-inferring already-processed images.
	"""
	os.makedirs(os.path.dirname(predictions_out) or ".", exist_ok=True)

	done_ids = set()
	if resume and os.path.exists(predictions_out):
		with open(predictions_out, "r") as f:
			for line in f:
				if line.strip():
					done_ids.add(json.loads(line)["image_id"])
		print(f"Resuming: {len(done_ids)} images already processed, skipping them.")

	image_ids = gt.image_ids if limit is None else gt.image_ids[:limit]

	with open(predictions_out, "a") as f:
		for i, image_id in enumerate(image_ids):
			if image_id in done_ids:
				continue

			t0 = time.time()
			image = Image.open(gt.image_path(image_id)).convert("RGB")
			result = laod_yolo_structured(image)
			elapsed = time.time() - t0

			record = {
				"image_id": image_id,
				"llm_labels": result["llm_labels"],
				"detections": result["detections"],
				"elapsed_sec": elapsed,
			}
			f.write(json.dumps(record) + "\n")
			f.flush()

			print(f"[{i+1}/{len(image_ids)}] image_id={image_id} "
				  f"labels={result['llm_labels']} "
				  f"n_det={len(result['detections'])} "
				  f"({elapsed:.1f}s)")


def load_predictions(predictions_out):
	preds_by_image_id = {}
	with open(predictions_out, "r") as f:
		for line in f:
			if line.strip():
				rec = json.loads(line)
				preds_by_image_id[rec["image_id"]] = rec
	return preds_by_image_id


def build_eval_inputs(gt, preds_by_image_id):
	"""
	Converts cached predictions + COCO GT into the parallel-list formats that
	compute_ca_ap and compute_snap expect (same image order for both).
	"""
	image_ids = [iid for iid in gt.image_ids if iid in preds_by_image_id]

	caap_preds, caap_gts = [], []
	snap_preds, snap_gts = [], []

	for image_id in image_ids:
		pred = preds_by_image_id[image_id]

		caap_preds.append([
			[d["box"][0], d["box"][1], d["box"][2], d["box"][3], d["confidence"]]
			for d in pred["detections"]
		])
		caap_gts.append(gt.gt_boxes_caap(image_id))

		snap_preds.append([[d["label"], d["confidence"]] for d in pred["detections"]])
		snap_gts.append(gt.gt_labels_snap(image_id))

	return image_ids, caap_preds, caap_gts, snap_preds, snap_gts


def per_image_caap_snap(image_ids, caap_preds, caap_gts, snap_preds, snap_gts,
						 iou_thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
						 sim_thresholds=(0.6, 0.7, 0.8, 0.9)):
	"""
	Computes CAAP and SNAP independently for each image (single-image "dataset"),
	for the checkpoint-B correlation analysis in experiment_plan.md. This is
	distinct from the corpus-level compute_ca_ap/compute_snap calls, which pool
	predictions across all images for a single dataset-level score.
	"""
	rows = []
	for i, image_id in enumerate(image_ids):
		caap_i = compute_ca_ap([caap_preds[i]], [caap_gts[i]], iou_thresholds)
		snap_i = compute_snap([snap_preds[i]], [snap_gts[i]], sim_thresholds)
		rows.append({"image_id": image_id, "caap": float(caap_i), "snap": float(snap_i)})
	return rows


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--annotations", required=True)
	parser.add_argument("--images", required=True)
	parser.add_argument("--n", type=int, default=200, help="number of images to evaluate (subset)")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--predictions-out", default="eval_out/coco_val_subset_predictions.jsonl")
	parser.add_argument("--report-out", default="eval_out/coco_val_subset_report.json")
	parser.add_argument("--skip-inference", action="store_true",
						 help="reuse existing predictions-out file, skip running the model")
	args = parser.parse_args()

	full_gt = CocoGroundTruth(args.annotations, args.images)
	gt = full_gt.subset(args.n, seed=args.seed)
	print(f"Evaluating on {len(gt)} images (seed={args.seed}) out of {len(full_gt)} total in {args.annotations}")

	if not args.skip_inference:
		run_inference(gt, args.predictions_out, limit=args.n)

	preds_by_image_id = load_predictions(args.predictions_out)
	image_ids, caap_preds, caap_gts, snap_preds, snap_gts = build_eval_inputs(gt, preds_by_image_id)

	print(f"\nComputing corpus-level CAAP/SNAP over {len(image_ids)} images...")
	caap_score = compute_ca_ap(caap_preds, caap_gts, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
	snap_score = compute_snap(snap_preds, snap_gts, [0.6, 0.7, 0.8, 0.9])
	print(f"CAAP = {caap_score:.4f}")
	print(f"SNAP = {snap_score:.4f}")

	print("\nComputing per-image CAAP/SNAP for checkpoint-B correlation analysis...")
	per_image = per_image_caap_snap(image_ids, caap_preds, caap_gts, snap_preds, snap_gts)

	caap_vals = [r["caap"] for r in per_image]
	snap_vals = [r["snap"] for r in per_image]
	try:
		import numpy as np
		pearson_r = float(np.corrcoef(caap_vals, snap_vals)[0, 1])
	except Exception as e:
		pearson_r = None
		print(f"Could not compute correlation: {e}")

	print(f"Per-image CAAP vs SNAP Pearson r = {pearson_r}")

	report = {
		"n_images": len(image_ids),
		"corpus_caap": caap_score,
		"corpus_snap": snap_score,
		"per_image_pearson_r": pearson_r,
		"per_image": per_image,
	}
	os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
	with open(args.report_out, "w") as f:
		json.dump(report, f, indent=2)
	print(f"\nReport written to {args.report_out}")


if __name__ == "__main__":
	main()

