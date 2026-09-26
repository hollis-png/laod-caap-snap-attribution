"""
Exploratory script: recompute corpus-level SNAP on a 200-image subset of
COCO-Val using a parameterized CLIP checkpoint, to check whether checkpoint
choice explains the ~0.08-0.1 gap between our SNAP=0.4559 (full 5000 images,
openai/clip-vit-base-patch32) and the paper's reported ~0.54 for its "LO"
bucket.

This is a standalone script (does not import or modify eval/snap.py or
eval/run_batch_eval.py) but faithfully reproduces:
  - the text-embedding + pooling logic from eval/snap.py's embed_labels()
    (including the get_text_features() -> pooler_output workaround)
  - the exact corpus-level SNAP aggregation formula from eval/snap.py's
    compute_snap() (SNAP@.6:.9, four similarity thresholds, COCO-style
    interpolated-precision AP pooled across images, averaged over thresholds)
  - the same GT/prediction extraction shape used by eval/run_batch_eval.py's
    build_eval_inputs() (snap_preds = [[label, confidence], ...],
    snap_gts = gt.gt_labels_snap(image_id))

Usage:
    conda run -n laod python eval/snap_checkpoint_compare.py --checkpoint openai/clip-vit-base-patch32
    conda run -n laod python eval/snap_checkpoint_compare.py --checkpoint openai/clip-vit-large-patch14
    conda run -n laod python eval/snap_checkpoint_compare.py --checkpoint openai/clip-vit-base-patch16
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.coco_loader import CocoGroundTruth

ANNOTATIONS = r"./datasets/coco/annotations/instances_val2017.json"
IMAGES = r"./datasets/coco/val2017"
PREDICTIONS_PATH = r"C:\LAOD\eval_out\coco_val_subset_predictions.jsonl"
N_SUBSET = 200
SIM_THRESHOLDS = (0.6, 0.7, 0.8, 0.9)


# ---------------------------------------------------------------------------
# Parameterized CLIP embedding (mirrors eval/snap.py embed_labels(), but takes
# a checkpoint name instead of a hardcoded module-level constant).
# ---------------------------------------------------------------------------

_model_cache = {}


def _load_clip(checkpoint_name):
	if checkpoint_name not in _model_cache:
		model = CLIPModel.from_pretrained(checkpoint_name)
		model.eval()
		tokenizer = CLIPTokenizerFast.from_pretrained(checkpoint_name)
		_model_cache[checkpoint_name] = (model, tokenizer)
	return _model_cache[checkpoint_name]


def embed_labels(labels, checkpoint_name, device="cpu"):
	if len(labels) == 0:
		return np.zeros((0, 512), dtype=np.float32)

	model, tokenizer = _load_clip(checkpoint_name)
	model = model.to(device)

	inputs = tokenizer(labels, padding=True, return_tensors="pt").to(device)
	with torch.no_grad():
		output = model.get_text_features(**inputs)

	# Same workaround as eval/snap.py: newer transformers versions return a
	# BaseModelOutputWithPooling instead of a raw tensor from
	# get_text_features(); older versions returned the tensor directly.
	text_features = output.pooler_output if hasattr(output, "pooler_output") else output

	text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
	return text_features.cpu().numpy()


def cosine_similarity_matrix(emb_a, emb_b):
	if emb_a.shape[0] == 0 or emb_b.shape[0] == 0:
		return np.zeros((emb_a.shape[0], emb_b.shape[0]))
	return emb_a @ emb_b.T


# ---------------------------------------------------------------------------
# Corpus-level SNAP aggregation, copied faithfully from eval/snap.py
# compute_ap_single_image_semantic() + compute_snap(), only parameterized
# on checkpoint_name so embed_labels() routes to the requested CLIP model.
# ---------------------------------------------------------------------------

def compute_ap_single_image_semantic(predictions_per_image, ground_truths_per_image,
									  sim_threshold, pred_embeddings, gt_embeddings):
	num_gt = len(ground_truths_per_image)
	num_pred = len(predictions_per_image)

	if num_pred == 0 and num_gt == 0:
		return [], [], 0
	if num_pred == 0 and num_gt > 0:
		return [], [], num_gt

	predictions_per_image = sorted(predictions_per_image, key=lambda x: x[1], reverse=True)

	sim_matrix = cosine_similarity_matrix(pred_embeddings, gt_embeddings)

	true_positives = np.zeros(num_pred)
	false_positives = np.zeros(num_pred)
	gt_matched = np.zeros(num_gt, dtype=bool)

	for i in range(num_pred):
		if num_gt == 0:
			false_positives[i] = 1
			continue

		sims = sim_matrix[i].copy()
		sims[gt_matched] = -1.0
		best_gt_idx = int(np.argmax(sims))
		best_sim = sims[best_gt_idx]

		if best_sim >= sim_threshold:
			true_positives[i] = 1
			gt_matched[best_gt_idx] = True
		else:
			false_positives[i] = 1

	return true_positives, false_positives, num_gt


def compute_snap(all_predictions, all_ground_truths, checkpoint_name,
				  sim_thresholds=SIM_THRESHOLDS):
	pred_embeddings_per_image = [
		embed_labels([p[0] for p in img_preds], checkpoint_name) for img_preds in all_predictions
	]
	gt_embeddings_per_image = [
		embed_labels([g[0] for g in img_gts], checkpoint_name) for img_gts in all_ground_truths
	]

	overall_aps = []

	for sim_thresh in sim_thresholds:
		matched_pairs = []

		for img_idx in range(len(all_predictions)):
			predictions_img = all_predictions[img_idx]
			ground_truths_img = all_ground_truths[img_idx]

			tps_img, fps_img, _ = compute_ap_single_image_semantic(
				predictions_img,
				ground_truths_img,
				sim_thresh,
				pred_embeddings=pred_embeddings_per_image[img_idx],
				gt_embeddings=gt_embeddings_per_image[img_idx],
			)

			sorted_preds = sorted(predictions_img, key=lambda x: x[1], reverse=True)
			for pred, is_tp in zip(sorted_preds, tps_img):
				matched_pairs.append((pred[1], int(is_tp)))

		total_num_gt = sum(len(g) for g in all_ground_truths)

		if total_num_gt == 0:
			overall_aps.append(1.0 if len(matched_pairs) == 0 else 0.0)
			continue

		if not matched_pairs:
			overall_aps.append(0.0)
			continue

		matched_pairs = sorted(matched_pairs, key=lambda x: x[0], reverse=True)

		tps_cumulative = np.cumsum([m[1] for m in matched_pairs])
		fps_cumulative = np.cumsum([1 - m[1] for m in matched_pairs])

		precision = tps_cumulative / (tps_cumulative + fps_cumulative)
		recall = tps_cumulative / total_num_gt

		interpolated_precision = np.maximum.accumulate(precision[::-1])[::-1]
		ap = np.sum(interpolated_precision * np.diff(np.concatenate(([0.0], recall))))

		overall_aps.append(ap)

	return float(np.mean(overall_aps))


# ---------------------------------------------------------------------------
# Data loading: first 200 images by image_id order from the predictions file
# (fallback, since no separate 200-image predictions file distinct from the
# full 5000-image predictions.jsonl was found in eval_out/).
# ---------------------------------------------------------------------------

def load_predictions(predictions_path):
	preds_by_image_id = {}
	with open(predictions_path, "r") as f:
		for line in f:
			if line.strip():
				rec = json.loads(line)
				preds_by_image_id[rec["image_id"]] = rec
	return preds_by_image_id


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", required=True)
	parser.add_argument("--n", type=int, default=N_SUBSET)
	args = parser.parse_args()

	gt = CocoGroundTruth(ANNOTATIONS, IMAGES)
	preds_by_image_id = load_predictions(PREDICTIONS_PATH)

	# First N images by image_id order (gt.image_ids is already sorted),
	# restricted to those that actually have cached predictions.
	image_ids = [iid for iid in gt.image_ids if iid in preds_by_image_id][: args.n]

	snap_preds, snap_gts = [], []
	for image_id in image_ids:
		pred = preds_by_image_id[image_id]
		snap_preds.append([[d["label"], d["confidence"]] for d in pred["detections"]])
		snap_gts.append(gt.gt_labels_snap(image_id))

	print(f"Checkpoint={args.checkpoint} n_images={len(image_ids)} "
		  f"first_id={image_ids[0]} last_id={image_ids[-1]}")

	snap_score = compute_snap(snap_preds, snap_gts, args.checkpoint)
	print(f"SNAP({args.checkpoint}, n={len(image_ids)}) = {snap_score:.4f}")

	out_path = os.path.join(
		os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
		"eval_out",
		f"snap_checkpoint_compare_{args.checkpoint.replace('/', '_')}.json",
	)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, "w") as f:
		json.dump({
			"checkpoint": args.checkpoint,
			"n_images": len(image_ids),
			"first_image_id": image_ids[0],
			"last_image_id": image_ids[-1],
			"snap": snap_score,
		}, f, indent=2)
	print(f"Wrote {out_path}")


if __name__ == "__main__":
	main()

