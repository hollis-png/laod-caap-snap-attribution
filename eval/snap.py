import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizerFast

_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
_clip_model = None
_clip_tokenizer = None


def _load_clip():
	"""Lazily load the CLIP text encoder used for SNAP's semantic similarity."""
	global _clip_model, _clip_tokenizer
	if _clip_model is None:
		_clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME)
		_clip_model.eval()
		_clip_tokenizer = CLIPTokenizerFast.from_pretrained(_CLIP_MODEL_NAME)
	return _clip_model, _clip_tokenizer


def embed_labels(labels, device="cpu"):
	"""
	Computes L2-normalized CLIP text embeddings for a list of class-name strings.

	Args:
		labels (list of str): class names to embed.
		device (str): torch device for the CLIP model.

	Returns:
		np.ndarray of shape (len(labels), d): normalized embeddings.
	"""
	if len(labels) == 0:
		return np.zeros((0, 512), dtype=np.float32)

	model, tokenizer = _load_clip()
	model = model.to(device)

	inputs = tokenizer(labels, padding=True, return_tensors="pt").to(device)
	with torch.no_grad():
		output = model.get_text_features(**inputs)

	# transformers>=4.5x returns a BaseModelOutputWithPooling instead of a raw
	# tensor for get_text_features; older versions returned the tensor directly.
	text_features = output.pooler_output if hasattr(output, "pooler_output") else output

	text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
	return text_features.cpu().numpy()


def cosine_similarity_matrix(emb_a, emb_b):
	"""
	Pairwise cosine similarity between two sets of already-normalized embeddings.

	Args:
		emb_a (np.ndarray): shape (n, d).
		emb_b (np.ndarray): shape (m, d).

	Returns:
		np.ndarray of shape (n, m).
	"""
	if emb_a.shape[0] == 0 or emb_b.shape[0] == 0:
		return np.zeros((emb_a.shape[0], emb_b.shape[0]))
	return emb_a @ emb_b.T


def compute_ap_single_image_semantic(predictions_per_image, ground_truths_per_image, sim_threshold, pred_embeddings=None, gt_embeddings=None):
	"""
	Computes semantic-matching TP/FP for a single image at a single similarity threshold,
	mirroring caap.compute_ap_single_image but matching on label semantics instead of box IoU.

	Args:
		predictions_per_image (list of list): [class_name, confidence] per predicted label.
		ground_truths_per_image (list of list): [class_name] per ground-truth label
			(box coordinates are irrelevant to SNAP and may be included but are ignored).
		sim_threshold (float): cosine similarity threshold for considering a match (tau_s).
		pred_embeddings (np.ndarray, optional): precomputed CLIP embeddings for
			predictions_per_image labels, shape (num_pred, d). Computed if not given.
		gt_embeddings (np.ndarray, optional): precomputed CLIP embeddings for
			ground_truths_per_image labels, shape (num_gt, d). Computed if not given.

	Returns:
		tuple: (true_positives, false_positives, num_gt) matching caap's return shape.
	"""
	num_gt = len(ground_truths_per_image)
	num_pred = len(predictions_per_image)

	if num_pred == 0 and num_gt == 0:
		return [], [], 0
	if num_pred == 0 and num_gt > 0:
		return [], [], num_gt

	predictions_per_image = sorted(predictions_per_image, key=lambda x: x[1], reverse=True)

	if pred_embeddings is None:
		pred_embeddings = embed_labels([p[0] for p in predictions_per_image])
	if gt_embeddings is None:
		gt_embeddings = embed_labels([g[0] for g in ground_truths_per_image])

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


def compute_snap(all_predictions, all_ground_truths, sim_thresholds=(0.6, 0.7, 0.8, 0.9)):
	"""
	Computes Semantic Naming Average Precision (SNAP) across multiple images and
	similarity thresholds (SNAP@.6:.9 as defined in the LAOD paper, Sec 4.2).

	Unlike CAAP, matching here ignores bounding-box geometry entirely: a prediction
	is a true positive if its CLIP text-embedding cosine similarity to some
	unmatched ground-truth label exceeds the threshold, independent of localization.

	Args:
		all_predictions (list of list of list): all_predictions[i] = list of
			[class_name, confidence] for image i.
		all_ground_truths (list of list of list): all_ground_truths[i] = list of
			[class_name] (or [class_name, ...] with box fields ignored) for image i.
		sim_thresholds (iterable of float): similarity thresholds to average over.

	Returns:
		float: the mean SNAP over all images and specified similarity thresholds.
	"""
	# Embeddings only depend on the label strings, not the threshold, so compute once.
	pred_embeddings_per_image = [
		embed_labels([p[0] for p in img_preds]) for img_preds in all_predictions
	]
	gt_embeddings_per_image = [
		embed_labels([g[0] for g in img_gts]) for img_gts in all_ground_truths
	]

	overall_aps = []

	for sim_thresh in sim_thresholds:
		matched_pairs = []  # (confidence, is_tp)

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

	return np.mean(overall_aps)

