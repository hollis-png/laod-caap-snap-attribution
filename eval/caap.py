import numpy as np

def calculate_iou(box1, box2):
	"""
	Calculates the Intersection over Union (IoU) of two bounding boxes.
	Boxes are expected in format [x_min, y_min, x_max, y_max].
	"""
	x_min_inter = max(box1[0], box2[0])
	y_min_inter = max(box1[1], box2[1])
	x_max_inter = min(box1[2], box2[2])
	y_max_inter = min(box1[3], box2[3])

	inter_width = max(0, x_max_inter - x_min_inter)
	inter_height = max(0, y_max_inter - y_min_inter)
	inter_area = inter_width * inter_height

	box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
	box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

	union_area = box1_area + box2_area - inter_area

	if union_area == 0:
		return 0.0

	return inter_area / union_area


def compute_ap_single_image(predictions_per_image, ground_truths_per_image, iou_threshold):
	"""
	Computes Average Precision (AP) for a single image at a single IoU threshold in a class-agnostic manner.

	Args:
		predictions_per_image (list of list): Predicted detections for one image,
											 format: [x1, y1, x2, y2, confidence].
		ground_truths_per_image (list of list): Ground truths for one image,
												format: [x1, y1, x2, y2, class_category].
												Only box coordinates are used for CA-AP.
		iou_threshold (float): The IoU threshold for considering a detection valid.

	Returns:
		tuple: (list of int: True/False positives, list of int: False negatives mask)
	"""
	num_gt = len(ground_truths_per_image)
	num_pred = len(predictions_per_image)

	if num_pred == 0 and num_gt == 0:
		return [], [], 0  # No detections, no GTs, contributes neutrally to AP calculation

	if num_pred == 0 and num_gt > 0:
		# All ground truths are false negatives if there are no predictions
		return [], [], num_gt  # FP list empty, TP list empty, total FNs = num_gt

	# Sort predictions by confidence in descending order
	predictions_per_image = sorted(predictions_per_image, key=lambda x: x[4], reverse=True)

	true_positives = np.zeros(num_pred)
	false_positives = np.zeros(num_pred)

	gt_matched = np.zeros(num_gt, dtype=bool)

	for i, pred_data in enumerate(predictions_per_image):
		pred_box = pred_data[:4]  # Extract box coordinates

		best_iou = 0.0
		best_gt_idx = -1

		for gt_idx, gt_data in enumerate(ground_truths_per_image):
			gt_box = gt_data[:4]  # Extract box coordinates

			if not gt_matched[gt_idx]:
				iou = calculate_iou(pred_box, gt_box)
				if iou > best_iou:
					best_iou = iou
					best_gt_idx = gt_idx

		if best_gt_idx != -1 and best_iou >= iou_threshold:
			true_positives[i] = 1
			gt_matched[best_gt_idx] = True
		else:
			false_positives[i] = 1

	return true_positives, false_positives, num_gt  # Return counts for cumulative calculation


def compute_ca_ap(all_predictions, all_ground_truths, iou_thresholds):
	"""
	Computes Class-Agnostic Average Precision (CA-AP) across multiple images and IoU thresholds.

	Matching is done per image (a detection can only match a ground-truth box of its own image,
	each ground-truth box at most once); the resulting (confidence, is_tp) pairs of all images are then
	pooled and ranked globally to build one precision-recall curve per IoU threshold.

	(An earlier version re-matched the pooled detections against the ground-truth boxes of ALL images,
	which let a detection match a box in a different image and inflated the corpus-level score,
	increasingly so for larger corpora. Single-image calls were not affected.)

	Args:
		all_predictions (list of list of list): predictions[i] = list of detections for image i,
											  each [x1, y1, x2, y2, confidence].
		all_ground_truths (list of list of list): ground_truths[i] = list of GTs for image i,
												 each [x1, y1, x2, y2, class_category].
		iou_thresholds (list of float): e.g. [0.5, 0.55, ..., 0.95].

	Returns:
		float: mean CA-AP over the specified IoU thresholds.
	"""
	overall_aps = []

	for iou_thresh in iou_thresholds:
		matched_pairs = []  # (confidence, is_tp) over all images
		total_num_gt = 0

		for predictions_img, ground_truths_img in zip(all_predictions, all_ground_truths):
			tps_img, _fps_img, num_gt_img = compute_ap_single_image(predictions_img, ground_truths_img, iou_thresh)
			total_num_gt += num_gt_img
			if len(predictions_img) > 0 and len(tps_img) > 0:
				sorted_conf = sorted((p[4] for p in predictions_img), reverse=True)  # same order as tps_img
				matched_pairs.extend(zip(sorted_conf, (int(t) for t in tps_img)))

		if total_num_gt == 0:
			overall_aps.append(1.0 if not matched_pairs else 0.0)
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
		overall_aps.append(np.sum(interpolated_precision * np.diff(np.concatenate(([0.0], recall)))))

	return np.mean(overall_aps)
