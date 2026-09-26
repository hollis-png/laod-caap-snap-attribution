"""
Runs Grounding DINO detection over cached llm_labels from the validated
YOLO-World run (coco_val_subset_predictions.jsonl), skipping the LLM step
entirely. Writes predictions in the same schema to gdino_predictions.jsonl.

Usage:
	python eval/run_gdino_eval.py --annotations D:\\datasets\\coco\\annotations\\instances_val2017.json \
		--images D:\\datasets\\coco\\val2017 \
		--source-predictions eval_out/coco_val_subset_predictions.jsonl \
		--predictions-out eval_out/gdino_predictions.jsonl \
		[--limit N]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from eval.coco_loader import CocoGroundTruth

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None


def _get_model():
	global _processor, _model
	if _model is None:
		_processor = AutoProcessor.from_pretrained(MODEL_ID)
		_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
		_model.eval()
	return _processor, _model


def gdino_detect(image, llm_labels):
	"""
	Runs Grounding DINO on `image` using an already-generated llm_labels list
	(same normalization laod_gdino() applies: pedestrian/people/man/woman -> person),
	skipping the Gemma LLM call. Returns list of {"box","label","confidence"} dicts.
	"""
	processor, model = _get_model()

	norm_labels = []
	for label in llm_labels:
		l = label.lower().strip()
		l = l.replace("pedestrian", "person").replace("people", "person")
		l = l.replace("man", "person").replace("woman", "person")
		if l:
			norm_labels.append(l)

	if not norm_labels:
		return []

	# Grounding DINO's canonical prompt format is a single dot-joined string,
	# e.g. "a cat. a dog.". Newer `transformers` processor versions can accept
	# a list-of-candidate-labels and auto-merge it, but that auto-merge silently
	# breaks if any label itself contains a "." (see
	# `_is_list_of_candidate_labels` in processing_grounding_dino.py), so we
	# build the merged string ourselves for compatibility across versions.
	merged_text = ". ".join(label.rstrip(".") for label in norm_labels) + "."

	inputs = processor(images=image, text=merged_text, return_tensors="pt").to(DEVICE)
	with torch.no_grad():
		outputs = model(**inputs)

	results = processor.post_process_grounded_object_detection(
		outputs,
		inputs.input_ids,
		threshold=0.4,
		text_threshold=0.3,
		target_sizes=[image.size[::-1]],
	)

	result = results[0]
	detections = []
	for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
		x1, y1, x2, y2 = [float(v) for v in box.tolist()]
		detections.append({
			"box": [x1, y1, x2, y2],
			"label": label,
			"confidence": float(score.item()),
		})
	return detections


def load_source_predictions(path):
	preds = {}
	with open(path, "r") as f:
		for line in f:
			if line.strip():
				rec = json.loads(line)
				preds[rec["image_id"]] = rec
	return preds


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--annotations", required=True)
	parser.add_argument("--images", required=True)
	parser.add_argument("--source-predictions", default="eval_out/coco_val_subset_predictions.jsonl")
	parser.add_argument("--predictions-out", default="eval_out/gdino_predictions.jsonl")
	parser.add_argument("--limit", type=int, default=None)
	parser.add_argument("--resume", action="store_true", default=True)
	args = parser.parse_args()

	gt = CocoGroundTruth(args.annotations, args.images)
	source_preds = load_source_predictions(args.source_predictions)
	image_ids = [iid for iid in gt.image_ids if iid in source_preds]
	if args.limit is not None:
		image_ids = image_ids[: args.limit]

	os.makedirs(os.path.dirname(args.predictions_out) or ".", exist_ok=True)

	done_ids = set()
	if args.resume and os.path.exists(args.predictions_out):
		with open(args.predictions_out, "r") as f:
			for line in f:
				if line.strip():
					done_ids.add(json.loads(line)["image_id"])
		print(f"Resuming: {len(done_ids)} images already processed, skipping them.")

	with open(args.predictions_out, "a") as f:
		for i, image_id in enumerate(image_ids):
			if image_id in done_ids:
				continue

			llm_labels = source_preds[image_id]["llm_labels"]
			image = Image.open(gt.image_path(image_id)).convert("RGB")

			t0 = time.time()
			detections = gdino_detect(image, llm_labels)
			elapsed = time.time() - t0

			record = {
				"image_id": image_id,
				"llm_labels": llm_labels,
				"detections": detections,
				"elapsed_sec": elapsed,
			}
			f.write(json.dumps(record) + "\n")
			f.flush()

			print(f"[{i+1}/{len(image_ids)}] image_id={image_id} "
				  f"n_labels={len(llm_labels)} n_det={len(detections)} ({elapsed:.2f}s)")


if __name__ == "__main__":
	main()

