"""
Step 2: feed Qwen2.5-VL-generated labels (eval_out/qwen_llm_labels.jsonl) into
the existing validated YOLO-World open-vocabulary detection path
(eval.laod_infer's YOLO-World model + set_classes/predict pattern), producing
eval_out/qwen_yoloworld_predictions.jsonl in the same schema as the validated
coco_val_subset_predictions.jsonl / gdino_predictions.jsonl files.

Resumable: skips image_ids already present in the output file.

Usage:
    python eval/run_qwen_yoloworld_detect.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from eval.laod_infer import _get_yolo_model

IMAGES_DIR = r"./datasets/coco/val2017"
LABELS_PATH = r"C:\LAOD\eval_out\qwen_llm_labels.jsonl"
OUT_PATH = r"C:\LAOD\eval_out\qwen_yoloworld_predictions.jsonl"


def image_path(image_id):
    return f"{IMAGES_DIR}\\{image_id:012d}.jpg"


def detect_with_labels(model, image, llm_labels):
    if len(llm_labels) == 0:
        return []
    model.to("cpu")
    model.set_classes(llm_labels)
    results = model.predict(image, verbose=False)
    result = results[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_idx = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = result.names[cls_idx]
            detections.append({"box": xyxy, "label": label, "confidence": conf})
    return detections


def main():
    records = []
    with open(LABELS_PATH, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Total labeled images: {len(records)}", flush=True)

    done_ids = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r") as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line)["image_id"])
        print(f"Resuming: {len(done_ids)} already processed.", flush=True)

    todo = [r for r in records if r["image_id"] not in done_ids]
    print(f"{len(todo)} images remaining", flush=True)

    model = _get_yolo_model()

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "a") as f:
        for i, rec in enumerate(todo):
            image_id = rec["image_id"]
            llm_labels = rec["llm_labels"]
            t0 = time.time()
            try:
                image = Image.open(image_path(image_id)).convert("RGB")
                detections = detect_with_labels(model, image, llm_labels)
            except Exception as e:
                print(f"[{i+1}/{len(todo)}] image_id={image_id} ERROR: {e}", flush=True)
                continue
            elapsed = time.time() - t0

            out_record = {
                "image_id": image_id,
                "llm_labels": llm_labels,
                "detections": detections,
                "elapsed_sec": elapsed,
            }
            f.write(json.dumps(out_record) + "\n")
            f.flush()
            print(f"[{i+1}/{len(todo)}] image_id={image_id} n_det={len(detections)} ({elapsed:.2f}s)", flush=True)

    print("=== run_qwen_yoloworld_detect complete ===", flush=True)


if __name__ == "__main__":
    main()

