"""E3 step 2: run YOLO-World X (same weights, SHA-256 41e771bf...) on the regenerated Qwen labels. Same call pattern as the original
run_qwen_yoloworld_detect.py (model.to('cpu'); set_classes(labels); predict(image, verbose=False), default conf 0.25)."""
import json, os, sys, time
from PIL import Image
from ultralytics import YOLO
IMAGES = os.path.expanduser("~/laod-exp/coco/val2017"); LAB = os.path.expanduser("~/laod-exp/qwen_llm_labels_ndhu.jsonl"); OUT = os.path.expanduser("~/laod-exp/qwen_yoloworld_predictions_ndhu.jsonl")
recs = [json.loads(l) for l in open(LAB) if l.strip()]
done = {json.loads(l)["image_id"] for l in open(OUT) if l.strip()} if os.path.exists(OUT) else set()
todo = [r for r in recs if r["image_id"] not in done]; print(len(recs), "labelled;", len(todo), "to detect", flush=True)
model = YOLO(os.path.expanduser("~/laod-exp/yolov8x-worldv2.pt")); t0 = time.time()
with open(OUT, "a") as f:
    for k, r in enumerate(todo):
        dets = []
        if r["llm_labels"]:
            img = Image.open(f"{IMAGES}/{r['image_id']:012d}.jpg").convert("RGB")
            model.to("cpu"); model.set_classes(r["llm_labels"]); res = model.predict(img, verbose=False)[0]
            if res.boxes is not None:
                for b in res.boxes: dets.append({"box": b.xyxy[0].tolist(), "label": res.names[int(b.cls[0].item())], "confidence": float(b.conf[0].item())})
        f.write(json.dumps({"image_id": r["image_id"], "llm_labels": r["llm_labels"], "detections": dets}) + "\n"); f.flush()
        if k % 250 == 0: print(f"[{k+1}/{len(todo)}] n_det={len(dets)} ({time.time()-t0:.0f}s)", flush=True)
print("DONE", flush=True)
