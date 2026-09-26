"""
Render the 140 NEW no_overlap cases (200-sample minus original 60) to a new
output directory, reusing the exact drawing logic from
eval/render_no_overlap_cases.py (RED = detection box, GREEN = all COCO GT
boxes), but pointed at a different input/output so the original 60 images
are left untouched.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from eval.coco_loader import CocoGroundTruth

ANNOTATIONS = r"./datasets/coco/annotations/instances_val2017.json"
IMAGES_DIR = r"./datasets/coco/val2017"
SAMPLE_PATH = r"C:\LAOD\eval_out\no_overlap_sample_new140.json"
OUT_DIR = r"C:\LAOD\eval_out\no_overlap_images_200"

RED = (0, 0, 255)      # BGR
GREEN = (0, 200, 0)    # BGR


def sanitize(label):
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip())
    return s.strip("_") or "label"


def draw_box(img, box, color, text, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 1)
        ty1 = max(0, y1 - th - baseline - 4)
        cv2.rectangle(img, (x1, ty1), (x1 + tw + 4, ty1 + th + baseline + 4), color, -1)
        text_color = (255, 255, 255)
        cv2.putText(img, text, (x1 + 2, ty1 + th + 1), font, font_scale, text_color, 1, cv2.LINE_AA)


def main():
    gt = CocoGroundTruth(ANNOTATIONS, IMAGES_DIR)

    with open(SAMPLE_PATH, "r") as f:
        cases = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    succeeded = 0
    failed = 0
    failures = []
    used_names = {}

    for case in cases:
        image_id = case["image_id"]
        det_label = case["det_label"]
        det_confidence = case["det_confidence"]
        det_box = case["det_box"]

        img_filename = f"{image_id:012d}.jpg"
        img_path = os.path.join(IMAGES_DIR, img_filename)

        if not os.path.exists(img_path):
            failed += 1
            failures.append((image_id, det_label, f"image file not found: {img_path}"))
            continue

        img = cv2.imread(img_path)
        if img is None:
            failed += 1
            failures.append((image_id, det_label, f"cv2.imread returned None for: {img_path}"))
            continue

        try:
            gt_boxes = gt.gt_boxes_caap(image_id)
        except Exception as e:
            gt_boxes = []
            failures.append((image_id, det_label, f"gt_boxes_caap error (non-fatal, drew 0 GT boxes): {e}"))

        for gt_box in gt_boxes:
            box_coords = gt_box[:4]
            gt_class = gt_box[4]
            draw_box(img, box_coords, GREEN, gt_class)

        det_text = f"{det_label} ({det_confidence:.2f})"
        draw_box(img, det_box, RED, det_text)

        base_name = f"{image_id}_{sanitize(det_label)}"
        count = used_names.get(base_name, 0)
        used_names[base_name] = count + 1
        out_name = f"{base_name}.jpg" if count == 0 else f"{base_name}_{count}.jpg"
        out_path = os.path.join(OUT_DIR, out_name)
        ok = cv2.imwrite(out_path, img)
        if not ok:
            failed += 1
            failures.append((image_id, det_label, f"cv2.imwrite failed for: {out_path}"))
            continue

        succeeded += 1

    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    if failures:
        print("Failure details:")
        for image_id, det_label, reason in failures:
            print(f"  image_id={image_id} det_label={det_label!r}: {reason}")


if __name__ == "__main__":
    main()

