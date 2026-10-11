# Released data

Both files are outputs of the pipeline (Gemma-3 4B + YOLO-World X) on the COCO val2017 split (5,000 images). They contain no images and no COCO annotation text; image ids and boxes refer to COCO val2017.

## `coco_val_subset_predictions.jsonl`
One JSON object per image (5,000 lines). Fields: `image_id` (COCO id), `llm_labels` (the object names the LLM produced), `detections` (list of `{box, label, confidence}`), `elapsed_sec`.

## `detections.csv`
One row per detection (27,273 rows). Columns: `image_id`; `label` (LLM label); `conf` (detector confidence); `box` (x1, y1, x2, y2); `best_iou` (IoU with the closest COCO box); `caap_tp` (1 if the detection has IoU ≥ 0.5 with some ground-truth box of its image, the detection-level rule of §3.2; 15,134 of 27,273 detections, not the one-to-one matching of the AP metric, which gives 14,250); `max_sim_img` (highest CLIP text similarity between the label and any ground-truth label of the image); `near_cls`, `near_sim` (class and similarity of the nearest ground-truth box); `exact` (1 if the label string equals that class, a literal match only; the paper's name-matching rule for the failure groups is wider and is in `attribution_recompute.py`); `vocab` (`native`, `plural_of_native` or `out_of_vocabulary`).

COCO annotations (CC BY 4.0) were used to compute `best_iou`, `near_cls` and the similarity columns. COCO images are not included; download them from the COCO site.

The `vocab` and `exact` columns were recomputed on 2026-10-12 after a bug fix in the vocabulary classification (see the correction notice in the repository README): 134 detections whose labels are plural forms of COCO-80 names (horses, buses, suitcases, vases) are now `plural_of_native`, and `exact` now matches singular forms of the label against the nearest class.
