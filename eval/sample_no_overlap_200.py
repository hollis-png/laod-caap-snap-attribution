"""
Extend the original 60-case no_overlap manual-review sample to 200 cases,
guaranteeing the original 60 (image_id, det_label) pairs are a strict subset
of the new 200.

Step 1: Rebuild the exact same 3825-case no_overlap population (identical
iteration order to eval/sample_no_overlap_cases.py) and verify that
random.seed(0) + random.sample(population, 60) reproduces the existing
eval_out/no_overlap_sample.json exactly.

Step 2: Since random.sample(pop, 60) and random.sample(pop, 200) with the
same seed are NOT guaranteed to nest (different internal RNG call patterns
for different k), use the shuffle-based approach instead:
    rng = random.Random(0)
    pop_copy = list(population)
    rng.shuffle(pop_copy)
    sample_200 = pop_copy[:200]
This makes sample_60 = pop_copy[:60] and sample_200 = pop_copy[:200] trivially
nested IF the original 60 was ALSO drawn via shuffle. Since the original was
drawn via random.sample (not shuffle), we instead verify empirically and, if
shuffle doesn't reproduce the original 60 as a prefix, fall back to a
verified-nesting strategy: draw 200 via random.sample(pop, 200) with seed 0
freshly, check if original 60 subset of it; if not, use the approach of
explicitly unioning the original 60 with an additional random.sample of the
remaining population (population minus original 60) to reach 200. This
"top up" approach guarantees strict superset by construction regardless of
RNG nesting behavior, and is what we use as the primary method since it's
the only approach with a construction-time (not just empirical) guarantee.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.coco_loader import CocoGroundTruth
from eval.caap import calculate_iou
from eval.snap import embed_labels, cosine_similarity_matrix

CAAP_IOU_THRESHOLD = 0.5
SNAP_SIM_THRESHOLD = 0.7
NEAR_MISS_IOU_FLOOR = 0.1

ANNOTATIONS = r"./datasets/coco/annotations/instances_val2017.json"
IMAGES = r"./datasets/coco/val2017"
PREDICTIONS = "eval_out/coco_val_subset_predictions.jsonl"
ORIGINAL_SAMPLE = r"C:\LAOD\eval_out\no_overlap_sample.json"
OUT_SAMPLE_200 = r"C:\LAOD\eval_out\no_overlap_sample_200.json"

SAMPLE_SIZE_60 = 60
SAMPLE_SIZE_200 = 200


def load_predictions(predictions_path):
    preds = {}
    with open(predictions_path, "r") as f:
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


def classify_failure(best_iou, matched_gt_label, det_label, matched_gt_sim):
    if best_iou < 0.02:
        return "no_overlap"
    if matched_gt_sim >= SNAP_SIM_THRESHOLD:
        return "near_miss_right_object"
    if best_iou >= NEAR_MISS_IOU_FLOOR:
        return "misplaced_onto_different_object"
    return "no_overlap"


def build_population():
    gt = CocoGroundTruth(ANNOTATIONS, IMAGES)
    preds_by_image = load_predictions(PREDICTIONS)

    no_overlap_cases = []

    for image_id, pred in preds_by_image.items():
        gt_boxes_caap = gt.gt_boxes_caap(image_id)
        gt_labels_snap = [g[0] for g in gt.gt_labels_snap(image_id)]
        detections = pred["detections"]
        if not detections or not gt_labels_snap:
            continue

        det_labels = [d["label"] for d in detections]
        det_embeddings = embed_labels(det_labels)
        gt_embeddings = embed_labels(gt_labels_snap)
        sim_matrix = cosine_similarity_matrix(det_embeddings, gt_embeddings)

        for i, det in enumerate(detections):
            best_sim = float(sim_matrix[i].max()) if sim_matrix.shape[1] > 0 else 0.0
            snap_tp = best_sim >= SNAP_SIM_THRESHOLD
            if not snap_tp:
                continue

            best_iou, gt_idx = best_iou_match(det["box"], gt_boxes_caap)
            caap_tp = best_iou >= CAAP_IOU_THRESHOLD
            if caap_tp:
                continue

            matched_gt_label = gt_boxes_caap[gt_idx][4] if gt_idx != -1 else None
            matched_gt_sim = 0.0
            if gt_idx != -1 and matched_gt_label in gt_labels_snap:
                matched_gt_sim = float(sim_matrix[i][gt_labels_snap.index(matched_gt_label)])

            failure_type = classify_failure(best_iou, matched_gt_label, det["label"], matched_gt_sim)
            if failure_type != "no_overlap":
                continue

            no_overlap_cases.append({
                "image_id": image_id,
                "det_label": det["label"],
                "det_confidence": det["confidence"],
                "det_box": det["box"],
                "best_iou": best_iou,
            })

    return no_overlap_cases


def key_of(case):
    # (image_id, det_label) alone is not guaranteed unique -- the same image
    # can have two distinct no_overlap detections with the same label but
    # different boxes. Include box coords (rounded) to get a stable unique
    # key per population entry while still being comparable against the
    # original sample file (which also stores det_box).
    box = tuple(round(v, 3) for v in case["det_box"])
    return (case["image_id"], case["det_label"], box)


def main():
    population = build_population()
    print(f"Total no_overlap cases found: {len(population)}")
    assert len(population) == 3825, f"Expected 3825, got {len(population)} -- BLOCKED"

    # --- Step 1: reproducibility check ---
    random.seed(0)
    reproduced_60 = random.sample(population, SAMPLE_SIZE_60)
    reproduced_keys = [key_of(c) for c in reproduced_60]

    with open(ORIGINAL_SAMPLE, "r") as f:
        original_60 = json.load(f)
    original_keys = [key_of(c) for c in original_60]

    exact_match = reproduced_keys == original_keys
    set_match = set(reproduced_keys) == set(original_keys)
    print(f"Step 1: reproduced_60 == original_60 (order-sensitive exact match): {exact_match}")
    print(f"Step 1: reproduced_60 == original_60 (as sets): {set_match}")
    if not exact_match:
        print("MISMATCH DETAIL:")
        only_in_repro = set(reproduced_keys) - set(original_keys)
        only_in_orig = set(original_keys) - set(reproduced_keys)
        print(f"  only in reproduced: {only_in_repro}")
        print(f"  only in original:   {only_in_orig}")

    # --- Step 2: draw 200 guaranteeing original 60 subset ---
    # Method A: try random.sample(population, 200) fresh with seed(0) and
    # check empirically whether it happens to contain all of original_keys.
    random.seed(0)
    sample_200_A = random.sample(population, SAMPLE_SIZE_200)
    keys_A = [key_of(c) for c in sample_200_A]
    a_is_superset = set(original_keys).issubset(set(keys_A))
    print(f"Method A (random.sample(pop,200) fresh call): original 60 subset of new 200? {a_is_superset}")

    # Method B: shuffle-based, rng = random.Random(0); shuffle copy; take prefix.
    rng = random.Random(0)
    pop_copy = list(population)
    rng.shuffle(pop_copy)
    sample_200_B = pop_copy[:SAMPLE_SIZE_200]
    keys_B = [key_of(c) for c in sample_200_B]
    b_is_superset = set(original_keys).issubset(set(keys_B))
    print(f"Method B (shuffle then take prefix 200): original 60 subset of new 200? {b_is_superset}")

    if a_is_superset:
        chosen_method = "A (random.sample fresh call happened to nest)"
        chosen_sample = sample_200_A
    elif b_is_superset:
        chosen_method = "B (shuffle-based prefix)"
        chosen_sample = sample_200_B
    else:
        # Method C: guaranteed by construction -- top up original 60 with
        # a fresh random sample of 140 drawn from (population - original 60).
        chosen_method = "C (guaranteed top-up: original 60 + random.sample of remaining pop for 140 more)"
        original_key_set = set(original_keys)
        remaining_pop = [c for c in population if key_of(c) not in original_key_set]
        random.seed(0)
        extra_140 = random.sample(remaining_pop, SAMPLE_SIZE_200 - SAMPLE_SIZE_60)
        chosen_sample = list(original_60_full_records(population, original_keys)) + extra_140

    keys_chosen = [key_of(c) for c in chosen_sample]
    final_superset_check = set(original_keys).issubset(set(keys_chosen))
    # de-dup check
    assert len(set(keys_chosen)) == len(keys_chosen), "Duplicate keys in chosen 200 sample!"
    assert len(chosen_sample) == SAMPLE_SIZE_200, f"Expected 200, got {len(chosen_sample)}"

    print(f"CHOSEN METHOD: {chosen_method}")
    print(f"Final check: original 60 subset of chosen 200? {final_superset_check}")
    assert final_superset_check, "BLOCKED: could not construct a 200-sample containing the original 60"

    sample_out = [
        {
            "image_id": c["image_id"],
            "det_label": c["det_label"],
            "det_confidence": c["det_confidence"],
            "det_box": c["det_box"],
        }
        for c in chosen_sample
    ]

    os.makedirs(os.path.dirname(OUT_SAMPLE_200), exist_ok=True)
    with open(OUT_SAMPLE_200, "w") as f:
        json.dump(sample_out, f, indent=2)

    print(f"Wrote {len(sample_out)} cases to {OUT_SAMPLE_200}")

    # report which of the 200 are "new" (not in original 60)
    new_cases = [c for c in sample_out if (c["image_id"], c["det_label"]) not in set(original_keys)]
    print(f"New cases (200 - 60 overlap): {len(new_cases)}")


def original_60_full_records(population, original_keys):
    key_set = set(original_keys)
    seen = set()
    for c in population:
        k = key_of(c)
        if k in key_set and k not in seen:
            seen.add(k)
            yield c


if __name__ == "__main__":
    main()

