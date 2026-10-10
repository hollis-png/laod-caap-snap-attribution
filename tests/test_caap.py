"""Regression tests for eval.caap.compute_ca_ap.

The first test pins down the bug found in the 2026-10 self-review: the original corpus-level function re-matched all pooled
detections against the ground-truth boxes of ALL images, so a detection could match a box of a different image.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.caap import compute_ca_ap  # noqa: E402

ALL_TH = [0.5 + 0.05 * k for k in range(10)]


def det(x1, y1, x2, y2, conf):
    return [x1, y1, x2, y2, conf]


def gt(x1, y1, x2, y2):
    return [x1, y1, x2, y2, "obj"]


def test_detection_cannot_match_a_box_in_another_image():
    # image 0 has a detection and no GT; image 1 has a GT at exactly the same coordinates and no detection
    preds = [[det(10, 10, 50, 50, 0.9)], []]
    gts = [[], [gt(10, 10, 50, 50)]]
    assert compute_ca_ap(preds, gts, [0.5]) == 0.0


def test_perfect_detections_score_one():
    preds = [[det(10, 10, 50, 50, 0.9)], [det(0, 0, 20, 20, 0.8)]]
    gts = [[gt(10, 10, 50, 50)], [gt(0, 0, 20, 20)]]
    assert compute_ca_ap(preds, gts, ALL_TH) == pytest.approx(1.0)


def test_ground_truth_box_is_matched_at_most_once():
    # two detections on one GT: the second is a duplicate (false positive), ranked above nothing else
    preds = [[det(10, 10, 50, 50, 0.9), det(10, 10, 50, 50, 0.8)]]
    gts = [[gt(10, 10, 50, 50)]]
    assert compute_ca_ap(preds, gts, [0.5]) == pytest.approx(1.0)  # the TP comes first, so precision at full recall is 1
    preds = [[det(100, 100, 120, 120, 0.9), det(10, 10, 50, 50, 0.8)]]  # a miss ranked above the hit
    assert compute_ca_ap(preds, gts, [0.5]) == pytest.approx(0.5)


def test_iou_threshold_is_respected():
    # IoU of these two boxes is 0.6: a hit at thresholds <= 0.6, a miss above
    preds = [[det(0, 0, 10, 10, 0.9)]]
    gts = [[gt(0, 0, 10, 6)]]
    assert compute_ca_ap(preds, gts, [0.5]) == pytest.approx(1.0)
    assert compute_ca_ap(preds, gts, [0.7]) == pytest.approx(0.0)
