"""Regression tests from the 2026-10 independent code review.

1. SNAP must not depend on the order in which predictions are passed (embeddings were once misaligned with the confidence-sorted predictions).
2. A hand-computed two-image AP value that exercises the precision envelope.
No CLIP weights are needed: the text embedder is replaced by a one-hot stub.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import snap  # noqa: E402
from eval.caap import compute_ca_ap  # noqa: E402

VOCAB = {"cat": 0, "dog": 1, "car": 2}


def stub_embed(labels):
    out = np.zeros((len(labels), len(VOCAB)))
    for i, name in enumerate(labels):
        out[i, VOCAB[name]] = 1.0
    return out


@pytest.fixture(autouse=True)
def _stub_embedder(monkeypatch):
    monkeypatch.setattr(snap, "embed_labels", stub_embed)


def test_snap_independent_of_prediction_order():
    gts = [[["cat"]]]
    sorted_preds = [[["cat", 0.9], ["dog", 0.2]]]
    shuffled_preds = [[["dog", 0.2], ["cat", 0.9]]]
    expected = snap.compute_snap(sorted_preds, gts, sim_thresholds=(0.9,))
    assert expected == pytest.approx(1.0)
    assert snap.compute_snap(shuffled_preds, gts, sim_thresholds=(0.9,)) == pytest.approx(expected)


def test_hand_computed_two_image_ap_with_envelope():
    # image 0: one GT, a true positive (0.9) and a duplicate on the same GT (0.8, false positive); image 1: one GT, a true positive (0.7)
    # ranked TP, FP, TP -> precision 1, 1/2, 2/3; recall 1/2, 1/2, 1; envelope 1, 2/3, 2/3; AP = 1*1/2 + 2/3*0 + 2/3*1/2 = 5/6
    box = [10, 10, 50, 50]
    preds = [[box + [0.9], box + [0.8]], [box + [0.7]]]
    gts = [[box + ["obj"]], [box + ["obj"]]]
    assert compute_ca_ap(preds, gts, [0.5]) == pytest.approx(5 / 6)
