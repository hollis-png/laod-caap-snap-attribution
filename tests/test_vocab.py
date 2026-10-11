"""Regression tests for eval/vocab.py (the first release misfiled horses, buses, vases, suitcases and ties as out-of-vocabulary)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.vocab import COCO80_NAMES, normalize_plural, vocabulary_category


def test_native_names_are_native():
	assert len(COCO80_NAMES) == 80
	for name in COCO80_NAMES:
		assert vocabulary_category(name) == "native"


def test_regular_plurals_of_coco_names():
	for plural in ["horses", "buses", "vases", "suitcases", "ties", "chairs", "cows", "bottles", "benches", "couches",
	               "ovens", "tvs", "remotes", "bananas", "knives", "people", "mice"]:
		assert vocabulary_category(plural) == "plural_of_native", plural


def test_singular_of_plural_coco_names():
	assert vocabulary_category("ski") == "plural_of_native"
	assert vocabulary_category("scissor") == "plural_of_native"


def test_out_of_vocabulary_stays_out():
	for label in ["plate", "cabinet", "helmet", "glasses", "boxes", "sky", "buss", "hors"]:
		assert vocabulary_category(label) == "out_of_vocabulary", label


def test_normalize_plural_returns_coco_name():
	assert normalize_plural("horses") == "horse"
	assert normalize_plural("buses") == "bus"
	assert normalize_plural("bus") == "bus"
	assert normalize_plural("vases") == "vase"
	assert normalize_plural("suitcases") == "suitcase"
