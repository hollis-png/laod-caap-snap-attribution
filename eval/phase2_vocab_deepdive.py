"""
Phase 2, vocabulary deep-dive: RQ2 found that non-COCO-80-native wording has
roughly half the CAAP-TP rate of native wording. This script surfaces what
that "non-native wording" actually looks like -- top frequent terms, and for
each, its best-matching COCO-80 category by CLIP similarity, so we can judge
whether the gap is driven by synonyms ("vehicle" vs "car"), finer-grained
descriptions ("wooden chair"), or genuinely different naming conventions.

Usage:
	python eval/phase2_vocab_deepdive.py --predictions eval_out/coco_val_subset_predictions.jsonl \
		--out eval_out/phase2_vocab_deepdive_report.json
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.snap import embed_labels, cosine_similarity_matrix
from eval.phase2_stage2a import COCO80_NAMES, vocabulary_category, normalize_plural


def load_predictions(predictions_path):
	preds = []
	with open(predictions_path, "r") as f:
		for line in f:
			if line.strip():
				preds.append(json.loads(line))
	return preds


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--predictions", default="eval_out/coco_val_subset_predictions.jsonl")
	parser.add_argument("--out", default="eval_out/phase2_vocab_deepdive_report.json")
	parser.add_argument("--top-n", type=int, default=30)
	args = parser.parse_args()

	preds = load_predictions(args.predictions)

	native_label_counter = Counter()
	plural_label_counter = Counter()
	oov_label_counter = Counter()

	for pred in preds:
		for det in pred["detections"]:
			label = det["label"].lower().strip()
			category = vocabulary_category(label)
			if category == "native":
				native_label_counter[label] += 1
			elif category == "plural_of_native":
				plural_label_counter[label] += 1
			else:
				oov_label_counter[label] += 1

	# Only the genuinely out-of-vocabulary terms need a "what is this
	# standing in for" lookup -- plural-of-native terms already have an
	# obvious answer (their singular form).
	top_novel = oov_label_counter.most_common(args.top_n)
	novel_terms = [term for term, _ in top_novel]

	# For each top novel term, find its closest COCO-80 category by CLIP
	# similarity, to see what it's "standing in for".
	coco80_list = sorted(COCO80_NAMES)
	novel_embeddings = embed_labels(novel_terms)
	coco80_embeddings = embed_labels(coco80_list)
	sim_matrix = cosine_similarity_matrix(novel_embeddings, coco80_embeddings)

	novel_term_analysis = []
	for i, (term, count) in enumerate(top_novel):
		best_idx = int(sim_matrix[i].argmax())
		best_match = coco80_list[best_idx]
		best_sim = float(sim_matrix[i][best_idx])
		novel_term_analysis.append({
			"term": term,
			"count": count,
			"closest_coco80_category": best_match,
			"similarity_to_closest": round(best_sim, 4),
		})

	report = {
		"total_native_labels": sum(native_label_counter.values()),
		"total_plural_of_native_labels": sum(plural_label_counter.values()),
		"total_out_of_vocabulary_labels": sum(oov_label_counter.values()),
		"unique_native_terms": len(native_label_counter),
		"unique_plural_of_native_terms": len(plural_label_counter),
		"unique_out_of_vocabulary_terms": len(oov_label_counter),
		"top_native_terms": native_label_counter.most_common(args.top_n),
		"top_plural_of_native_terms": plural_label_counter.most_common(args.top_n),
		"top_out_of_vocabulary_terms_with_closest_match": novel_term_analysis,
	}

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	with open(args.out, "w") as f:
		json.dump(report, f, indent=2)

	print(f"Native (exact COCO-80 match):      {report['total_native_labels']:>5} ({report['unique_native_terms']} unique)")
	print(f"Plural-of-native (grammar only):   {report['total_plural_of_native_labels']:>5} ({report['unique_plural_of_native_terms']} unique)")
	print(f"Out-of-vocabulary (genuinely new): {report['total_out_of_vocabulary_labels']:>5} ({report['unique_out_of_vocabulary_terms']} unique)")

	print(f"\nTop plural-of-native terms (pure grammatical mismatch, not semantic novelty):")
	for term, count in plural_label_counter.most_common(args.top_n):
		print(f"  {term:<25} (x{count:>3})  -> singular: {normalize_plural(term)}")

	print(f"\nTop {args.top_n} out-of-vocabulary terms and their closest COCO-80 category:")
	for entry in novel_term_analysis:
		print(f"  {entry['term']:<25} (x{entry['count']:>3})  -> {entry['closest_coco80_category']:<15} "
			  f"(sim={entry['similarity_to_closest']:.3f})")

	print(f"\nWritten to {args.out}")


if __name__ == "__main__":
	main()

