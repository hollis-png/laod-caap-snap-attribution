import sys
sys.path.insert(0, ".")
from eval.snap import compute_snap, embed_labels, cosine_similarity_matrix

# Case 1: exact match, should be near-perfect SNAP
preds_exact = [[["cat", 0.9], ["sofa", 0.8]]]
gts_exact = [[["cat"], ["sofa"]]]
snap_exact = compute_snap(preds_exact, gts_exact)
print(f"Case 1 (exact match): SNAP = {snap_exact:.4f} (expect near 1.0)")

# Case 2: semantically close but not identical (kitten vs cat), mid-range similarity
preds_close = [[["kitten", 0.9]]]
gts_close = [[["cat"]]]
snap_close = compute_snap(preds_close, gts_close)
print(f"Case 2 (kitten vs cat): SNAP = {snap_close:.4f} (expect moderate, between 0 and exact-match case)")

# Case 3: completely unrelated words, should be near 0
preds_unrelated = [[["airplane", 0.9]]]
gts_unrelated = [[["banana"]]]
snap_unrelated = compute_snap(preds_unrelated, gts_unrelated)
print(f"Case 3 (airplane vs banana): SNAP = {snap_unrelated:.4f} (expect near 0)")

# Case 4: raw cosine similarity sanity check
emb = embed_labels(["cat", "kitten", "dog", "airplane"])
sim = cosine_similarity_matrix(emb, emb)
print("\nCase 4: raw pairwise cosine similarities")
labels = ["cat", "kitten", "dog", "airplane"]
for i, li in enumerate(labels):
    for j, lj in enumerate(labels):
        if i < j:
            print(f"  sim({li}, {lj}) = {sim[i][j]:.4f}")

# Case 5: multi-image, multi-object aggregation with one miss (over-generation penalty)
preds_multi = [
    [["cat", 0.9], ["sofa", 0.8], ["elephant", 0.7]],  # elephant is a false positive
    [["remote", 0.95]],
]
gts_multi = [
    [["cat"], ["sofa"]],
    [["remote"]],
]
snap_multi = compute_snap(preds_multi, gts_multi)
print(f"\nCase 5 (multi-image with 1 FP over-generation): SNAP = {snap_multi:.4f} (expect penalized below Case 1)")

