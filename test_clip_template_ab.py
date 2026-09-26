import sys
sys.path.insert(0, ".")
from eval.snap import embed_labels, cosine_similarity_matrix

pairs = [
    ("cat", "cat", "identical"),
    ("cat", "kitten", "near-synonym"),
    ("cat", "dog", "related-animal"),
    ("cat", "sofa", "unrelated"),
    ("cat", "airplane", "unrelated"),
    ("airplane", "banana", "unrelated"),
    ("sedan", "car", "hypernym"),
    ("sedan", "coupe", "sibling-fine-grained"),
]

print("=== Bare word (no template) ===")
words = sorted(set([p[0] for p in pairs] + [p[1] for p in pairs]))
emb_bare = embed_labels(words)
idx = {w: i for i, w in enumerate(words)}
sim_bare = cosine_similarity_matrix(emb_bare, emb_bare)
for a, b, tag in pairs:
    print(f"  sim({a}, {b}) = {sim_bare[idx[a]][idx[b]]:.4f}  [{tag}]")

print("\n=== With 'a photo of a {label}' template ===")
templated = [f"a photo of a {w}" for w in words]
emb_tmpl = embed_labels(templated)
sim_tmpl = cosine_similarity_matrix(emb_tmpl, emb_tmpl)
for a, b, tag in pairs:
    print(f"  sim(photo of {a}, photo of {b}) = {sim_tmpl[idx[a]][idx[b]]:.4f}  [{tag}]")

