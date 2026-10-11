"""E20: SNAP on our 5,000-image run under the text encoder and controls of the LAOD v2 repository (2026-10-06).

LAOD v2 replaced CLIP by Sentence-BERT (all-MiniLM-L6-v2, bare labels) and scores every cell next to a label-shuffled twin, because CLIP puts
unrelated COCO labels at a median cosine of 0.786. Here the same detections and ground-truth labels are scored with
  clip   openai/clip-vit-base-patch32, raw text embeddings (eval/snap.py embed_labels; the encoder of the paper)
  minilm sentence-transformers/all-MiniLM-L6-v2 re-implemented with transformers: mean pooling over tokens, then L2 normalisation (what the model does)
each with the true predicted labels and with the predicted labels shuffled across all detections of the corpus (seed 20261011).
Matching is eval/snap.compute_ap_single_image_semantic (per image, by confidence, one-to-one, no IoU); the corpus AP is the all-point AP as in compute_snap.
Thresholds: 0.50-0.95 step 0.05 (LAOD v2 disjoint grid: LO 0.50-0.60, MI 0.65-0.80, HI 0.85-0.95), and the four of our paper (0.6, 0.7, 0.8, 0.9).
Usage (NDHU, cwd = the folder that contains eval/): python e20_snap_sbert.py <predictions.jsonl> <instances_val2017.json> <out.json>
"""
import json, sys
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, ".")
from eval.snap import embed_labels, compute_ap_single_image_semantic

pred_p, coco_p, out_p = sys.argv[1:4]
SEED = 20261011
coco = json.load(open(coco_p)); cat = {c["id"]: c["name"] for c in coco["categories"]}
gt = {}
for a in coco["annotations"]:
    gt.setdefault(a["image_id"], []).append(cat[a["category_id"]])
preds = {}
for l in open(pred_p):
    if l.strip():
        r = json.loads(l); preds[r["image_id"]] = [(d["label"], d["confidence"]) for d in r["detections"]]
ids = sorted(preds)
all_labels = sorted({p[0] for i in ids for p in preds[i]} | {g for i in ids for g in gt.get(i, [])})

_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
_mod = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()

def embed_minilm(labels):
    out = []
    for i in range(0, len(labels), 256):
        b = _tok(labels[i:i + 256], padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            h = _mod(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.concatenate(out)

ENCODERS = {"clip": lambda L: embed_labels(L), "minilm": embed_minilm}
THRESH = [round(float(t), 2) for t in np.arange(0.50, 0.951, 0.05)]

def corpus_snap(tau, pred_lab, emb):
    """pred_lab: image_id -> list of (label, conf); emb: label -> vector. All-point AP over the corpus."""
    pairs = []; n_gt = 0
    for i in ids:
        p = pred_lab[i]; g = gt.get(i, []); n_gt += len(g)
        if not p: continue
        pe = np.stack([emb[x[0]] for x in p]); ge = np.stack([emb[x] for x in g]) if g else np.zeros((0, pe.shape[1]))
        tp, fp, _ = compute_ap_single_image_semantic([[x[0], x[1]] for x in p], [[x] for x in g], tau, pred_embeddings=pe, gt_embeddings=ge)
        for (_, c), t in zip(sorted(p, key=lambda x: -x[1]), tp):
            pairs.append((c, int(t)))
    pairs.sort(key=lambda x: -x[0]); t = np.cumsum([x[1] for x in pairs]); f = np.cumsum([1 - x[1] for x in pairs])
    prec = t / (t + f); rec = t / n_gt; ip = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(ip * np.diff(np.concatenate(([0.0], rec)))))

rng = np.random.default_rng(SEED)
flat = [(i, k) for i in ids for k in range(len(preds[i]))]
perm = rng.permutation(len(flat)); labs = [preds[i][k][0] for i, k in flat]
shuffled = {i: list(preds[i]) for i in ids}
for (i, k), j in zip(flat, perm):
    shuffled[i][k] = (labs[j], preds[i][k][1])

res = {"n_images": len(ids), "n_detections": len(flat), "n_gt": sum(len(gt.get(i, [])) for i in ids), "thresholds": THRESH, "cells": {}}
for name, fn in ENCODERS.items():
    E = fn(all_labels); emb = {l: E[k] for k, l in enumerate(all_labels)}
    gtl = sorted({g for i in ids for g in gt.get(i, [])}); G = np.stack([emb[g] for g in gtl])
    S = G @ G.T; off = S[~np.eye(len(gtl), dtype=bool)]
    res["cells"][name] = {"unrelated_pair_cosine": {"n_pairs": int(off.size), "median": float(np.median(off)), "min": float(off.min()),
                          "admitted_at": {str(t): float((off >= t).mean()) for t in (0.5, 0.6, 0.7, 0.8, 0.9)}}}
    for tag, pl in (("real", preds), ("shuffled", shuffled)):
        per = {str(t): corpus_snap(t, pl, emb) for t in THRESH}
        mean = lambda ts: float(np.mean([per[str(t)] for t in ts]))
        res["cells"][name][tag] = dict(per_threshold=per, LO=mean((0.5, 0.55, 0.6)), MI=mean((0.65, 0.7, 0.75, 0.8)), HI=mean((0.85, 0.9, 0.95)),
                                       ten_threshold_mean=mean(THRESH), paper_four_mean=mean((0.6, 0.7, 0.8, 0.9)))
        print(name, tag, {k: round(v, 4) for k, v in res["cells"][name][tag].items() if k != "per_threshold"}, flush=True)
    print(name, "unrelated-pair cosine", res["cells"][name]["unrelated_pair_cosine"], flush=True)
json.dump(res, open(out_p, "w"), indent=1)
