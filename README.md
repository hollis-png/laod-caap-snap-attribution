# laod-caap-snap-attribution

[![arXiv](https://img.shields.io/badge/arXiv-2609.32567-b31b1b.svg)](https://arxiv.org/abs/2609.32567)

Code accompanying the paper *"Attribution Gaps in Zero-Training LLM+OVOD Pipelines: A Fine-Grained Analysis of the CAAP–SNAP Discrepancy"* ([arXiv:2609.32567](https://arxiv.org/abs/2609.32567), 2026).

This repository does **not** re-implement or re-host [LAOD](https://github.com/furkanmumcu/LAOD) itself. It contains the batch-evaluation infrastructure, metric implementations, and analysis scripts we wrote on top of it — In the LAOD version we built on (last commit 2025-07-30) the pipeline code (`laod.py`, `demo.py`, `draw_utils.py`) had an empty `eval/snap.py` and no batch-evaluation harness, so everything under `eval/` here is original. LAOD was rewritten on 2026-10-06 and now ships its own evaluation protocol (Sentence-BERT SNAP, maxDets = 100, disjoint CAAP intervals); this repository does not track that version and its numbers are not directly comparable with it.

## Correction notice (2026-10)

`eval/caap.py` as first released (and as used for arXiv v1) contained a bug: the corpus-level `compute_ca_ap` re-matched the pooled detections against the ground-truth boxes of **all** images, so a detection could match a box in a different image, increasingly so for larger corpora. This inflated corpus-level CAAP (0.38 on the 5,000-image split; the corrected value is **0.21**, per-IoU-interval LO/MI/HI = 0.268 / 0.231 / 0.113). Single-image calls and all detection-level analyses (`phase2_*`, vocabulary and audit scripts) were not affected. The current `eval/caap.py` matches per image and is covered by `tests/test_caap.py` (run `pytest tests`). Anything that cites a corpus CAAP from arXiv v1 or from an older log should be recomputed. The `selfreview_analysis/` directory holds the scripts of the follow-up analysis that found and quantified this and the other corrections. The corrected numbers and the revised analysis are in the updated version of the paper (arXiv v2).

A second, smaller fix: `eval/snap.py` computed label embeddings in the caller's order but matched predictions in confidence order, so embeddings and predictions were misaligned whenever the input was not already sorted by confidence. The prediction files used in the paper were already confidence-sorted, so the reported SNAP numbers are unchanged; the fix is covered by `tests/test_snap_order_and_ap.py`.

## Setup

1. Clone the original LAOD repository and place this repo's `eval/` directory inside it (or add it to `PYTHONPATH`):
   ```bash
   git clone https://github.com/furkanmumcu/LAOD.git
   cp -r eval LAOD/eval
   cd LAOD
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download COCO val2017 images and `instances_val2017.json`, and place them under a `datasets/coco/` directory (or pass `--images`/`--annotations` paths explicitly — every script that touches COCO takes these as CLI arguments).
4. Model weights: `google/gemma-3-4b-it` (gated, requires HuggingFace access approval), `yolov8x-worldv2.pt` (YOLO-World X), `IDEA-Research/grounding-dino-tiny`, `Qwen/Qwen2.5-VL-7B-Instruct`, and a CLIP checkpoint (`openai/clip-vit-base-patch32` by default) all download automatically via `transformers`/`ultralytics` on first run.

## What's here

### Core infrastructure (`eval/`)

| File | Purpose |
|---|---|
| `coco_loader.py` | COCO `instances_val2017.json` parser with deterministic subset sampling |
| `laod_infer.py` | Structured batch-inference wrapper around LAOD's `laod_yolo()` (returns boxes/labels/confidence instead of demo drawing code) |
| `caap.py` | Class-agnostic average precision (vectorized IoU matching across multiple thresholds) |
| `snap.py` | CLIP-based semantic naming accuracy — LAOD ships this file empty; this is our from-scratch implementation, matched to the original paper's description |
| `run_batch_eval.py` | Main batch-evaluation driver: resumable JSON cache, corpus-level CAAP/SNAP, per-image correlation |

### Paper §3.2 — RQ1–RQ4 main analysis

| File | Purpose |
|---|---|
| `phase2_stage2a.py` | Detection-level TP/FP labeling on the 200-image pilot; defines `vocabulary_category()` / `normalize_plural()` / `COCO80_NAMES` reused by later scripts |
| `phase2_5000_analysis.py` | Read-only re-analysis of RQ1–RQ4 on the full 5,000-image corpus (logic mirrors `phase2_stage2a.py` exactly, writes to a separate output file) |
| `phase2_rq3.py` | RQ3 failure-mode classification (near-miss / no-overlap / misplaced) |

### Paper §3.3 — Vocabulary deepdive

| File | Purpose |
|---|---|
| `vocab_deepdive_extract.py` | Extracts the 150 most frequent out-of-vocabulary labels and their best-IoU-matched ground-truth category |
| `vocab_deepdive_extract_full.py` | Extends extraction to all 829 unique out-of-vocabulary labels (99.3% coverage) |
| `phase2_vocab_deepdive.py` | Computes weighted CAAP-TP rate per semantic-relation category (synonym / finer_grained / out_of_ontology; the v1 label "noise" was renamed because it suggested the labels were wrong rather than outside COCO's 80 classes) |

### Paper §3.4 — Manual "no overlap" inspection

| File | Purpose |
|---|---|
| `sample_no_overlap_cases.py` | Deterministic n=60 sample from the no-overlap subset |
| `sample_no_overlap_200.py` | Extends the sample to n=200 (verified strict superset of the n=60 sample) |
| `render_no_overlap_cases.py` / `render_no_overlap_140.py` | Renders predicted box (red) against all COCO ground-truth boxes (green) for manual classification |

### Paper §3.5 — SNAP checkpoint ablation

| File | Purpose |
|---|---|
| `snap_checkpoint_compare.py` | Re-runs SNAP with three different CLIP checkpoints to rule out checkpoint choice as the cause of the SNAP reproduction gap |

### Paper §3.6 — Robustness checks (detector backbone / LLM swap)

| File | Purpose |
|---|---|
| `run_gdino_eval.py`, `gdino_rq2_eval.py` | Grounding DINO detector swap, reusing the same LLM-generated labels as the YOLO-World run |
| `verify_gdino.py`, `independent_verify.py` | From-scratch re-implementations used to independently verify the Grounding DINO / main-analysis numbers |
| `run_qwen_llm_gen.py` | Qwen2.5-VL 7B label generation (LLM swap, same prompt as Gemma-3) |
| `run_qwen_yoloworld_detect.py` | YOLO-World detection stage on Qwen-generated labels |
| `qwen_eval_compare.py` | Qwen vs. Gemma-3 comparison — computes CAAP-TP split for both under a consistent denominator convention (see the paper's Appendix A.4 for why this matters) |
| `qwen_smoke_test.py`, `qwen_timing_test.py` | Small-scale sanity/timing checks before full-corpus Qwen runs |

### Environment sanity checks (not part of the formal analysis)

`check_coco_fields.py`, `test_clip_template_ab.py`, `test_snap_sanity.py`, `test_yolo_device.py`, `test_yolo_device2.py`, `test_yolo_win.py` — one-off scripts used while setting up the pipeline (verifying COCO annotation fields, comparing CLIP prompt templates, checking YOLO-World device placement under CUDA). Kept for transparency; not required to reproduce the paper's results.

### Batch launch scripts (`.bat`)

Windows batch scripts used to launch long-running full-corpus evaluations (`run_full_5000.bat`, `run_validation.bat`, `run_validation_v5.bat`, `run_qwen_eval.bat`, `run_qwen_full.bat`, `run_qwen_yolo.bat`). These assume a `conda` environment named `laod` and reference paths relative to wherever you clone this repo alongside LAOD — adjust paths at the top of each script for your own setup.

### Follow-up analysis (`selfreview_analysis/`)

Scripts written after a self-review of the paper (all read stored predictions or labels; none changes `eval/`). Several run on a separate GPU machine and take absolute paths at the top; they are included for transparency and need adapting to run elsewhere.

| File | Purpose |
|---|---|
| `attribution_recompute.py`, `rq1_size_recheck.py`, `caap_check.py` | Re-derivation of the RQ1 size analysis (selection artifact), the exact-match failure decomposition, and the CAAP matching-bug check |
| `e1_lvis_rescore.py`, `e2_corem_rescore.py` | Re-score the detections against LVIS v1 and COCO-ReM annotations |
| `e3_*.py`, `e4_gdino_caap.py` | Regenerate Qwen2.5-VL labels and detections; corrected CAAP for Grounding DINO |
| `e6_soft_ap.py` | Exploratory AP with soft validity labels (optimistic, see the script header) |
| `e7_caap_other_gt.py` | Corpus CAAP against COCO-ReM, LVIS and their union |
| `e8_stuff_exclusion.py` | Share of valid-but-unannotated detections with stuff labels set aside |
| `e9_duplicates_by_vocab.py` | Duplicate detections by vocabulary category |
| `e10_hi_interval.py` | Which AP implementation choices move the HI-interval CAAP |
| `e11_pilot_subset.py` | Recompute the 200-image pilot with the full run's predictions |
| `e13_prompt_variants.py` | Prompt-variant robustness run |
| `e14_size_by_vocab.py` | Box-size distribution by vocabulary category |
| `e15_second_annotator.py` | Agreement of the second annotator's labels with the first pass |
| `e16_eq7_caap.py` | CAAP under the alternative reading (Eq. 7) of LAOD's interval definition |
| `e17_cluster_bootstrap.py` | Image-clustered bootstrap intervals for the headline rates |
| `e19_laod_legacy_grid.py` | Our per-IoU CAAP summarised over the threshold sets that LAOD's code produced for its paper (HI includes the unreachable IoU 1.00) |
| `e18_repeat.sh`, `e18_repeat_summary.py` | Repeat runs of the Gemma label generation (sampling variance of CAAP and SNAP) |
| `compute_agreement_hh.py` | Cohen's kappa between two human annotators |
| `judge/` | LLM-judge runs for the vocabulary contrast, including the text-only variant (`run_textonly.py`); needs an OpenRouter key in the environment of the `llm` CLI, never in the repo |

## Reproducing the paper's numbers

Each script under `eval/` takes `--annotations` and `--images` (or similar) CLI arguments pointing at your local COCO installation; run `python eval/<script>.py --help` for the exact flags. The overall pipeline order is:

1. `run_batch_eval.py` — generate predictions for the full corpus (Gemma-3 + YOLO-World)
2. `phase2_5000_analysis.py` — RQ1–RQ4 on the full corpus
3. `vocab_deepdive_extract_full.py` → `phase2_vocab_deepdive.py` — vocabulary deepdive
4. `sample_no_overlap_200.py` → manual classification (this step is inherently manual; the paper's category labels are not auto-generated)
5. `run_gdino_eval.py` and `run_qwen_llm_gen.py` + `run_qwen_yoloworld_detect.py` — robustness checks
6. `qwen_eval_compare.py` — final cross-configuration comparison

## Citation

If you use this code, please cite the paper and the original [LAOD](https://arxiv.org/abs/2507.10844) work this analysis builds on.

```bibtex
@article{yen2026attribution,
  title   = {Attribution Gaps in Zero-Training LLM+OVOD Pipelines: A Fine-Grained Analysis of the CAAP--SNAP Discrepancy},
  author  = {Yen, Yu-Feng},
  journal = {arXiv preprint arXiv:2609.32567},
  year    = {2026}
}
```

## Data and third-party licences

**Not included.** No COCO images are redistributed; download them yourself from the COCO site. Model weights are not included either.

**What the pipeline depends on, and under which terms** (check the linked pages yourself before reuse; this list is a pointer, not legal advice):

| Component | Notes |
|---|---|
| COCO val2017 annotations | CC BY 4.0. The images carry their original Flickr licences and are not redistributed here. |
| `google/gemma-3-4b-it` | Gated on Hugging Face; use is governed by the Gemma Terms of Use. |
| YOLO-World X (`yolov8x-worldv2.pt`, run through `ultralytics`) | YOLO-World is released under GPL-3.0, and `ultralytics` is under its own copyleft licence (AGPL-3.0 at the time of writing; verify). This repo only calls them and does not copy their code; if you redistribute a combined work, check the terms. |
| `Qwen/Qwen2.5-VL-7B-Instruct`, `IDEA-Research/grounding-dino-tiny` | See their model cards. |
| CLIP `openai/clip-vit-base-patch32` | OpenAI's CLIP code is MIT-licensed. The Hugging Face weights have no licence field; the model card limits use to research and calls deployed use cases out of scope. We use CLIP only as a text encoder for the SNAP score. |
| LVIS v1, COCO-ReM | Used only for the re-scoring scripts `e1`/`e2`/`e7`; see their own terms. |
| LLM judges (DeepSeek-V4-Flash, Mistral Large 4 via OpenRouter) | Only label strings are sent, never images. API keys must come from the environment, never from this repo. |

**Annotations.** The human labels from the audits are described in the paper (Appendix B lists what is released). They were produced by the author and by anonymous annotators who did not see the AI labels.

## License

MIT — see `LICENSE`. This license covers only the code in this repository; it does not extend to the third-party LAOD repository or to the model weights / datasets referenced above, which retain their own licenses.
