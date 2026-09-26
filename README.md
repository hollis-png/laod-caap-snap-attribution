# laod-caap-snap-attribution

Code accompanying the paper *"Attribution Gaps in Zero-Training LLM+OVOD Pipelines: A Fine-Grained Analysis of the CAAP–SNAP Discrepancy"* (arXiv, 2026).

This repository does **not** re-implement or re-host [LAOD](https://github.com/furkanmumcu/LAOD) itself. It contains the batch-evaluation infrastructure, metric implementations, and analysis scripts we wrote on top of it — LAOD's own pipeline code (`laod.py`, `demo.py`, `draw_utils.py`) has an empty `eval/snap.py` and no batch-evaluation harness, so everything under `eval/` here is original.

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
| `phase2_vocab_deepdive.py` | Computes weighted CAAP-TP rate per semantic-relation category (synonym / finer_grained / noise) |

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

## Reproducing the paper's numbers

Each script under `eval/` takes `--annotations` and `--images` (or similar) CLI arguments pointing at your local COCO installation; run `python eval/<script>.py --help` for the exact flags. The overall pipeline order is:

1. `run_batch_eval.py` — generate predictions for the full corpus (Gemma-3 + YOLO-World)
2. `phase2_5000_analysis.py` — RQ1–RQ4 on the full corpus
3. `vocab_deepdive_extract_full.py` → `phase2_vocab_deepdive.py` — vocabulary deepdive
4. `sample_no_overlap_200.py` → manual classification (this step is inherently manual; the paper's category labels are not auto-generated)
5. `run_gdino_eval.py` and `run_qwen_llm_gen.py` + `run_qwen_yoloworld_detect.py` — robustness checks
6. `qwen_eval_compare.py` — final cross-configuration comparison

## Citation

If you use this code, please cite the paper (see the arXiv listing for the BibTeX entry) and the original [LAOD](https://arxiv.org/abs/2507.10844) work this analysis builds on.

## License

MIT — see `LICENSE`. This license covers only the code in this repository; it does not extend to the third-party LAOD repository or to the model weights / datasets referenced above, which retain their own licenses.
