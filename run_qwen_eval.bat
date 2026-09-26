@echo off
set PYTHONUTF8=1
cd /d C:\LAOD
call C:\Miniconda3\Scripts\activate.bat laod
python -u eval\qwen_eval_compare.py --annotations ./datasets/coco/annotations/instances_val2017.json --images ./datasets/coco/val2017 --qwen-predictions eval_out\qwen_yoloworld_predictions.jsonl --gemma-predictions eval_out\coco_val_subset_predictions.jsonl --out eval_out\qwen_rq2_report.json > eval_out\qwen_eval_compare.log 2>&1

