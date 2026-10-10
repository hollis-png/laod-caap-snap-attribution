#!/bin/bash
cd ~/laod-exp
export TORCH_DISABLE_NATIVE_JIT=1
for r in 1 2; do
  .venv/bin/python eval/run_batch_eval.py --n 5000 --annotations ~/laod-caap-check/instances_val2017.json --images ~/laod-exp/coco/val2017 --predictions-out eval_out/repeat/pred_r$r.jsonl --report-out eval_out/repeat/report_r$r.json > eval_out/repeat/run_r$r.log 2>&1
done
echo DONE > eval_out/repeat/DONE
