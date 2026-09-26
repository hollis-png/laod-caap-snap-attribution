@echo off
cd /d C:\LAOD
set PYTHONUTF8=1
"C:\miniconda3\envs\laod\python.exe" -u eval\run_qwen_llm_gen.py --limit 30 > eval_out\qwen_fix_validation_log.txt 2> eval_out\qwen_fix_validation_err.txt

