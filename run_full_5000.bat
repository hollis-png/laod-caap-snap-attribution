@echo off
cd /d C:\LAOD
set PYTHONUTF8=1
"C:\miniconda3\envs\laod\python.exe" -u eval\run_qwen_llm_gen.py > eval_out\qwen_full5000_log.txt 2> eval_out\qwen_full5000_err.txt

