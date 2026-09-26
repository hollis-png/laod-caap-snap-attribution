@echo off
set PYTHONUTF8=1
cd /d C:\LAOD
call C:\Miniconda3\Scripts\activate.bat laod
python -u eval\run_qwen_llm_gen.py > eval_out\qwen_llm_gen_full.log 2>&1

