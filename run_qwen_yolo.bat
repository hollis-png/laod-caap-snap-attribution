@echo off
set PYTHONUTF8=1
cd /d C:\LAOD
call C:\Miniconda3\Scripts\activate.bat laod
python -u eval\run_qwen_yoloworld_detect.py > eval_out\qwen_yoloworld_full.log 2>&1

