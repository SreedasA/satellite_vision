@echo off
rem Live progress bar for the latest (or a named) training run.
rem Usage:  scripts\watch_training.cmd            (latest run)
rem         scripts\watch_training.cmd RUN_NAME   (a specific run)
cd /d "%~dp0.."
set PYTHONUTF8=1
if "%~1"=="" (
  "S:\sat2city-data\venv\Scripts\python.exe" -m sat2city.progress
) else (
  "S:\sat2city-data\venv\Scripts\python.exe" -m sat2city.progress --run %1
)
