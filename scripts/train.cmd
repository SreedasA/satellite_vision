@echo off
rem Run a training job on this PC: offline, independent of Claude, PC kept awake.
rem Usage (from the project folder, in any Command Prompt / PowerShell / Windows Terminal):
rem   scripts\train.cmd --dataset dfc23 --split block --epochs 30 --run-name p2_dfc23_block_30ep
rem Watch progress in a second window:  scripts\watch_training.cmd
rem Closing this window stops the training.
cd /d "%~dp0.."
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set MLFLOW_DISABLE_AGENT_HINT=1
if not exist "S:\sat2city-data\logs" mkdir "S:\sat2city-data\logs"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set TS=%%i
set LOG=S:\sat2city-data\logs\train_%TS%.log
echo.
echo  Sat2City3D training started %date% %time%
echo  Arguments : %*
echo  Log file  : %LOG%
echo  Progress  : run  scripts\watch_training.cmd  in another window
echo  Keep this window open; the PC is kept awake until training ends.
echo.
"S:\sat2city-data\venv\Scripts\python.exe" -m sat2city.train %* > "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo.
if %RC%==0 (echo  Training finished OK.) else (echo  Training FAILED with exit code %RC% - see the log above.)
echo  Log: %LOG%
