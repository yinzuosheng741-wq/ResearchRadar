@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONNOUSERSITE=1
if exist "runtime\python.exe" (
  "runtime\python.exe" desktop.py %*
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" desktop.py %*
) else (
  echo Python runtime missing. See docs/DESKTOP.md.
  pause
  exit /b 1
)
if errorlevel 1 pause
