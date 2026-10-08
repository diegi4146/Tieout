@echo off
rem Rebuilds every table from SEC data. Cached downloads are reused;
rem add --refresh to pull fresh filings, e.g.  run_pipeline.bat --refresh
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setting up the Python environment, this takes a minute...
    python -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || goto :error
)
".venv\Scripts\python.exe" -m secanomaly run %*
pause
goto :eof
:error
echo Setup failed. Make sure Python 3.12 or newer is installed and on PATH.
pause
