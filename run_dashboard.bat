@echo off
rem Opens the dashboard in your browser. Creates the virtual environment on first use.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setting up the Python environment, this takes a minute...
    python -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || goto :error
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless false
goto :eof
:error
echo Setup failed. Make sure Python 3.12 or newer is installed and on PATH.
pause
