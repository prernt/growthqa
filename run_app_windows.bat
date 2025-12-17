@echo off
setlocal

set ROOT=%~dp0
cd /d "%ROOT%"

set VENV=.venv_infer
set LOCK=classifier_output\saved_models_selected\requirements_lock.txt

if not exist "%LOCK%" (
  echo [ERROR] Missing lock file: %LOCK%
  exit /b 1
)

if not exist "%VENV%\Scripts\python.exe" (
  echo Creating inference venv: %VENV%
  py -3.11 -m venv "%VENV%"
  "%VENV%\Scripts\python.exe" -m pip install --upgrade pip
  "%VENV%\Scripts\python.exe" -m pip install -r "%LOCK%"
)

"%VENV%\Scripts\python.exe" -m streamlit run app\streamlit_app.py
