@echo off
REM AURA Command Line Interface - Windows Batch Wrapper
REM This makes 'aura' available as a system command

set "AURA_ROOT=%~dp0"
set "VENV_ACTIVATE=%AURA_ROOT%.venv\Scripts\activate.bat"
set "PYTHON_PATH=%AURA_ROOT%.venv\Scripts\python.exe"

REM Check if virtual environment exists
if not exist "%PYTHON_PATH%" (
    echo Error: Virtual environment not found at %AURA_ROOT%.venv
    echo Please run: python -m venv .venv
    echo Then: .venv\Scripts\activate
    echo And: pip install -r requirements.txt
    exit /b 1
)

REM Activate virtual environment and run AURA CLI
call "%VENV_ACTIVATE%" >nul 2>&1
"%PYTHON_PATH%" "%AURA_ROOT%aura.py" %*
