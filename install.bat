@echo off
REM AURA One-Stop Installation Script for Windows
REM Simple batch file alternative to install.ps1

setlocal EnableDelayedExpansion

echo.
echo ╔══════════════════════════════════════════╗
echo ║           AURA AI Engine Setup           ║
echo ║     Hardware-Aware AI Intelligence       ║
echo ╚══════════════════════════════════════════╝
echo.

REM Check if we're in the right directory
if not exist "aura.py" (
    echo ❌ Error: aura.py not found
    echo Please run this script from the AURA project root directory
    pause
    exit /b 1
)

REM Check Python
echo 🔍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python !PYTHON_VERSION! found

REM Check Ollama
echo 🔍 Checking Ollama installation...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama not found
    echo Please install Ollama from https://ollama.com
    echo Then re-run this script
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('ollama --version 2^>^&1') do set OLLAMA_VERSION=%%i
echo ✅ Ollama found: !OLLAMA_VERSION!

REM Create virtual environment
echo 🐍 Setting up Python virtual environment...
if exist ".venv" (
    echo Virtual environment already exists, using existing one...
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

REM Activate virtual environment and install dependencies
echo 📦 Installing Python dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
python -m pip install psutil numpy requests colorama pytest >nul 2>&1

REM Try to install FAISS (optional)
echo 🔄 Attempting to install FAISS (optional)...
python -m pip install faiss-cpu >nul 2>&1
if errorlevel 1 (
    echo ⚠️  FAISS installation failed (will use fallback mode)
) else (
    echo ✅ FAISS installed successfully
)

echo ✅ Dependencies installed

REM Analyze hardware
echo 🔍 Analyzing your hardware...
python aura.py hardware >temp_hw.txt 2>&1
for /f "tokens=*" %%i in ('findstr "Performance Tier" temp_hw.txt') do set HW_LINE=%%i
for /f "tokens=3" %%i in ("!HW_LINE!") do set HW_TIER=%%i
del temp_hw.txt >nul 2>&1

if "!HW_TIER!"=="" set HW_TIER=BALANCED
echo 🏆 Hardware Tier: !HW_TIER!

REM Install recommended models
echo 🤖 Installing recommended models...
set /p INSTALL_MODELS=Download AI models now? This may take several minutes (Y/n): 
if "!INSTALL_MODELS!"=="" set INSTALL_MODELS=Y
if /i "!INSTALL_MODELS!"=="Y" (
    echo 📥 Downloading TinyLlama (essential, fast model)...
    ollama pull tinyllama:latest
    
    if /i "!HW_TIER!"=="HIGH-EFFICIENCY" (
        echo 📥 Downloading DeepSeek Coder 1.3B (efficient coding model)...
        ollama pull deepseek-coder:1.3b
    ) else (
        echo 📥 Downloading Llama2 7B (general purpose model)...
        ollama pull llama2:7b
        echo 📥 Downloading DeepSeek Coder 6.7B (coding specialist)...
        ollama pull deepseek-coder:6.7b
    )
    echo ✅ Model downloads completed
) else (
    echo ⏭️  Skipping model downloads
)

REM Install AURA CLI
echo 🔧 Installing AURA CLI to system PATH...
powershell -ExecutionPolicy Bypass -File install-aura.ps1 -Install >nul 2>&1
if errorlevel 1 (
    echo ⚠️  CLI installation had issues, but you can use: python aura.py
) else (
    echo ✅ AURA CLI installed
)

REM Test installation
echo 🧪 Testing installation...
python aura.py hardware >temp_test.txt 2>&1
findstr "AURA Hardware Analysis" temp_test.txt >nul
if errorlevel 1 (
    echo ⚠️  Installation test had issues
) else (
    echo ✅ Installation test passed
)
del temp_test.txt >nul 2>&1

REM Success message
echo.
echo ╔══════════════════════════════════════════╗
echo ║          🎉 INSTALLATION COMPLETE        ║
echo ╚══════════════════════════════════════════╝
echo.
echo 🚀 AURA is now ready to use!
echo.
echo 💡 Quick Start:
echo    aura "Write a Python function"
echo    aura "Tell me about AI"
echo    aura hardware
echo.
echo 📚 Documentation: CLI_GUIDE.md
echo 🆘 Help: python aura.py --help
echo.
echo 🌟 Welcome to the future of AI inference!
echo.

pause
