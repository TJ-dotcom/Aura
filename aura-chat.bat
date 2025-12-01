@echo off
REM AURA Quick Chat - Uses TinyLlama for fast responses
if "%~1"=="" (
    echo Usage: aura-chat "your question or message"
    echo Example: aura-chat "How are you today?"
    exit /b 1
)

echo 💬 QUICK CHAT - Using TinyLlama (Fast Response)
echo ============================================================
python main.py --model-path "tinyllama" --temperature 0.7 --max-tokens 300 --log-level WARNING %*
