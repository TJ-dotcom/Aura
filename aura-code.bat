@echo off
REM AURA Coding Assistant - Uses DeepSeek Coder 6.7B for programming tasks
if "%~1"=="" (
    echo Usage: aura-code "your coding task or question"
    echo Example: aura-code "Write a Python function to sort a list"
    exit /b 1
)

echo 🔧 CODING TASK - Using DeepSeek Coder 6.7B Model
echo ============================================================
python main.py --model-path "deepseek-coder:6.7b" --temperature 0.2 --max-tokens 600 --log-level WARNING %*
