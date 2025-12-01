@echo off
REM AURA Writing Assistant - Uses Llama2 for creative writing tasks
if "%~1"=="" (
    echo Usage: aura-write "your writing task or prompt"
    echo Example: aura-write "Write a short story about robots"
    exit /b 1
)

echo ✍️ WRITING TASK - Using Llama2 Model
echo ============================================================
python main.py --model-path "llama2" --temperature 0.8 --max-tokens 800 --log-level WARNING %*
