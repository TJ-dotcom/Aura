@echo off
REM AURA Analysis Assistant - Uses Llama2 with RAG for in-depth analysis
if "%~1"=="" (
    echo Usage: aura-analyze "your analysis question or topic"
    echo Example: aura-analyze "Compare Python vs JavaScript for web development"
    exit /b 1
)

echo 🔍 ANALYSIS TASK - Using Llama2 with Knowledge Enhancement
echo ============================================================
python main.py --model-path "llama2" --temperature 0.4 --max-tokens 700 --rag --log-level WARNING %*
