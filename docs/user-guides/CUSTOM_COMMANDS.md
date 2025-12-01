# 🚀 AURA Custom Commands Guide

## Task-Specific AI Assistants

Each command uses the optimal model for its specific task type:

### 🔧 Coding Tasks - `aura-code.bat`
**Model:** DeepSeek Coder 6.7B (Specialized for programming)  
**Settings:** Low temperature (0.2) for precise code, 600 tokens  

```bash
aura-code.bat "Write a Python function to implement binary search"
aura-code.bat "Create a REST API using Flask"
aura-code.bat "Debug this code: [paste your code]"
aura-code.bat "Explain how recursion works with examples"
```

### ✍️ Creative Writing - `aura-write.bat`
**Model:** Llama2 (General purpose, creative)  
**Settings:** High temperature (0.8) for creativity, 800 tokens  

```bash
aura-write.bat "Write a short story about time travel"
aura-write.bat "Create a poem about artificial intelligence"
aura-write.bat "Write a business proposal for a tech startup"
```

### 💬 Quick Chat - `aura-chat.bat`  
**Model:** TinyLlama (Fast, lightweight)  
**Settings:** Balanced temperature (0.7), 300 tokens for quick responses  

```bash
aura-chat.bat "How are you today?"
aura-chat.bat "What's the weather like for coding?"
aura-chat.bat "Quick question about Python syntax"
```

### 🔍 Analysis & Research - `aura-analyze.bat`
**Model:** Llama2 with RAG (Knowledge-enhanced)  
**Settings:** Low temperature (0.4) for accuracy, 700 tokens, document search enabled  

```bash
aura-analyze.bat "Compare Python vs JavaScript for web development"
aura-analyze.bat "What are the pros and cons of microservices?"
aura-analyze.bat "Analyze the trends in AI development"
```

## Alternative: Using main.py Directly

You can also use the main interface with custom settings:

```bash
# Coding with DeepSeek
python main.py --model-path "deepseek-coder:6.7b" --temperature 0.2 "Your coding question"

# Writing with Llama2
python main.py --model-path "llama2" --temperature 0.8 --max-tokens 800 "Your writing prompt"

# Quick chat with TinyLlama  
python main.py --model-path "tinyllama" --max-tokens 200 "Your quick question"

# Analysis with knowledge enhancement
python main.py --model-path "llama2" --rag --temperature 0.4 "Your analysis topic"
```

## Performance Comparison

| Model | Speed (TPS) | Quality | Best For | Resource Usage |
|-------|-------------|---------|----------|----------------|
| DeepSeek Coder 6.7B | ~1.5 | Excellent | Programming tasks | High |
| Llama2 | ~2-4 | High | General tasks, writing | Medium |  
| TinyLlama | ~9-12 | Good | Quick questions | Low |

## Tips for Best Results

1. **Be Specific:** More detailed prompts yield better results
2. **Use the Right Tool:** Match your task to the specialized command
3. **Model Loading:** First use of each model takes ~3 minutes to load
4. **Subsequent Runs:** Much faster once model is loaded in Ollama

## Available Models in Your System

- `tinyllama` - Fast, lightweight (637MB)
- `llama2` - Balanced performance (3.8GB)  
- `deepseek-coder:6.7b` - Coding specialist (3.8GB)

Your RTX 4060 Laptop GPU (8GB VRAM) is perfectly sized for these models!
