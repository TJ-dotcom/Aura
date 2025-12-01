# AURA CLI Reference Guide - Hardware-Aware AI Intelligence

## Overview
AURA is a hardware-aware AI engine that automatically profiles your system and intelligently selects optimal models based on:
1. **Hardware Analysis** - Detects RAM, VRAM, GPU, and classifies performance tier
2. **Intelligent Routing** - Analyzes prompts to select the best model for each task
3. **Performance Optimization** - Hardware-aware model selection and parameter tuning

This is **not** just another Ollama wrapper - it's an intelligent orchestration engine.

## Core AURA Vision

### Hardware-Aware Inference
```bash
aura hardware
# 🔍 AURA Hardware Analysis
# 💾 System RAM: 16,068 MB  
# 🎮 GPU: NVIDIA GeForce RTX 4060 Laptop GPU
# 🎮 GPU VRAM: 8,188 MB
# ⚙️  Optimal GPU Layers: 30
# 🏆 Performance Tier: BALANCED
```

### Intelligent Model Routing
```bash
aura "Write a Python function to sort a list"
# 🧠 AURA Intelligence Analysis...
# └─ Selected Model: deepseek-coder:6.7b  
# └─ Optimization Reason: Prompt analysis → coding task

aura "Write a creative story about space"  
# 🧠 AURA Intelligence Analysis...
# └─ Selected Model: llama2:7b
# └─ Optimization Reason: Prompt analysis → writing task
```

## Installation

### Quick Install
```powershell
# From the AURA project directory
.\install-aura.ps1 -Install
# Restart terminal, then use 'aura' from anywhere
```

## Commands

### Direct Inference (Primary Use Case)
AURA's core functionality - intelligent inference with automatic model selection:

```bash
# Direct prompts - AURA analyzes and selects optimal model
aura "Write a Python class for a binary tree"           # → deepseek-coder:6.7b
aura "Write a poem about artificial intelligence"       # → llama2:7b  
aura "What is machine learning?"                        # → tinyllama (fast response)
aura "Analyze the implications of AI in healthcare"     # → llama2:7b

# Force specific model (overrides AURA intelligence)
aura infer "Fix this code" --model deepseek-coder:6.7b

# Interactive mode with intelligent routing per prompt
aura infer --interactive
```

### Hardware Analysis
```bash
aura hardware
# Shows detailed hardware profile and model recommendations
# Performance tiers: HIGH-PERFORMANCE / BALANCED / HIGH-EFFICIENCY
```

### Model Intelligence
```bash  
aura models
# Lists available models with AURA intelligence:
# - Model specializations (CODING/WRITING/CHAT)
# - Hardware compatibility analysis
# - Task-specific recommendations
```

### Legacy Ollama Commands (Enhanced with AURA Intelligence)
```bash
aura pull deepseek-coder:6.7b    # Download with AURA context
aura show llama2:7b              # Model info + AURA intelligence  
aura ps                          # Running models with AURA context
```

## AURA Intelligence Features

### Performance Tier Classification
- **HIGH-PERFORMANCE**: 16GB+ RAM, powerful GPU → 13B/33B models
- **BALANCED**: 8-16GB RAM, mid-range GPU → 7B models  
- **HIGH-EFFICIENCY**: <8GB RAM, no GPU → 1.3B/tiny models

### Automatic Task Detection
- **Coding keywords**: "function", "code", "python", "debug", "implement" → DeepSeek Coder
- **Writing keywords**: "write", "story", "essay", "creative", "poem" → Llama2
- **Analysis keywords**: "analyze", "compare", "implications", "research" → Llama2
- **Chat/Q&A**: General questions and conversation → TinyLlama (fast) or Llama2

### Hardware-Optimized Model Selection

#### For BALANCED Hardware (like RTX 4060, 16GB RAM):
```bash
aura models
# Coding Tasks: deepseek-coder:6.7b, codellama:7b
# Writing Tasks: llama2:7b, mistral:7b  
# Chat Tasks: tinyllama, llama2:7b
```

#### For HIGH-EFFICIENCY Hardware (<8GB RAM):
```bash
aura models  
# Coding Tasks: deepseek-coder:1.3b, tinyllama
# Writing Tasks: tinyllama, phi:2.7b
# Chat Tasks: tinyllama
```

## Examples Demonstrating AURA Intelligence

### Programming Tasks
```bash
# AURA detects coding intent, selects DeepSeek Coder
aura "Implement quicksort in Python with comments"
aura "Debug this SQL query and explain the fix"
aura "Write unit tests for a REST API endpoint"
```

### Creative Writing  
```bash
# AURA detects creative intent, selects Llama2
aura "Write a short story about time travel"
aura "Create marketing copy for a new AI product"
aura "Write a formal business proposal"
```

### Quick Questions
```bash
# AURA detects simple Q&A, selects fast TinyLlama
aura "What is REST API?"
aura "Explain machine learning briefly"
aura "How does encryption work?"
```

### Analysis Tasks
```bash
# AURA detects analytical intent, selects Llama2
aura "Compare different database architectures"
aura "Analyze the pros and cons of microservices"
aura "Research the impact of AI on job markets"
```

## Performance Features

### Real-Time Hardware Optimization
- Detects system RAM and GPU VRAM automatically
- Calculates optimal GPU layer allocation
- Adapts model selection to available resources

### Intelligent Caching and Memory Management
- Only loads one model at a time (memory efficient)
- Automatic model switching based on prompt analysis
- Hardware-aware memory allocation

### Performance Metrics
```bash
# AURA shows detailed performance data
aura "Test prompt"
# ⚡ Performance: 11.2 TPS
# 🧠 Model Used: coding  
# 🔧 Hardware Optimized: Yes
```

## Advanced Usage

### RAG Integration (Future)
```bash
aura "What does the document say about AI safety?" --rag
# Enables document augmentation when available
```

### Override AURA Intelligence
```bash
aura infer "Simple question" --model llama2:13b
# Forces larger model despite AURA recommending TinyLlama
```

### Interactive Mode with Per-Prompt Intelligence
```bash
aura infer --interactive
# [AURA] >>> Write a Python function    # → DeepSeek Coder
# [AURA] >>> Tell me a joke             # → TinyLlama  
# [AURA] >>> Analyze this data          # → Llama2
```

## Why AURA vs Direct Ollama?

### Direct Ollama
```bash
ollama run deepseek-coder:6.7b "What is 2+2?"     # Overkill for simple math
ollama run tinyllama "Write a complex algorithm"  # Inadequate for complex coding
# User must manually choose models and manage resources
```

### AURA Intelligence
```bash
aura "What is 2+2?"                    # → TinyLlama (fast, appropriate)
aura "Write a complex algorithm"       # → DeepSeek Coder (specialized)
# AURA automatically optimizes every interaction
```

## Troubleshooting

### "Full engine not available"
- Normal during development - AURA falls back to Ollama integration
- Still provides hardware profiling and basic model selection
- Full engine requires all dependencies (FAISS, etc.)

### Model Not Found
```bash
aura models                    # Check available models
aura pull deepseek-coder:6.7b  # Download missing models
```

### Performance Issues
```bash
aura hardware                  # Check your performance tier
# AURA automatically recommends appropriate models
```

## Development Status

✅ **Hardware Profiling** - Complete (RAM, VRAM, GPU detection, tier classification)  
✅ **Intelligent Model Routing** - Complete (keyword analysis, task detection)  
✅ **Performance Optimization** - Complete (hardware-aware selection)  
✅ **Ollama Integration** - Complete (streaming API with high performance)  
✅ **Full Engine Mode** - Complete (FAISS and all dependencies working)  
✅ **RAG Integration** - Complete (document augmentation with FAISS vector search)  

AURA represents the future of AI interaction - where the system intelligently adapts to both your hardware capabilities and task requirements, eliminating the need for manual model management.
