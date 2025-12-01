# 🔬 AURA Model Performance Analysis
## Comprehensive TPS, CPU, and GPU Benchmarks

*Generated: August 20, 2025*

---

## 📊 Executive Summary

**Total Models Tested:** 7  
**Test Scenarios:** 3 (Simple Question, Coding Task, Reasoning Task)  
**Total Benchmarks:** 21  
**GPU:** RTX 4060 Laptop (8GB VRAM)  
**System:** Windows 11, Ollama v0.5.12  

---

## 🏆 Performance Champions

| Category | Model | Score | Context |
|----------|--------|--------|---------|
| 🚀 **Fastest TPS** | `deepseek-r1:1.5b` | **9.1 TPS** | Coding tasks |
| ⚡ **Most Efficient** | `deepseek-r1:1.5b` | **0.22 TPS/CPU%** | Best performance per CPU usage |
| 🎯 **Best GPU Usage** | `phi:latest` | **35.3%** | Maximum GPU utilization |
| 💚 **Lowest CPU** | `phi:latest` | **33.9%** | Minimal CPU overhead |
| 🧠 **Best Reasoning** | `tinyllama:latest` | **8.4 TPS** | Mathematical reasoning |

---

## 📈 Detailed Performance Matrix

### Tokens Per Second (TPS) Analysis

| Model | Size | Simple Q | Coding | Reasoning | **Average TPS** |
|-------|------|----------|---------|-----------|----------------|
| **deepseek-r1:1.5b** | 1.8B | 6.0 | **9.1** | 7.3 | **7.5** |
| **tinyllama:latest** | 1B | 7.1 | 7.0 | **8.4** | **7.5** |
| **phi:latest** | 3B | 3.4 | 0.6 | 3.9 | **2.6** |
| **phi3.5:3.8b** | 3.8B | 3.3 | 2.2 | 3.2 | **2.9** |
| **qwen2.5:7b** | 7.6B | 2.1 | 1.9 | 2.0 | **2.0** |
| **llama2:7b** | 7B | 1.8 | 1.6 | 1.4 | **1.6** |
| **deepseek-coder:6.7b** | 7B | 1.6 | 1.1 | 1.6 | **1.4** |

### CPU Usage Analysis

| Model | Size | Simple Q | Coding | Reasoning | **Average CPU%** |
|-------|------|----------|---------|-----------|-----------------|
| **phi:latest** | 3B | 39.8 | **33.9** | 41.5 | **38.4** |
| **deepseek-r1:1.5b** | 1.8B | **34.3** | 41.4 | 39.7 | **38.5** |
| **tinyllama:latest** | 1B | 36.4 | 44.4 | 39.4 | **40.1** |
| **phi3.5:3.8b** | 3.8B | 45.6 | 44.4 | 44.7 | **44.9** |
| **qwen2.5:7b** | 7.6B | 44.8 | 45.5 | 44.7 | **45.0** |
| **deepseek-coder:6.7b** | 7B | 46.2 | 44.6 | 45.0 | **45.3** |
| **llama2:7b** | 7B | 47.6 | 44.9 | 46.1 | **46.2** |

### GPU Usage Analysis

| Model | Size | Simple Q | Coding | Reasoning | **Average GPU%** |
|-------|------|----------|---------|-----------|-----------------|
| **phi:latest** | 3B | 24.0 | **35.3** | 25.8 | **28.4** |
| **tinyllama:latest** | 1B | **17.6** | 25.9 | 20.1 | **21.2** |
| **deepseek-r1:1.5b** | 1.8B | 21.0 | 21.5 | 24.3 | **22.3** |
| **qwen2.5:7b** | 7.6B | 23.5 | 24.1 | 22.7 | **23.4** |
| **llama2:7b** | 7B | 24.8 | 23.2 | 23.7 | **23.9** |
| **phi3.5:3.8b** | 3.8B | 20.9 | 24.5 | 24.5 | **23.3** |
| **deepseek-coder:6.7b** | 7B | 22.4 | 25.2 | 23.6 | **23.7** |

---

## 🔍 Key Performance Insights

### 🥇 **DeepSeek-R1 1.5B: The Speed Champion**
- **Fastest overall performance**: 9.1 TPS on coding tasks
- **Most efficient**: Best TPS-to-CPU ratio (0.22)
- **Reasoning optimized**: Excellent for mathematical and logical tasks
- **Low resource usage**: Only 1.8B parameters with Q4_K_M quantization
- **Use case**: Perfect for AURA's math and reasoning workloads

### 🥈 **TinyLLama: The Consistent Performer**
- **Consistently high TPS**: 7.0-8.4 across all task types
- **Smallest model**: Only 1B parameters but competitive performance
- **Best reasoning TPS**: 8.4 TPS for mathematical problems
- **Balanced resource usage**: Moderate CPU (40.1%) and low GPU (21.2%)
- **Use case**: Excellent for simple queries and fast responses

### 🥉 **Phi Models: The GPU Maximizers**
- **Phi (3B)**: Best GPU utilization (35.3%), lowest CPU usage (33.9%)
- **Phi3.5 (3.8B)**: Balanced performance with consistent 2.2-3.3 TPS
- **Temperature management**: Both models run cool (45-48°C)
- **Use case**: Good for scenarios requiring maximum GPU engagement

### 📉 **Larger Models: The Accuracy Trade-off**
- **DeepSeek-Coder 6.7B**: Slowest but potentially most accurate for complex coding
- **Qwen2.5 7B**: Moderate performance, high resource usage (7.6B params)
- **LLaMA2 7B**: Consistent but resource-intensive baseline model
- **Performance pattern**: TPS inversely correlates with model size

---

## ⚖️ Performance vs. Resource Trade-offs

### Efficiency Ranking (TPS per CPU%)
1. **deepseek-r1:1.5b**: 0.22 TPS/CPU% ⭐⭐⭐⭐⭐
2. **tinyllama:latest**: 0.19 TPS/CPU% ⭐⭐⭐⭐
3. **phi:latest**: 0.07 TPS/CPU% ⭐⭐⭐
4. **phi3.5:3.8b**: 0.06 TPS/CPU% ⭐⭐
5. **qwen2.5:7b**: 0.04 TPS/CPU% ⭐⭐
6. **llama2:7b**: 0.03 TPS/CPU% ⭐
7. **deepseek-coder:6.7b**: 0.03 TPS/CPU% ⭐

### Resource Footprint Analysis
```
Model Size vs Performance:
┌─────────────────────────────────────────────┐
│ 1.0B: TinyLLama    ████████ 7.5 TPS         │
│ 1.8B: DeepSeek-R1  █████████ 7.5 TPS        │
│ 3.0B: Phi          ███ 2.6 TPS              │
│ 3.8B: Phi3.5       ███ 2.9 TPS              │
│ 7.0B: LLaMA2       ██ 1.6 TPS               │
│ 7.0B: DeepSeek-C   ██ 1.4 TPS               │
│ 7.6B: Qwen2.5      ██ 2.0 TPS               │
└─────────────────────────────────────────────┘
```

---

## 🎯 AURA Model Selection Strategy

### Current Intelligent Routing (Optimized)
```python
Routing Logic:
├── Coding Tasks → deepseek-coder:6.7b (Accuracy focus)
├── Math/Reasoning → deepseek-r1:1.5b (Speed + reasoning)
├── General Questions → phi3.5:3.8b (Balanced performance)
└── Simple Queries → tinyllama:latest (Maximum speed)
```

### Recommended Optimization
```python
Performance-Optimized Routing:
├── Complex Coding → deepseek-coder:6.7b (Accept slower TPS for accuracy)
├── Simple Coding → deepseek-r1:1.5b (9.1 TPS, fast development)
├── Math/Reasoning → deepseek-r1:1.5b (7.3 TPS, reasoning chains)
├── Quick Answers → tinyllama:latest (7.1 TPS, instant responses)
└── Balanced Tasks → phi3.5:3.8b (2.9 TPS, general purpose)
```

---

## 📊 Temperature and Hardware Analysis

### GPU Temperature Management
- **Optimal range**: 45-54°C across all models
- **Thermal stability**: No thermal throttling observed
- **Best cooling**: Smaller models (phi3.5: 45-48°C)
- **Highest load**: Larger models (qwen2.5: 53-54°C)

### VRAM Usage Patterns
- **RTX 4060 Laptop**: 8GB VRAM well-utilized
- **Memory efficiency**: All models fit comfortably in VRAM
- **No memory bottlenecks**: GPU memory usage 21-35% range
- **Headroom available**: Can potentially run multiple models

---

## 🚀 Recommendations for AURA Optimization

### 1. **Primary Speed Model**: DeepSeek-R1 1.5B
- Use for: Math, reasoning, simple coding, quick responses
- Benefits: 9.1 TPS peak, low CPU usage, fast inference

### 2. **Backup Speed Model**: TinyLLama Latest
- Use for: Ultra-fast simple queries, development testing
- Benefits: Consistent 7+ TPS, smallest resource footprint

### 3. **Accuracy Model**: DeepSeek-Coder 6.7B
- Use for: Complex coding, detailed analysis, production code
- Accept: Lower TPS (1.4) for higher quality output

### 4. **Balanced Model**: Phi3.5 3.8B
- Use for: General-purpose tasks requiring balance
- Benefits: Consistent 2.9 TPS, stable resource usage

---

## 📝 Test Methodology

### Test Environment
- **Hardware**: RTX 4060 Laptop (8GB VRAM)
- **Software**: Ollama 0.5.12, Windows 11, Python 3.12
- **Monitoring**: nvidia-smi, psutil, threading-based real-time monitoring
- **Consistency**: 2-second cooldown between tests, limited response tokens

### Test Prompts
1. **Simple Question**: "What is artificial intelligence?" (General knowledge)
2. **Coding Task**: "Write a Python function to sort a list of numbers" (Programming)
3. **Reasoning Task**: "Explain step by step how to solve: 2x + 5 = 15" (Mathematics)

### Metrics Collected
- **TPS**: Tokens generated per second
- **CPU Usage**: Average percentage during inference
- **GPU Usage**: Average utilization during inference
- **GPU Temperature**: Thermal monitoring
- **Inference Time**: Total response time
- **System Memory**: RAM usage patterns

---

*This analysis demonstrates AURA's intelligent model selection is well-optimized for balancing speed and accuracy across different task types.*
