# AURA Project File Structure Management

## Current Status: File Structure Cleanup & Organization

### 📁 **Core Production Files (KEEP)**

#### Essential Runtime Files
- `aura.py` - Main CLI interface with intelligent routing
- `main.py` - Legacy entry point (preserved for compatibility)
- `aura.bat` - Windows PATH integration
- `requirements.txt` - Python dependencies

#### Task-Specific Commands
- `aura-code.bat` - Coding tasks with DeepSeek Coder
- `aura-write.bat` - Creative writing with Llama2
- `aura-chat.bat` - Quick responses with TinyLlama
- `aura-analyze.bat` - Analysis with RAG integration

#### Core Engine
- `aura_engine/` - Complete 3-phase system
  - `__init__.py`, `cli.py`, `engine.py`, `models.py`
  - `hardware/` - Hardware profiling system
  - `llama_wrapper/` - llama.cpp integration
  - `ollama_wrapper/` - Ollama integration (primary)
  - `orchestrator/` - Model management and routing
  - `performance/` - Performance monitoring
  - `rag/` - RAG pipeline (Phase 3)

#### Installation & Setup
- `install.ps1` - Comprehensive PowerShell installer
- `install.bat` - Simple batch alternative
- `install-aura.ps1` - CLI PATH integration

#### Data & Models
- `models/` - Local model storage
- `rag_data/` - RAG indices and documents
- `.venv/` - Python virtual environment

### 📚 **Documentation Files (KEEP & ORGANIZE)**

#### Primary Documentation
- `README.md` - Main project documentation
- `project-context.md` - Original project mandate
- `OPERATIONAL_LOG.md` - Complete development history

#### User Guides
- `CLI_GUIDE.md` - Command reference
- `CUSTOM_COMMANDS.md` - Task-specific usage guide

#### Technical Documentation
- `DEFINITIVE_CPU_ANALYSIS.md` - **FINAL** CPU usage analysis
- `OLLAMA_VS_LLAMA_CPP_ANALYSIS.md` - **USER EDITED** - Architecture comparison
- `Model_selection.md` - Model selection strategy
- `TESTING_PROTOCOL.md` - Quality assurance framework

#### Performance & Benchmarking
- `BENCHMARKS.md` - Performance results
- `BENCHMARKING_STRATEGY.md` - Performance measurement framework
- `PERFORMANCE_ANALYSIS_PLAN.md` - Optimization methodology

### 🧪 **Testing Files (CONSOLIDATE)**

#### Core Tests
- `tests/` - Comprehensive test suite (121+ tests)
- `ingest_documents.py` - RAG document ingestion

#### Specialized Analysis (KEEP BEST, REMOVE DUPLICATES)
- `analyze_ollama_architecture.py` - **COMPREHENSIVE** - Definitive analysis
- `cpu_impact_analysis.py` - **USER EDITED** - Real-world impact analysis

### 🗑️ **Debug/Development Files (CLEAN UP)**

#### Debug Scripts (REMOVE - No longer needed)
- `debug_api_call.py`
- `debug_hardware.py` 
- `debug_model_selection.py`
- `debug_ollama_calls.py`
- `debug_parameters.py`
- `debug_payload.py`
- `debug_router.py`
- `debug_routing.py`
- `debug_specific_prompt.py`
- `debug_test.py`

#### Temporary Development Files (REMOVE)
- `demonstrate_phase3.py`
- `streaming_test.py`
- `profile_test.py`
- `test_ollama.py`
- `test_rag.py`
- `preload_models.py`
- `aura_tasks.py`

#### GPU Optimization Scripts (ARCHIVE - Keep for reference)
- `gpu_only_enforcer.py`
- `eliminate_cpu_usage.py`
- `setup-gpu-only.bat`
- `monitor.ps1`

#### Redundant Analysis Files (REMOVE)
- `test_cpu_optimization.py` (superseded by comprehensive analysis)
- `CPU_USAGE_ANALYSIS.md` (superseded by DEFINITIVE_CPU_ANALYSIS.md)
- `OLLAMA_CPU_ANALYSIS.md` (superseded by DEFINITIVE_CPU_ANALYSIS.md)

#### Empty/Test Files (REMOVE)
- `test_document.md` (empty)
- `validation_test.md` (minimal test content)

### 📋 **File Structure Actions Required**

#### 1. Remove Debug Files
```powershell
Remove-Item debug_*.py
Remove-Item demonstrate_phase3.py, streaming_test.py, profile_test.py
Remove-Item test_ollama.py, test_rag.py, preload_models.py, aura_tasks.py
Remove-Item test_cpu_optimization.py
Remove-Item test_document.md, validation_test.md
```

#### 2. Archive GPU Scripts
```powershell
New-Item -ItemType Directory -Path "archive_gpu_scripts"
Move-Item gpu_only_enforcer.py, eliminate_cpu_usage.py, setup-gpu-only.bat, monitor.ps1 archive_gpu_scripts/
```

#### 3. Consolidate Analysis Files
```powershell
# Remove superseded analysis files
Remove-Item CPU_USAGE_ANALYSIS.md, OLLAMA_CPU_ANALYSIS.md
# Keep: DEFINITIVE_CPU_ANALYSIS.md, OLLAMA_VS_LLAMA_CPP_ANALYSIS.md (user edited)
```

#### 4. Update Documentation References
- Update README.md with clean file structure
- Update OPERATIONAL_LOG.md to reflect final state
- Ensure all documentation references valid files only

### 🎯 **Final Clean Structure**

```
aura-ai-engine/
├── 🚀 CORE SYSTEM
│   ├── aura.py, main.py, aura.bat
│   ├── aura-code.bat, aura-write.bat, aura-chat.bat, aura-analyze.bat
│   └── aura_engine/ (complete engine with all modules)
├── 📦 INSTALLATION  
│   ├── install.ps1, install.bat, install-aura.ps1
│   └── requirements.txt
├── 📚 DOCUMENTATION
│   ├── README.md, project-context.md, OPERATIONAL_LOG.md
│   ├── CLI_GUIDE.md, CUSTOM_COMMANDS.md
│   ├── DEFINITIVE_CPU_ANALYSIS.md, OLLAMA_VS_LLAMA_CPP_ANALYSIS.md
│   ├── Model_selection.md, TESTING_PROTOCOL.md
│   └── BENCHMARKS.md, BENCHMARKING_STRATEGY.md, PERFORMANCE_ANALYSIS_PLAN.md
├── 🧪 TESTING & ANALYSIS
│   ├── tests/ (comprehensive test suite)
│   ├── ingest_documents.py
│   ├── analyze_ollama_architecture.py
│   └── cpu_impact_analysis.py
├── 🗃️ DATA
│   ├── models/, rag_data/
│   └── .venv/
└── 📦 ARCHIVE
    └── archive_gpu_scripts/ (GPU optimization scripts for reference)
```

### 🎉 **Benefits of Clean Structure**

1. **Professional Presentation** - Clear, organized codebase
2. **Reduced Confusion** - Remove redundant and outdated files  
3. **Easier Maintenance** - Focus on essential files only
4. **Portfolio Ready** - Clean structure for demonstrations
5. **Better Performance** - Fewer files to scan and load

This cleanup will transform the project from a development workspace into a polished, production-ready AI system.
