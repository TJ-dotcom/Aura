#!/usr/bin/env python3
"""
AURA Command Line Interface - Hardware-Aware AI Engine
Core Vision: Intelligent model routing with hardware optimization
"""

import argparse
import sys
import os
import logging
from typing import Optional, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aura_engine.models import EngineConfig
from aura_engine.hardware import HardwareProfiler

# Try to import full engine, fallback to limited mode if dependencies missing
try:
    from aura_engine.engine import InferenceEngine
    from aura_engine.orchestrator import ModelOrchestrator
    from aura_engine.orchestrator.enhanced_router import EnhancedRouter
    ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Full engine not available ({e}). Using Ollama fallback mode.")
    ENGINE_AVAILABLE = False


class AuraCLI:
    """
    AURA CLI - Intelligent AI Engine with Hardware Optimization
    Core Features:
    1. Hardware-aware inference (auto-detects RAM/VRAM, optimizes settings)
    2. Intelligent model routing (analyzes prompts, selects optimal model)
    3. RAG integration (optional document augmentation)
    """
    
    def __init__(self):
        self.config = None
        self.engine = None
        self.hardware_profiler = HardwareProfiler()
        
    def setup_logging(self, level: str = 'INFO'):
        """Setup logging configuration."""
        numeric_level = getattr(logging, level.upper(), None)
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def cmd_infer(self, args):
        """
        CORE AURA COMMAND: Intelligent inference with hardware optimization
        This is the main entry point that demonstrates AURA's core vision:
        - Hardware profiling and optimization
        - Intelligent model routing based on prompt analysis
        - Optional RAG augmentation
        """
        if not ENGINE_AVAILABLE:
            if args.verbose:
                print("❌ Full AURA engine not available. Using Ollama fallback...")
            self._run_ollama_fallback(args.prompt, args.model, args.verbose)
            return
            
        if args.verbose:
            print("🔍 AURA Hardware Profiling...")
        hardware_profile = self.hardware_profiler.get_hardware_profile()
        
        if args.verbose:
            print(f"   └─ System RAM: {hardware_profile.system_ram_mb} MB")
            if hardware_profile.gpu_vram_mb:
                print(f"   └─ GPU VRAM: {hardware_profile.gpu_vram_mb} MB ({hardware_profile.gpu_name})")
                print(f"   └─ Optimal GPU Layers: {hardware_profile.optimal_gpu_layers}")
            else:
                print("   └─ No GPU detected - CPU-only execution")
            print(f"   └─ Performance Tier: {hardware_profile.performance_tier.upper()}")
        
        # Initialize AURA engine with hardware-optimized config
        self.config = EngineConfig(
            temperature=args.temperature or 0.7,
            max_tokens=args.max_tokens or 500,
            log_level='INFO' if args.verbose else 'WARNING'
        )
        
        self.engine = InferenceEngine(self.config)
        
        # Initialize Phase 2 with model orchestration
        hardware_profile = self.hardware_profiler.get_hardware_profile()
        model_paths = self._get_model_paths_for_tier(hardware_profile)
        self.engine.initialize(model_paths)
        
        if args.prompt:
            # Single prompt mode with intelligent routing
            self._run_intelligent_inference(args.prompt, args.enable_rag, args.model, args.verbose)
        else:
            # Interactive mode
            self._run_interactive_mode(args.enable_rag, args.model, args.verbose)
    
    def _run_intelligent_inference(self, prompt: str, enable_rag: bool, force_model: Optional[str] = None, verbose: bool = False):
        """
        Core AURA Intelligence: Analyze prompt and select optimal model
        """
        if verbose:
            print("\n🧠 AURA Intelligence Analysis...")
        
        if force_model:
            if verbose:
                print(f"   └─ User specified model: {force_model}")
            selected_model = force_model
            optimization_reason = "User override"
            model_type = None  # Unknown when forced
        else:
            # AURA's intelligent model routing with enhanced analysis
            router = EnhancedRouter()
            model_type = router.analyze_prompt(prompt)
            selected_model = self._get_optimal_model_for_type(model_type)
            optimization_reason = f"Prompt analysis → {model_type.value} task"
            
        if verbose:
            print(f"   └─ Selected Model: {selected_model}")
            print(f"   └─ Optimization Reason: {optimization_reason}")
        
        if enable_rag and verbose:
            print("   └─ RAG Mode: Enabled")
        
        # Execute with AURA's optimized pipeline
        if verbose:
            print("\n⚡ AURA Inference Pipeline...")
        try:
            result = self.engine.process_prompt(
                prompt=prompt,
                model_path=selected_model,
                enable_rag=enable_rag,
                force_model_type=model_type  # Pass detected model type for caching
            )
            
            if verbose:
                print("\n" + "=" * 80)
                print(f"📤 AURA RESPONSE ({selected_model})")
                print("=" * 80)
            
            print(result.response)
            
            if verbose:
                print("\n" + "=" * 80)
                print(f"⚡ Performance: {result.metrics.tokens_per_second:.1f} TPS")
                print(f"🧠 Model Used: {result.model_used.value}")
                print(f"🔧 Hardware Optimized: Yes")
                if result.rag_context:
                    print(f"📚 RAG Context: {len(result.rag_context)} chars")
                print("=" * 80)
            
        except Exception as e:
            print(f"AURA Error: {e}")
    
    def _get_optimal_model_for_type(self, model_type) -> str:
        """
        Get the optimal model based on hardware profile and task type
        Following the Model Selection Plan v2 with tiered deployment
        """
        hardware_profile = self.hardware_profiler.get_hardware_profile()
        
        # Determine VRAM availability for tier selection
        vram_gb = hardware_profile.gpu_vram_mb / 1024 if hardware_profile.gpu_vram_mb else 0
        
        # Tiered model selection following Model Selection Plan v2
        if vram_gb > 10:  # Tier 1: High-Performance  
            model_map = {
                'coder': 'deepseek-coder:6.7b',       # DeepSeek-R1-D 8B (substitute: deepseek-coder)
                'writer': 'llama2:7b',                 # DeepSeek-R1-D 8B (substitute: llama2)
                'general': 'deepseek-r1:1.5b',        # MATH: DeepSeek-R1-D 7B (substitute: deepseek-r1)
                'math': 'deepseek-r1:1.5b'            # Dedicated math model
            }
        elif vram_gb >= 7.5:   # Tier 2: Balanced (>= 7.5GB to catch 8GB cards)
            model_map = {
                'coder': 'deepseek-coder:6.7b',       # CODING: DeepSeek-R1-D 7B (substitute: deepseek-coder)
                'writer': 'phi3.5:3.8b',              # TEXT: Orca-2 7B (substitute: phi3.5)
                'general': 'deepseek-r1:1.5b',        # MATH: DeepSeek-R1-D 7B (substitute: deepseek-r1)  
                'math': 'deepseek-r1:1.5b'            # Dedicated math model
            }
        else:  # Tier 3: High-Efficiency (< 7.5GB VRAM)
            model_map = {
                'coder': 'deepseek-r1:1.5b',          # CODING: Qwen2 (substitute: deepseek-r1)
                'writer': 'phi3.5:3.8b',              # TEXT: Phi-3.5-mini ✅ CORRECT
                'general': 'deepseek-r1:1.5b',        # MATH: DeepSeek-R1-D 1.5B ✅ CORRECT
                'math': 'deepseek-r1:1.5b'            # Dedicated math model
            }
            
        return model_map.get(model_type.value, 'tinyllama:latest')
    
    def _get_model_paths_for_tier(self, hardware_profile) -> Dict:
        """Get model paths dictionary for Phase 2 engine initialization."""
        # Import ModelType here to avoid circular imports
        from aura_engine.models import ModelType
        
        # Determine VRAM availability for tier selection
        vram_gb = hardware_profile.gpu_vram_mb / 1024 if hardware_profile.gpu_vram_mb else 0
        
        # Use the same tier logic as _get_optimal_model_for_type
        if vram_gb > 10:  # Tier 1: High-Performance  
            return {
                ModelType.CODER: 'deepseek-coder:6.7b',
                ModelType.WRITER: 'llama2:7b',
                ModelType.GENERAL: 'deepseek-r1:1.5b',
                ModelType.MATH: 'deepseek-r1:1.5b'
            }
        elif vram_gb >= 7.5:   # Tier 2: Balanced
            return {
                ModelType.CODER: 'deepseek-coder:6.7b',
                ModelType.WRITER: 'phi3.5:3.8b',
                ModelType.GENERAL: 'deepseek-r1:1.5b',
                ModelType.MATH: 'deepseek-r1:1.5b'
            }
        else:  # Tier 3: High-Efficiency
            return {
                ModelType.CODER: 'deepseek-r1:1.5b',
                ModelType.WRITER: 'phi3.5:3.8b',
                ModelType.GENERAL: 'deepseek-r1:1.5b',
                ModelType.MATH: 'deepseek-r1:1.5b'
            }
    
    def _run_interactive_mode(self, enable_rag: bool, force_model: Optional[str], verbose: bool = False):
        """Interactive mode with intelligent routing for each prompt."""
        print(f"\n💬 AURA Interactive Mode")
        print("🧠 Each prompt will be analyzed for optimal model selection")
        if verbose:
            print("🔧 Verbose mode enabled - showing detailed analysis")
        print("Type 'exit', 'quit', or 'bye' to stop")
        print("=" * 50)
        
        while True:
            try:
                prompt = input(f"\n[AURA] >>> ").strip()
                
                if prompt.lower() in ['exit', 'quit', 'bye']:
                    print("👋 AURA shutting down...")
                    break
                elif prompt.lower() == 'help':
                    self._show_interactive_help()
                    continue
                elif not prompt:
                    continue
                
                # Process each prompt with intelligent routing
                self._run_intelligent_inference(prompt, enable_rag, force_model, verbose)
                
            except KeyboardInterrupt:
                print("\n👋 AURA interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def cmd_pull(self, args):
        """Pull/download a model - similar to 'ollama pull'."""
        print(f"📥 Pulling model: {args.model}")
        
        # Check if it's available in Ollama first
        import subprocess
        try:
            result = subprocess.run(['ollama', 'pull', args.model], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ Successfully pulled {args.model}")
                self._update_model_registry(args.model)
            else:
                print(f"❌ Failed to pull {args.model}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⏱️ Pull timeout for {args.model} - continuing in background")
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first.")
        except Exception as e:
            print(f"❌ Error pulling {args.model}: {e}")
    
    def cmd_list(self, args):
        """List available models - similar to 'ollama list'."""
        print("📋 Available Models:")
        print("=" * 60)
        
        # Get models from Ollama
        import subprocess
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) >= 3:
                            name = parts[0]
                            size = parts[2]
                            task_type = self._detect_model_type(name)
                            print(f"  {name:<30} {size:<10} ({task_type})")
                else:
                    print("  No models found.")
            else:
                print("❌ Could not list models from Ollama")
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first.")
        
        # Show recommended models for current hardware
        print("\n🎯 Recommended Models for Your Hardware:")
        profiler = HardwareProfiler()
        profile = profiler.get_hardware_profile()
        print(f"Performance Tier: {profile.performance_tier.upper()}")
        
        recommendations = self._get_hardware_recommendations(profile)
        for category, models in recommendations.items():
            print(f"\n  {category.title()}:")
            for model in models:
                print(f"    • {model}")
    
    def cmd_show(self, args):
        """Show model information - similar to 'ollama show'."""
        print(f"🔍 Model Information: {args.model}")
        print("=" * 50)
        
        # Get info from Ollama
        import subprocess
        try:
            result = subprocess.run(['ollama', 'show', args.model], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
                
                # Add AURA-specific info
                task_type = self._detect_model_type(args.model)
                settings = self._get_model_settings(args.model, task_type)
                
                print("\n🎯 AURA Optimization Settings:")
                print(f"  Task Type: {task_type}")
                print(f"  Temperature: {settings['temperature']}")
                print(f"  Max Tokens: {settings['max_tokens']}")
                print(f"  Best For: {settings['description']}")
            else:
                print(f"❌ Model {args.model} not found")
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first.")
    
    def cmd_ps(self, args):
        """Show running models - similar to 'ollama ps'."""
        print("🖥️  Running Models:")
        print("=" * 40)
        
        import subprocess
        try:
            result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("No models currently running")
        except FileNotFoundError:
            print("❌ Ollama not found")
    
    def cmd_serve(self, args):
        """Start AURA server mode."""
        print("🚀 Starting AURA Server...")
        print("This would start a server interface (not implemented yet)")
        # Future: Start a web interface or API server
    
    def _run_ollama_fallback(self, prompt: str, model: Optional[str], verbose: bool = False):
        """Fallback to direct Ollama when full engine unavailable."""
        if verbose:
            print("🔄 Using Ollama fallback mode...")
        
        # Still do basic hardware profiling
        hardware_profile = self.hardware_profiler.get_hardware_profile()
        if verbose:
            print(f"📊 Hardware: {hardware_profile.performance_tier.upper()} tier")
        
        # Simple model selection if none specified
        if not model:
            if hardware_profile.performance_tier == 'high-performance':
                model = 'llama2:7b'
            elif hardware_profile.performance_tier == 'balanced':
                model = 'llama2:7b'
            else:
                model = 'tinyllama:latest'
            if verbose:
                print(f"🤖 Auto-selected: {model}")
        
        import subprocess
        try:
            if prompt:
                result = subprocess.run(['ollama', 'run', model, prompt], 
                                      capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print("\n" + "=" * 80)
                    print(f"📤 {model.upper()} RESPONSE (Ollama Fallback)")
                    print("=" * 80) 
                    print(result.stdout)
                    print("=" * 80)
                else:
                    print(f"❌ Error: {result.stderr}")
            else:
                # Interactive
                subprocess.run(['ollama', 'run', model], check=True)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_hardware(self, args):
        """Show detailed hardware profile and recommendations."""
        print("🔍 AURA Hardware Analysis")
        print("=" * 50)
        
        hardware_profile = self.hardware_profiler.get_hardware_profile()
        
        print(f"💾 System RAM: {hardware_profile.system_ram_mb:,} MB")
        if hardware_profile.gpu_vram_mb:
            print(f"🎮 GPU: {hardware_profile.gpu_name}")
            print(f"🎮 GPU VRAM: {hardware_profile.gpu_vram_mb:,} MB") 
            print(f"⚙️  Optimal GPU Layers: {hardware_profile.optimal_gpu_layers}")
        else:
            print("🎮 GPU: None detected (CPU-only mode)")
        
        print(f"🏆 Performance Tier: {hardware_profile.performance_tier.upper()}")
        print(f"🔧 CPU Cores: {hardware_profile.cpu_cores}")
        
        print("\n🎯 Recommended Models:")
        recommendations = self._get_hardware_recommendations(hardware_profile)
        for category, models in recommendations.items():
            print(f"\n  {category.title()}:")
            for model in models:
                print(f"    • {model}")
    
    def cmd_models(self, args):
        """Enhanced model listing with AURA intelligence."""
        print("🤖 AURA Model Intelligence")
        print("=" * 60)
        
        # Get models from Ollama
        import subprocess
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    print("📋 Available Models:")
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) >= 3:
                            name = parts[0]
                            size = parts[2]
                            task_type = self._detect_model_specialty(name)
                            optimal_for = self._get_optimal_use_case(name)
                            print(f"  {name:<25} {size:<8} → {task_type:<10} ({optimal_for})")
                else:
                    print("  No models found.")
            else:
                print("❌ Could not list models from Ollama")
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first.")
        
        # Show AURA's intelligent recommendations
        hardware_profile = self.hardware_profiler.get_hardware_profile()
        print(f"\n🧠 AURA Intelligence for {hardware_profile.performance_tier.upper()} Hardware:")
        recommendations = self._get_hardware_recommendations(hardware_profile)
        for category, models in recommendations.items():
            print(f"\n  {category.title()} Tasks:")
            for model in models:
                print(f"    🎯 {model}")
    
    def _detect_model_specialty(self, model_name: str) -> str:
        """Detect what a model is specialized for."""
        model_lower = model_name.lower()
        if any(x in model_lower for x in ['code', 'coder', 'deepseek']):
            return 'CODING'
        elif any(x in model_lower for x in ['tiny', 'small']):
            return 'CHAT'
        elif any(x in model_lower for x in ['llama', 'mistral']):
            return 'GENERAL'
        else:
            return 'UNKNOWN'
    
    def _get_optimal_use_case(self, model_name: str) -> str:
        """Get the optimal use case description for a model."""
        specialty = self._detect_model_specialty(model_name)
        use_cases = {
            'CODING': 'Programming, debugging, code review',
            'CHAT': 'Quick Q&A, fast responses',
            'GENERAL': 'Writing, analysis, complex reasoning',
            'UNKNOWN': 'General purpose'
        }
    def cmd_pull(self, args):
        """Pull/download a model with AURA intelligence."""
        print(f"📥 AURA Model Acquisition: {args.model}")
        
        # Show why AURA might recommend this model
        task_type = self._detect_model_specialty(args.model)
        print(f"🎯 Model Type: {task_type}")
        print(f"🏆 Best For: {self._get_optimal_use_case(args.model)}")
        
        import subprocess
        try:
            result = subprocess.run(['ollama', 'pull', args.model], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ Successfully pulled {args.model}")
                print("🧠 AURA will now consider this model for intelligent routing")
            else:
                print(f"❌ Failed to pull {args.model}: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⏱️ Pull timeout for {args.model} - continuing in background")
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first.")
        except Exception as e:
            print(f"❌ Error pulling {args.model}: {e}")
    
    def cmd_show(self, args):
        """Show model information with AURA intelligence."""
        print(f"🔍 AURA Model Analysis: {args.model}")
        print("=" * 50)
        
        # Get info from Ollama
        import subprocess
        try:
            result = subprocess.run(['ollama', 'show', args.model], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
                
                # Add AURA-specific intelligence
                task_type = self._detect_model_specialty(args.model)
                optimal_use = self._get_optimal_use_case(args.model)
                
                print("\n🧠 AURA Intelligence Analysis:")
                print(f"  Model Type: {task_type}")
                print(f"  Optimal For: {optimal_use}")
                
                # Show hardware compatibility
                hardware_profile = self.hardware_profiler.get_hardware_profile()
                print(f"  Hardware Tier: {hardware_profile.performance_tier.upper()}")
                
                if 'deepseek' in args.model.lower() or '6.7b' in args.model.lower():
                    if hardware_profile.performance_tier in ['balanced', 'high-performance']:
                        print("  ✅ Recommended for your hardware")
                    else:
                        print("  ⚠️  May be too large for your hardware")
                elif 'tinyllama' in args.model.lower():
                    print("  ⚡ Excellent for fast responses on any hardware")
                    
            else:
                print(f"❌ Model {args.model} not found")
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first.")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_ps(self, args):
        """Show running models with AURA context."""
        print("🖥️  AURA Process Monitor")
        print("=" * 40)
        
        import subprocess
        try:
            result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True)
            if result.returncode == 0:
                print("📋 Active Models:")
                print(result.stdout)
                print("\n🧠 AURA Note: Models shown are managed by Ollama")
                print("   AURA intelligently routes between these models")
            else:
                print("No models currently running")
        except FileNotFoundError:
            print("❌ Ollama not found")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _get_hardware_recommendations(self, profile) -> Dict[str, list]:
        """Get model recommendations based on hardware profile."""
        if profile.performance_tier == 'balanced':
            return {
                'coding': ['deepseek-coder:6.7b', 'codellama:7b'],
                'writing': ['llama2:7b', 'mistral:7b'],
                'chat': ['tinyllama:latest', 'llama2:7b']
            }
        elif profile.performance_tier == 'high-performance':
            return {
                'coding': ['deepseek-coder:33b', 'deepseek-coder:6.7b'],
                'writing': ['llama2:13b', 'llama2:7b'],
                'chat': ['llama2:7b', 'mistral:7b']
            }
        else:  # high-efficiency
            return {
                'coding': ['deepseek-coder:1.3b', 'tinyllama:latest'],
                'writing': ['tinyllama:latest', 'phi:2.7b'],
                'chat': ['tinyllama:latest']
            }
    
    def _show_interactive_help(self):
        """Show help in interactive mode."""
        print("\n🧠 AURA Interactive Intelligence:")
        print("  • AURA analyzes each prompt to select optimal model")
        print("  • Hardware profile optimizes performance automatically")
        print("  • Type naturally - AURA handles the complexity")
        print("\n📚 Commands:")
        print("  help     - Show this help")
        print("  exit     - Exit AURA")
        print("  quit     - Exit AURA")
        print("  bye      - Exit AURA")
    
    def _get_hardware_recommendations(self, profile) -> Dict[str, list]:
        """Get model recommendations based on hardware."""
        if profile.performance_tier == 'balanced':
            return {
                'coding': ['deepseek-coder:6.7b', 'codellama:7b'],
                'writing': ['llama2:7b', 'mistral:7b'],
                'chat': ['tinyllama', 'llama2:7b']
            }
        elif profile.performance_tier == 'high-performance':
            return {
                'coding': ['deepseek-coder:33b', 'codellama:13b'],
                'writing': ['llama2:13b', 'mistral:7b'],
                'chat': ['llama2:7b', 'mistral:7b']
            }
        else:  # high-efficiency
            return {
                'coding': ['deepseek-coder:1.3b', 'tinyllama'],
                'writing': ['tinyllama', 'phi:2.7b'],
                'chat': ['tinyllama']
            }
    
    def _update_model_registry(self, model: str):
        """Update local model registry."""
        # Future: Track pulled models and their metadata
        pass
    
    def _show_interactive_help(self):
        """Show help in interactive mode."""
        print("\n📚 Interactive Commands:")
        print("  help     - Show this help")
        print("  exit     - Exit interactive mode")
        print("  quit     - Exit interactive mode")
        print("  bye      - Exit interactive mode")


def create_parser():
    """Create the AURA argument parser with intelligent defaults."""
    parser = argparse.ArgumentParser(
        prog='aura',
        description='AURA AI Engine - Hardware-Aware Intelligent Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aura "Write a Python function to sort a list"     # Direct inference (quiet by default)
  aura --verbose "Fix this code"                     # Show detailed analysis  
  aura --model deepseek-coder:6.7b "Debug this"     # Force specific model
  aura --rag "What does the document say about AI?" # Enable RAG augmentation
  aura --interactive                                 # Interactive mode
  aura hardware                                      # Show hardware profile
  aura models                                        # List models with AURA intelligence
  
AURA's Core Intelligence:
  • Hardware profiling and optimization (silent by default, use --verbose for details)
  • Automatic model selection based on prompt analysis
  • Performance tier adaptation (high-performance/balanced/efficient)
  • Clean output focused on AI responses (use --verbose for technical details)
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='AURA Commands')
    
    # Main inference command (default)
    infer_parser = subparsers.add_parser('infer', help='Intelligent inference (default)')
    infer_parser.add_argument('prompt', nargs='?', help='Prompt for inference')
    infer_parser.add_argument('--model', help='Force specific model (overrides AURA intelligence)')
    infer_parser.add_argument('--rag', '--enable-rag', action='store_true', dest='enable_rag', 
                             help='Enable RAG document augmentation')
    infer_parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    infer_parser.add_argument('--temperature', type=float, help='Sampling temperature')
    infer_parser.add_argument('--max-tokens', type=int, help='Maximum tokens to generate')
    infer_parser.add_argument('-v', '--verbose', action='store_true', help='Show hardware analysis, model selection reasoning, and performance metrics')
    
    # Hardware analysis
    hw_parser = subparsers.add_parser('hardware', help='Show hardware profile and recommendations')
    
    # Model intelligence
    models_parser = subparsers.add_parser('models', help='List models with AURA intelligence')
    
    # Legacy Ollama-style commands (redirected to AURA intelligence)
    pull_parser = subparsers.add_parser('pull', help='Pull a model')
    pull_parser.add_argument('model', help='Model to pull')
    
    show_parser = subparsers.add_parser('show', help='Show model information')  
    show_parser.add_argument('model', help='Model to show info for')
    
    ps_parser = subparsers.add_parser('ps', help='List running models')
    
    return parser


def main():
    """Main AURA CLI entry point."""
    parser = create_parser()
    
    # Handle direct prompts without subcommand
    if len(sys.argv) > 1:
        # Check if first argument is a flag or a command
        first_arg = sys.argv[1]
        is_command = first_arg in ['infer', 'hardware', 'models', 'pull', 'show', 'ps']
        is_help = first_arg in ['-h', '--help']
        is_flag = first_arg.startswith('-')
        
        # If help flag, let argparse handle it normally
        if is_help:
            args = parser.parse_args()
        # If not a command or flag, treat as direct prompt
        elif not is_command and not is_flag:
            # Handle direct prompt: aura "prompt text"
            prompt = ' '.join(sys.argv[1:])
            args = argparse.Namespace()
            args.command = 'infer'
            args.prompt = prompt
            args.model = None
            args.enable_rag = False
            args.interactive = False
            args.temperature = None
            args.max_tokens = None
            args.verbose = False
        elif not is_command and is_flag and not is_help:
            # Handle direct prompt with flags: aura --verbose "prompt"
            # Find the prompt (first non-flag argument)
            prompt_args = []
            verbose = False
            model = None
            enable_rag = False
            interactive = False
            temperature = None
            max_tokens = None
            
            i = 1
            while i < len(sys.argv):
                arg = sys.argv[i]
                if arg in ['-v', '--verbose']:
                    verbose = True
                elif arg == '--model' and i + 1 < len(sys.argv):
                    model = sys.argv[i + 1]
                    i += 1  # Skip the model value
                elif arg in ['--rag', '--enable-rag']:
                    enable_rag = True
                elif arg == '--interactive':
                    interactive = True
                elif arg == '--temperature' and i + 1 < len(sys.argv):
                    temperature = float(sys.argv[i + 1])
                    i += 1
                elif arg == '--max-tokens' and i + 1 < len(sys.argv):
                    max_tokens = int(sys.argv[i + 1])
                    i += 1
                elif not arg.startswith('-'):
                    prompt_args.append(arg)
                i += 1
            
            prompt = ' '.join(prompt_args) if prompt_args else None
            
            args = argparse.Namespace()
            args.command = 'infer'
            args.prompt = prompt
            args.model = model
            args.enable_rag = enable_rag
            args.interactive = interactive
            args.temperature = temperature
            args.max_tokens = max_tokens
            args.verbose = verbose
        else:
            args = parser.parse_args()
            
            # If no command specified with flags, show help
            if not args.command:
                parser.print_help()
                return 1
    
    cli = AuraCLI()
    
    try:
        if args.command == 'infer' or not args.command:
            cli.cmd_infer(args)
        elif args.command == 'hardware':
            cli.cmd_hardware(args)  
        elif args.command == 'models':
            cli.cmd_models(args)
        elif args.command == 'pull':
            cli.cmd_pull(args)
        elif args.command == 'show':
            cli.cmd_show(args)
        elif args.command == 'ps':
            cli.cmd_ps(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 AURA interrupted")
        return 130
    except Exception as e:
        print(f"AURA Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
