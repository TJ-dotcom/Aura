"""
Core inference engine for AURA-Engine-Core.
"""

import logging
import os
from typing import Optional, Dict

from .models import EngineConfig, InferenceResult, PerformanceMetrics, ModelType, HardwareProfile
from .hardware import HardwareProfiler
from .llama_wrapper import LlamaWrapper
from .ollama_wrapper import OllamaWrapper, OllamaOutput
from .performance import PerformanceMonitor
from .orchestrator import ModelOrchestrator
from .rag import RAGPipeline


class InferenceEngine:
    """
    Core inference engine that orchestrates hardware profiling,
    model execution, and performance monitoring with intelligent model routing.
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.hardware_profiler = HardwareProfiler()
        self.performance_monitor = PerformanceMonitor(config)
        
        # Model wrappers - try Ollama first, fallback to llama.cpp
        self.ollama_wrapper: Optional[OllamaWrapper] = None
        self.llama_wrapper: Optional[LlamaWrapper] = None
        self.use_ollama = False
        
        # Try to initialize Ollama first
        try:
            self.ollama_wrapper = OllamaWrapper(config)
            self.use_ollama = True
            self.logger.info("Using Ollama backend")
        except Exception as e:
            self.logger.warning(f"Ollama not available: {e}")
            self.logger.info("Falling back to llama.cpp backend")
            try:
                self.llama_wrapper = LlamaWrapper(config)
                self.logger.info("Using llama.cpp backend")
            except Exception as e:
                self.logger.error(f"Neither Ollama nor llama.cpp available: {e}")
                # Will fail during initialization if no backend is available
        
        # Initialize RAG pipeline for Phase 3
        self.rag_pipeline: Optional[RAGPipeline] = None
        
        # State
        self.hardware_profile: Optional[HardwareProfile] = None
        self.model_orchestrator: Optional[ModelOrchestrator] = None
        self.is_initialized = False
    
    def initialize(self, model_paths: Optional[Dict[ModelType, str]] = None) -> None:
        """
        Initialize the inference engine by profiling hardware
        and preparing components.
        
        Args:
            model_paths: Optional dictionary of model paths for Phase 2 orchestration
        """
        self.logger.info("Initializing AURA-Engine-Core...")
        
        # Check if we have any working backend
        if not self.use_ollama and not self.llama_wrapper:
            raise RuntimeError("No inference backend available. Please install either Ollama or llama.cpp")
        
        # Profile hardware capabilities
        self.logger.info("Profiling hardware...")
        self.hardware_profile = self.hardware_profiler.get_hardware_profile()
        
        # Log hardware information
        self.logger.info(f"Hardware Profile: {self.hardware_profile}")
        
        # Initialize model orchestrator if model paths provided (Phase 2)
        if model_paths:
            if self.use_ollama and self.ollama_wrapper:
                # For Ollama, we handle routing directly and initialize cache
                self.logger.info("Initializing Ollama-based routing with model caching...")
                model_map = self._extract_model_map(model_paths)
                self.ollama_wrapper.initialize_model_cache(model_map)
                # Store model paths for direct routing
                self.model_paths = model_paths
            else:
                # For llama.cpp, use full orchestrator
                self.logger.info("Initializing model orchestrator for Phase 2...")
                wrapper = self.llama_wrapper
                self.model_orchestrator = ModelOrchestrator(
                    wrapper, 
                    self.hardware_profile, 
                    self.config
                )
                self.model_orchestrator.initialize(model_paths)
                self.logger.info("Model orchestrator initialized")
        
        # Validate configuration
        self.config.validate()
        
        self.is_initialized = True
        backend = "Ollama" if self.use_ollama else "llama.cpp"
        self.logger.info(f"Initialization complete using {backend} backend")
    
    def process_prompt(self, prompt: str, model_path: Optional[str] = None, 
                      gpu_layers_override: Optional[int] = None,
                      force_model_type: Optional[ModelType] = None,
                      enable_rag: bool = False) -> InferenceResult:
        """
        Process a prompt through the complete inference pipeline.
        
        Args:
            prompt: Input prompt for inference
            model_path: Path to the GGUF model file (Phase 1 only)
            gpu_layers_override: Override for GPU layers (None for auto-detection)
            force_model_type: Force a specific model type (Phase 2 only)
            enable_rag: Enable Retrieval-Augmented Generation (RAG) context retrieval
            
        Returns:
            InferenceResult: Complete inference result with metrics
            
        Raises:
            RuntimeError: If engine is not initialized
            FileNotFoundError: If model file is not found
        """
        if not self.is_initialized:
            raise RuntimeError("Engine must be initialized before processing prompts")
        
        original_prompt = prompt
        rag_context = None
        
        # Handle RAG if enabled
        if enable_rag:
            self.logger.info("RAG mode enabled. Retrieving context...")
            
            # Initialize RAG pipeline if not already done
            if self.rag_pipeline is None:
                self.rag_pipeline = RAGPipeline()
            
            try:
                rag_context = self.rag_pipeline.retrieve_context(prompt)
                if rag_context:
                    prompt = f"{rag_context}\n\nUser Query: {prompt}"
                    self.logger.info(f"Retrieved RAG context ({len(rag_context)} characters)")
                else:
                    self.logger.warning("No relevant context found in RAG index")
            except Exception as e:
                self.logger.error(f"RAG retrieval failed: {e}")
                # Continue without RAG context
        
        # Determine which mode we're in (Phase 1 or Phase 2)
        if self.model_orchestrator or (self.use_ollama and hasattr(self, 'model_paths')):
            # Phase 2: Use model orchestrator or direct Ollama routing
            return self._process_prompt_phase2(prompt, gpu_layers_override, force_model_type, rag_context)
        else:
            # Phase 1: Use single model
            if not model_path:
                raise ValueError("model_path is required for Phase 1 operation")
            return self._process_prompt_phase1(prompt, model_path, gpu_layers_override, rag_context)
    
    def _process_prompt_phase1(self, prompt: str, model_path: str, 
                              gpu_layers_override: Optional[int] = None, 
                              rag_context: Optional[str] = None) -> InferenceResult:
        """
        Process prompt using Phase 1 single-model approach.
        """
        self.logger.info(f"Processing prompt with model: {model_path}")
        
        # Start performance monitoring
        self.performance_monitor.start_inference_timer()
        
        try:
            # Execute inference based on available backend
            if self.use_ollama:
                # Use Ollama backend
                self.logger.info(f"Running Ollama inference...")
                processed_output = self.ollama_wrapper.run_inference(
                    model_path=model_path,
                    prompt=prompt
                )
                
                # Convert OllamaOutput to compatible format
                if processed_output.error_message:
                    raise RuntimeError(f"Ollama inference failed: {processed_output.error_message}")
                    
                # Create a mock output object that matches LlamaWrapper's interface
                class MockOutput:
                    def __init__(self, ollama_output: OllamaOutput):
                        self.response = ollama_output.response
                        self.tokens_generated = ollama_output.tokens_generated
                        self.model_load_time = ollama_output.model_load_time
                        self.error_message = ollama_output.error_message
                        self.native_tps = getattr(ollama_output, 'native_tps', None)  # Pass through native TPS
                
                processed_output = MockOutput(processed_output)
                
            else:
                # Use llama.cpp backend
                if not os.path.isfile(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                
                # Determine GPU layers to use
                gpu_layers = self._determine_gpu_layers(model_path, gpu_layers_override)
                
                self.logger.info(f"Running llama.cpp inference with {gpu_layers} GPU layers...")
                processed_output = self.llama_wrapper.run_inference(
                    model_path=model_path,
                    prompt=prompt,
                    gpu_layers=gpu_layers
                )
                
                # Check for inference errors
                if processed_output.error_message:
                    raise RuntimeError(f"Inference failed: {processed_output.error_message}")
            
            # Record performance metrics
            self.performance_monitor.record_first_token()  # Approximate
            
            # Use native TPS if available (from Ollama), otherwise let monitor calculate
            native_tps = getattr(processed_output, 'native_tps', None)
            self.performance_monitor.record_completion(
                processed_output.tokens_generated, 
                override_tps=native_tps
            )
            
            # Collect final metrics
            metrics = self.performance_monitor.get_metrics(
                model_load_time=processed_output.model_load_time or 0.0
            )
            
            # Create result
            result = InferenceResult(
                response=processed_output.response,
                metrics=metrics,
                model_used=ModelType.GENERAL,  # Phase 1 uses general model only
                rag_context=rag_context
            )
            
            # Log benchmark data if enabled
            if self.config.enable_benchmarking:
                self.performance_monitor.log_benchmark(
                    metrics=metrics,
                    hardware_profile=self.hardware_profile,
                    phase=1,
                    scenario="Baseline"
                )
            
            backend = "Ollama" if self.use_ollama else "llama.cpp"
            self.logger.info(f"{backend} inference completed: {processed_output.tokens_generated} tokens generated")
            return result
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise
    
    def _process_prompt_phase2(self, prompt: str, gpu_layers_override: Optional[int] = None,
                              force_model_type: Optional[ModelType] = None,
                              rag_context: Optional[str] = None) -> InferenceResult:
        """
        Process prompt using Phase 2 model orchestration approach.
        """
        self.logger.info("Processing prompt with model orchestration (Phase 2)")
        
        # Start performance monitoring
        self.performance_monitor.start_inference_timer()
        
        try:
            if self.use_ollama and hasattr(self, 'model_paths'):
                # Direct Ollama routing with caching
                from .orchestrator.enhanced_router import EnhancedRouter
                
                router = EnhancedRouter()
                selected_model_type = force_model_type or router.analyze_prompt(prompt)
                model_name = self.model_paths.get(selected_model_type, 'tinyllama:latest')
                
                self.logger.info(f"Direct Ollama routing: {selected_model_type.value} -> {model_name}")
                
                # Execute inference with model caching
                processed_output = self.ollama_wrapper.run_inference(
                    model_name=model_name,
                    prompt=prompt,
                    model_type=selected_model_type
                )
                
                # Convert OllamaOutput to compatible format
                if processed_output.error_message:
                    raise RuntimeError(f"Ollama inference failed: {processed_output.error_message}")
                    
                # Create a mock output object
                class MockOutput:
                    def __init__(self, ollama_output: OllamaOutput):
                        self.response = ollama_output.response
                        self.tokens_generated = ollama_output.tokens_generated
                        self.model_load_time = ollama_output.model_load_time
                        self.error_message = ollama_output.error_message
                        self.native_tps = getattr(ollama_output, 'native_tps', None)
                
                processed_output = MockOutput(processed_output)
                
            else:
                # Use orchestrator to determine model and path (llama.cpp)
                selected_model_type, model_path = self.model_orchestrator.process_prompt(
                    prompt, force_model_type, gpu_layers_override
                )
                
                # Determine GPU layers to use
                gpu_layers = self._determine_gpu_layers(model_path, gpu_layers_override)
                
                # Execute inference with llama.cpp
                self.logger.info(f"Running llama.cpp inference with {selected_model_type.value} model, {gpu_layers} GPU layers...")
                processed_output = self.llama_wrapper.run_inference(
                    model_path=model_path,
                    prompt=prompt,
                    gpu_layers=gpu_layers
                )
            
            # Check for inference errors
            if processed_output.error_message:
                raise RuntimeError(f"Inference failed: {processed_output.error_message}")
            
            # Record performance metrics
            self.performance_monitor.record_first_token()  # Approximate
            
            # Use native TPS if available (from Ollama), otherwise let monitor calculate
            native_tps = getattr(processed_output, 'native_tps', None)
            self.performance_monitor.record_completion(
                processed_output.tokens_generated, 
                override_tps=native_tps
            )
            
            # Collect final metrics
            metrics = self.performance_monitor.get_metrics(
                model_load_time=processed_output.model_load_time or 0.0
            )
            
            # Create result
            result = InferenceResult(
                response=processed_output.response,
                metrics=metrics,
                model_used=selected_model_type,
                rag_context=rag_context
            )
            
            # Log benchmark data if enabled
            if self.config.enable_benchmarking:
                scenario = "Model_Switching" if (self.model_orchestrator and self.model_orchestrator.get_current_model_info()) else "Baseline"
                self.performance_monitor.log_benchmark(
                    metrics=metrics,
                    hardware_profile=self.hardware_profile,
                    phase=2,
                    scenario=scenario
                )
            
            self.logger.info(f"Inference completed: {processed_output.tokens_generated} tokens generated using {selected_model_type.value} model")
            return result
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise
    
    def _determine_gpu_layers(self, model_path: str, override: Optional[int]) -> int:
        """
        Determine the optimal number of GPU layers to use.
        
        Args:
            model_path: Path to the model file
            override: User-specified override value
            
        Returns:
            int: Number of GPU layers to use
        """
        if override is not None:
            self.logger.info(f"Using user-specified GPU layers: {override}")
            return override
        
        if self.hardware_profile is None:
            self.logger.warning("No hardware profile available, using CPU-only")
            return 0
        
        # Estimate model size from file size (rough approximation)
        try:
            model_size_bytes = os.path.getsize(model_path)
            model_size_mb = model_size_bytes // (1024 * 1024)
            
            # Recalculate GPU layers based on actual model size
            gpu_layers = self.hardware_profiler.calculate_gpu_layers(
                model_size_mb=model_size_mb,
                available_vram_mb=self.hardware_profile.gpu_vram_mb or 0
            )
            
            self.logger.info(f"Auto-detected GPU layers: {gpu_layers} (model size: {model_size_mb}MB)")
            return gpu_layers
            
        except OSError as e:
            self.logger.error(f"Failed to get model file size: {e}")
            return self.hardware_profile.optimal_gpu_layers
    
    def get_orchestrator_info(self) -> Optional[Dict]:
        """
        Get information about the model orchestrator (Phase 2 only).
        
        Returns:
            Optional[Dict]: Orchestrator information or None if not available
        """
        if not self.model_orchestrator:
            return None
        
        return {
            'current_model': self.model_orchestrator.get_current_model_info() if self.model_orchestrator else 'Direct Ollama Routing',
            'memory_usage': self.model_orchestrator.get_memory_usage(),
            'available_models': list(self.model_orchestrator.get_available_models().keys())
        }
    
    def get_routing_explanation(self, prompt: str) -> Optional[Dict]:
        """
        Get detailed routing explanation for a prompt (Phase 2 only).
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            Optional[Dict]: Routing explanation or None if not available
        """
        if not self.model_orchestrator:
            return None
        
        return self.model_orchestrator.get_routing_explanation(prompt)
    
    def unload_current_model(self) -> None:
        """
        Unload the currently loaded model (Phase 2 only).
        """
        if self.model_orchestrator:
            self.model_orchestrator.unload_current_model()
            self.logger.info("Current model unloaded")
        else:
            self.logger.warning("No model orchestrator available for unloading")
    
    def shutdown(self) -> None:
        """
        Clean shutdown of the inference engine.
        """
        self.logger.info("Shutting down inference engine...")
        
        # Clean up model orchestrator if present
        if self.model_orchestrator:
            self.model_orchestrator.shutdown()
        
        self.is_initialized = False
        self.logger.info("Shutdown complete")
    
    def _extract_model_map(self, model_paths: Dict[ModelType, str]) -> Dict[str, str]:
        """Extract model names from model paths for cache initialization."""
        model_map = {}
        for model_type, path in model_paths.items():
            # Extract model name from path (for Ollama, path IS the model name)
            model_map[model_type.value] = path
        return model_map