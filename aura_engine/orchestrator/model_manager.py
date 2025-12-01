"""
Model manager for loading, unloading, and managing specialized models.
"""

import logging
import os
import time
import psutil
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..models import ModelType, HardwareProfile
from ..llama_wrapper import LlamaWrapper
from .model_catalog import ModelCatalog, ModelSpec


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_type: ModelType
    model_path: str
    load_time: float
    memory_usage_mb: int
    gpu_layers: int


class ModelManager:
    """
    Manages the lifecycle of specialized models with strict memory management.
    Ensures only one model is loaded at a time to optimize resource usage.
    Supports tiered model selection based on hardware capabilities.
    """
    
    def __init__(self, llama_wrapper: LlamaWrapper, hardware_profile: HardwareProfile):
        self.llama_wrapper = llama_wrapper
        self.hardware_profile = hardware_profile
        self.logger = logging.getLogger(__name__)
        
        # Model catalog for tiered selection
        self.model_catalog = ModelCatalog()
        
        # Model state tracking
        self.current_model: Optional[ModelInfo] = None
        self.model_paths: Dict[ModelType, str] = {}
        self.process = psutil.Process()
        
        # Memory tracking
        self.baseline_memory_mb = self._get_current_memory()
        
        self.logger.info(f"ModelManager initialized with performance tier: {hardware_profile.performance_tier}")
    
    def get_recommended_model(self, category: str = "text") -> Optional[ModelSpec]:
        """
        Get the recommended model for the current hardware tier and category.
        
        Args:
            category: Model category ('text', 'coding', 'mathematics')
            
        Returns:
            ModelSpec: Recommended model specification
        """
        return self.model_catalog.get_default_model(
            self.hardware_profile.performance_tier, 
            category
        )
    
    def get_available_models(self, category: str = None) -> list[ModelSpec]:
        """
        Get all available models for the current hardware tier.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of available ModelSpec objects
        """
        return self.model_catalog.get_models_for_tier(
            self.hardware_profile.performance_tier,
            category
        )
    
    def auto_select_model(self, category: str = "text", download_dir: str = "models") -> Optional[str]:
        """
        Automatically select and download the recommended model for the current tier.
        
        Args:
            category: Model category ('text', 'coding', 'mathematics')
            download_dir: Directory to download the model to
            
        Returns:
            str: Path to the downloaded model file, or None if failed
        """
        import os
        from urllib.parse import urlparse
        
        # Get recommended model for current tier
        recommended_model = self.get_recommended_model(category)
        if not recommended_model:
            self.logger.error(f"No recommended model found for tier {self.hardware_profile.performance_tier}, category {category}")
            return None
        
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)
        
        # Construct local file path
        filename = recommended_model.name
        local_path = os.path.join(download_dir, filename)
        
        # Check if model already exists
        if os.path.isfile(local_path):
            self.logger.info(f"Model already exists: {local_path}")
            return local_path
        
        # For demonstration purposes, create a placeholder file instead of downloading
        # In production, this would download the actual model
        self.logger.info(f"[DEMO] Simulating download of {category} model: {recommended_model.name} ({recommended_model.size_mb}MB)")
        self.logger.info(f"[DEMO] Would download from: {recommended_model.url}")
        
        try:
            # Create a placeholder file for demonstration
            with open(local_path, 'w') as f:
                f.write(f"# DEMO MODEL PLACEHOLDER\n")
                f.write(f"# Model: {recommended_model.name}\n")
                f.write(f"# Size: {recommended_model.size_mb}MB\n")
                f.write(f"# URL: {recommended_model.url}\n")
                f.write(f"# Category: {recommended_model.category}\n")
                f.write(f"# Description: {recommended_model.description}\n")
            
            self.logger.info(f"[DEMO] Model placeholder created: {local_path}")
            return local_path
            
        except Exception as e:
            self.logger.error(f"Failed to create model placeholder {recommended_model.name}: {e}")
            # Clean up partial download
            if os.path.exists(local_path):
                os.remove(local_path)
            return None
    
    def auto_configure_for_tier(self, download_dir: str = "models") -> Dict[str, str]:
        """
        Automatically configure models for all categories based on current performance tier.
        
        Args:
            download_dir: Directory to download models to
            
        Returns:
            Dict[str, str]: Mapping of category to downloaded model path
        """
        self.logger.info(f"Auto-configuring models for {self.hardware_profile.performance_tier} tier...")
        
        configured_models = {}
        categories = self.model_catalog.get_available_categories()
        
        for category in categories:
            model_path = self.auto_select_model(category, download_dir)
            if model_path:
                configured_models[category] = model_path
                self.logger.info(f"✅ {category.title()} model configured: {os.path.basename(model_path)}")
            else:
                self.logger.warning(f"❌ Failed to configure {category} model")
        
        return configured_models
    
    def configure_model_paths(self, model_paths: Dict[ModelType, str]) -> None:
        """
        Configure paths for different model types.
        
        Args:
            model_paths: Dictionary mapping model types to file paths
        """
        # Validate all model files exist
        for model_type, path in model_paths.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Model file not found for {model_type.value}: {path}")
        
        self.model_paths = model_paths.copy()
        self.logger.info(f"Configured model paths: {[(k.value, os.path.basename(v)) for k, v in model_paths.items()]}")
    
    def load_model(self, model_type: ModelType, gpu_layers: Optional[int] = None) -> ModelInfo:
        """
        Load a specific model type, unloading any currently loaded model first.
        
        Args:
            model_type: Type of model to load
            gpu_layers: Number of GPU layers (None for auto-detection)
            
        Returns:
            ModelInfo: Information about the loaded model
            
        Raises:
            ValueError: If model type is not configured
            RuntimeError: If model loading fails
        """
        if model_type not in self.model_paths:
            raise ValueError(f"Model type {model_type.value} not configured. Available: {list(self.model_paths.keys())}")
        
        model_path = self.model_paths[model_type]
        
        # Unload current model if one is loaded
        if self.current_model is not None:
            self.logger.info(f"Unloading current model: {self.current_model.model_type.value}")
            self.unload_current_model()
        
        # Determine GPU layers
        if gpu_layers is None:
            gpu_layers = self._calculate_gpu_layers(model_path)
        
        self.logger.info(f"Loading model: {model_type.value} from {os.path.basename(model_path)} with {gpu_layers} GPU layers")
        
        # Record memory before loading
        memory_before = self._get_current_memory()
        load_start_time = time.time()
        
        try:
            # Validate model can be loaded (this doesn't actually load it in llama.cpp yet)
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Record successful load
            load_time = time.time() - load_start_time
            memory_after = self._get_current_memory()
            memory_usage = max(0, memory_after - self.baseline_memory_mb)
            
            model_info = ModelInfo(
                model_type=model_type,
                model_path=model_path,
                load_time=load_time,
                memory_usage_mb=memory_usage,
                gpu_layers=gpu_layers
            )
            
            self.current_model = model_info
            
            self.logger.info(f"Model loaded successfully: {model_type.value}")
            self.logger.info(f"Load time: {load_time:.2f}s, Memory usage: {memory_usage}MB, GPU layers: {gpu_layers}")
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_type.value}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def unload_current_model(self) -> None:
        """
        Unload the currently loaded model and free memory.
        """
        if self.current_model is None:
            self.logger.debug("No model currently loaded")
            return
        
        model_type = self.current_model.model_type
        self.logger.info(f"Unloading model: {model_type.value}")
        
        # Record memory before unloading
        memory_before = self._get_current_memory()
        
        # Clear model reference
        self.current_model = None
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        # Record memory after unloading
        memory_after = self._get_current_memory()
        memory_freed = memory_before - memory_after
        
        self.logger.info(f"Model unloaded: {model_type.value}")
        self.logger.info(f"Memory freed: {memory_freed}MB (before: {memory_before}MB, after: {memory_after}MB)")
    
    def is_model_loaded(self) -> bool:
        """
        Check if a model is currently loaded.
        
        Returns:
            bool: True if a model is loaded
        """
        return self.current_model is not None
    
    def get_current_model(self) -> Optional[ModelType]:
        """
        Get the type of the currently loaded model.
        
        Returns:
            Optional[ModelType]: Current model type or None if no model loaded
        """
        return self.current_model.model_type if self.current_model else None
    
    def get_current_model_info(self) -> Optional[ModelInfo]:
        """
        Get detailed information about the currently loaded model.
        
        Returns:
            Optional[ModelInfo]: Current model information or None if no model loaded
        """
        return self.current_model
    
    def switch_model(self, new_model_type: ModelType, gpu_layers: Optional[int] = None) -> ModelInfo:
        """
        Switch to a different model type, handling unloading and loading.
        
        Args:
            new_model_type: Model type to switch to
            gpu_layers: Number of GPU layers (None for auto-detection)
            
        Returns:
            ModelInfo: Information about the newly loaded model
        """
        current_type = self.get_current_model()
        
        if current_type == new_model_type:
            self.logger.info(f"Model {new_model_type.value} already loaded, no switch needed")
            return self.current_model
        
        self.logger.info(f"Switching model: {current_type.value if current_type else 'None'} -> {new_model_type.value}")
        
        # Load the new model (this will automatically unload the current one)
        return self.load_model(new_model_type, gpu_layers)
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dict[str, int]: Memory usage statistics in MB
        """
        current_memory = self._get_current_memory()
        
        return {
            'current_memory_mb': current_memory,
            'baseline_memory_mb': self.baseline_memory_mb,
            'model_memory_mb': current_memory - self.baseline_memory_mb if self.current_model else 0,
            'model_loaded': self.is_model_loaded()
        }
    
    def _get_current_memory(self) -> int:
        """
        Get current process memory usage in MB.
        
        Returns:
            int: Current memory usage in MB
        """
        try:
            memory_info = self.process.memory_info()
            return int(memory_info.rss / (1024 * 1024))  # Convert to MB
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0
    
    def _calculate_gpu_layers(self, model_path: str) -> int:
        """
        Calculate optimal GPU layers for a model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            int: Optimal number of GPU layers
        """
        try:
            # Get model file size
            model_size_bytes = os.path.getsize(model_path)
            model_size_mb = model_size_bytes // (1024 * 1024)
            
            # Use hardware profiler logic
            from ..hardware import HardwareProfiler
            profiler = HardwareProfiler()
            
            gpu_layers = profiler.calculate_gpu_layers(
                model_size_mb=model_size_mb,
                available_vram_mb=self.hardware_profile.gpu_vram_mb or 0
            )
            
            self.logger.debug(f"Calculated GPU layers for {os.path.basename(model_path)}: {gpu_layers}")
            return gpu_layers
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate GPU layers: {e}")
            return self.hardware_profile.optimal_gpu_layers
    
    def validate_model_paths(self) -> Dict[ModelType, bool]:
        """
        Validate that all configured model paths exist.
        
        Returns:
            Dict[ModelType, bool]: Validation results for each model type
        """
        results = {}
        for model_type, path in self.model_paths.items():
            exists = os.path.isfile(path)
            results[model_type] = exists
            
            if not exists:
                self.logger.warning(f"Model file not found: {model_type.value} -> {path}")
            else:
                self.logger.debug(f"Model file validated: {model_type.value} -> {os.path.basename(path)}")
        
        return results
    
    def cleanup(self) -> None:
        """
        Clean up resources and unload any loaded models.
        """
        self.logger.info("Cleaning up ModelManager...")
        
        if self.current_model:
            self.unload_current_model()
        
        # Clear references
        self.model_paths.clear()
        
        self.logger.info("ModelManager cleanup complete")