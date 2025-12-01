"""
Model Cache Manager for AURA-Engine-Core.
Handles persistent model loading and intelligent model switching to minimize cold start penalties.
"""

import logging
import requests
import time
from typing import Dict, Optional, Set, List
from dataclasses import dataclass
from enum import Enum

from ..models import ModelType, HardwareProfile


@dataclass
class ModelCacheStatus:
    """Status of a cached model."""
    model_name: str
    model_type: ModelType
    is_loaded: bool
    vram_usage_mb: int
    last_used: float
    load_time_seconds: float


class ModelCacheManager:
    """
    Manages persistent model loading and intelligent switching to eliminate cold start penalties.
    """
    
    def __init__(self, ollama_host: str = "http://localhost:11434", 
                 max_cached_models: int = 3):
        self.logger = logging.getLogger(__name__)
        self.ollama_host = ollama_host
        self.max_cached_models = max_cached_models
        
        # Track loaded models
        self.cached_models: Dict[str, ModelCacheStatus] = {}
        self.model_priority: Dict[ModelType, int] = {
            ModelType.CODER: 3,    # Highest priority - complex models
            ModelType.MATH: 2,     # Medium priority  
            ModelType.WRITER: 1    # Lowest priority - lighter models
        }
        
        self.logger.info(f"Model Cache Manager initialized (max {max_cached_models} models)")
    
    def get_loaded_models(self) -> Dict[str, ModelCacheStatus]:
        """Get currently loaded models from Ollama."""
        try:
            response = requests.get(f"{self.ollama_host}/api/ps", timeout=5)
            if response.status_code == 200:
                data = response.json()
                current_models = {}
                
                for model_info in data.get('models', []):
                    model_name = model_info.get('name', '')
                    size_bytes = model_info.get('size', 0)
                    vram_mb = size_bytes / (1024 * 1024)  # Convert to MB
                    
                    # Determine model type based on name
                    model_type = self._infer_model_type(model_name)
                    
                    current_models[model_name] = ModelCacheStatus(
                        model_name=model_name,
                        model_type=model_type,
                        is_loaded=True,
                        vram_usage_mb=int(vram_mb),
                        last_used=time.time(),
                        load_time_seconds=0.0
                    )
                
                self.cached_models.update(current_models)
                return current_models
                
        except Exception as e:
            self.logger.error(f"Failed to get loaded models: {e}")
            return {}
    
    def _infer_model_type(self, model_name: str) -> ModelType:
        """Infer model type from model name."""
        name_lower = model_name.lower()
        
        if 'coder' in name_lower or 'code' in name_lower:
            return ModelType.CODER
        elif 'r1' in name_lower or 'math' in name_lower:
            return ModelType.MATH
        elif 'phi' in name_lower or 'writer' in name_lower:
            return ModelType.WRITER
        else:
            return ModelType.WRITER  # Default to writer
    
    def preload_core_models(self, model_map: Dict[str, str]) -> bool:
        """
        Preload the most important models at startup to eliminate cold starts.
        """
        self.logger.info("🚀 Starting core model preloading...")
        
        # Priority loading order: coder -> math -> writer
        priority_models = [
            (model_map.get('coder', ''), ModelType.CODER),
            (model_map.get('math', ''), ModelType.MATH),
            (model_map.get('writer', ''), ModelType.WRITER)
        ]
        
        loaded_count = 0
        for model_name, model_type in priority_models:
            if model_name and self._preload_single_model(model_name, model_type):
                loaded_count += 1
                
                # Check if we've hit VRAM limits
                if loaded_count >= self.max_cached_models:
                    self.logger.info(f"Reached maximum cached models limit ({self.max_cached_models})")
                    break
        
        self.logger.info(f"✅ Preloaded {loaded_count} core models")
        return loaded_count > 0
    
    def _preload_single_model(self, model_name: str, model_type: ModelType) -> bool:
        """Preload a single model."""
        try:
            self.logger.info(f"Loading {model_type.value} model: {model_name}")
            start_time = time.time()
            
            # Make a minimal request to trigger model loading
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hi",  # Minimal prompt
                    "stream": False,
                    "options": {"num_predict": 1}  # Generate only 1 token
                },
                timeout=60  # Allow time for model loading
            )
            
            load_time = time.time() - start_time
            
            if response.status_code == 200:
                # Update cache status
                self.cached_models[model_name] = ModelCacheStatus(
                    model_name=model_name,
                    model_type=model_type,
                    is_loaded=True,
                    vram_usage_mb=0,  # Will be updated by get_loaded_models()
                    last_used=time.time(),
                    load_time_seconds=load_time
                )
                
                self.logger.info(f"✅ Loaded {model_name} in {load_time:.1f}s")
                return True
            else:
                self.logger.error(f"Failed to load {model_name}: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error preloading {model_name}: {e}")
            return False
    
    def ensure_model_loaded(self, model_name: str, model_type: ModelType) -> bool:
        """
        Ensure a model is loaded, loading it if necessary.
        Implements intelligent cache management.
        """
        # Check if already loaded
        if model_name in self.cached_models and self.cached_models[model_name].is_loaded:
            self.cached_models[model_name].last_used = time.time()
            return True
        
        # Check cache capacity
        current_loaded = len([m for m in self.cached_models.values() if m.is_loaded])
        
        if current_loaded >= self.max_cached_models:
            # Need to evict least important model
            self._evict_least_important_model()
        
        # Load the requested model
        return self._preload_single_model(model_name, model_type)
    
    def _evict_least_important_model(self):
        """Evict the least important model based on priority and usage."""
        loaded_models = [(name, status) for name, status in self.cached_models.items() 
                        if status.is_loaded]
        
        if not loaded_models:
            return
        
        # Sort by priority (lower priority first) and last used (older first)
        def sort_key(item):
            name, status = item
            priority = self.model_priority.get(status.model_type, 0)
            return (priority, status.last_used)
        
        loaded_models.sort(key=sort_key)
        
        # Evict the least important model
        model_to_evict, status = loaded_models[0]
        
        self.logger.info(f"Evicting model {model_to_evict} to free VRAM")
        
        # Note: Ollama doesn't have explicit unload API, models are evicted automatically
        # We'll mark it as not loaded in our cache
        status.is_loaded = False
        
    def get_cache_status(self) -> Dict[str, Dict]:
        """Get detailed cache status for monitoring."""
        self.get_loaded_models()  # Refresh from Ollama
        
        status = {
            "total_cached": len(self.cached_models),
            "currently_loaded": len([m for m in self.cached_models.values() if m.is_loaded]),
            "max_capacity": self.max_cached_models,
            "models": {}
        }
        
        for name, cache_status in self.cached_models.items():
            status["models"][name] = {
                "type": cache_status.model_type.value,
                "loaded": cache_status.is_loaded,
                "vram_mb": cache_status.vram_usage_mb,
                "last_used_ago": time.time() - cache_status.last_used,
                "load_time": cache_status.load_time_seconds
            }
        
        return status


__all__ = ['ModelCacheManager', 'ModelCacheStatus']
