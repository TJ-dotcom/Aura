"""
Ollama wrapper for AURA-Engine-Core.
Adapts the llama.cpp interface to work with Ollama.
"""

import logging
import subprocess
import json
import requests
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..models import EngineConfig, ModelType
from .cache_manager import ModelCacheManager


@dataclass
class OllamaOutput:
    """Output from Ollama inference."""
    response: str
    tokens_generated: int = 0
    model_load_time: Optional[float] = None
    error_message: Optional[str] = None
    native_tps: Optional[float] = None  # Store Ollama's calculated TPS


class OllamaWrapper:
    """
    Wrapper for Ollama that provides llama.cpp-compatible interface.
    Enhanced with model caching to eliminate cold start penalties.
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.ollama_host = "http://127.0.0.1:11435"  # GPU-enabled server
        
        # DISABLED: Model cache manager causing CPU overload
        # self.cache_manager = ModelCacheManager(
        #     ollama_host=self.ollama_host,
        #     max_cached_models=3  # Adjust based on VRAM capacity
        # )
        self.cache_manager = None
        
        self._validate_ollama()
    
    def initialize_model_cache(self, model_map: Dict[str, str]) -> None:
        """Initialize model cache by preloading core models."""
        # DISABLED: Cache manager causing CPU overload and timeouts
        self.logger.info("🚫 Model cache disabled to prevent CPU overload")
        return
        
        self._validate_ollama()
    
    def _validate_ollama(self) -> None:
        """Validate that Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_host}/api/version", timeout=5)
            if response.status_code == 200:
                version_info = response.json()
                self.logger.info(f"Connected to Ollama version: {version_info.get('version', 'unknown')}")
            else:
                raise ConnectionError(f"Ollama returned status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Could not connect to Ollama: {e}")
            self.logger.info("Make sure Ollama is running with: 'ollama serve'")
            # Don't raise exception here - we'll try to start it later
    
    def list_available_models(self) -> Dict[str, Any]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                return response.json()
            return {"models": []}
        except requests.exceptions.RequestException:
            return {"models": []}
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model in Ollama."""
        try:
            self.logger.info(f"Pulling model {model_name} from Ollama...")
            response = requests.post(
                f"{self.ollama_host}/api/pull",
                json={"name": model_name},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    progress = json.loads(line.decode('utf-8'))
                    if 'status' in progress:
                        self.logger.info(f"Pull progress: {progress['status']}")
                    if progress.get('status') == 'success':
                        return True
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def run_inference(self, model_path: str = None, prompt: str = "", 
                     gpu_layers: int = None, model_name: str = None,
                     model_type: Optional[ModelType] = None) -> OllamaOutput:
        """
        Run inference using Ollama with intelligent model caching.
        
        Args:
            model_path: Ignored for Ollama (uses model_name instead)
            prompt: Input prompt
            gpu_layers: Ignored for Ollama (it handles GPU automatically)
            model_name: Ollama model name (e.g., 'deepseek-coder:6.7b')
            model_type: Model type for cache management
        """
        # If model_name not provided, try to extract from model_path
        if not model_name:
            if model_path:
                # Convert common model filenames to Ollama model names
                model_name = self._path_to_ollama_model(model_path)
            else:
                model_name = "llama2"  # Default model
        
        print(f"DEBUG: Starting Ollama inference with model: {model_name}")
        self.logger.info(f"Running Ollama inference with model: {model_name}")
        
        start_time = time.time()
        
        try:
            # DISABLED: Cache manager to prevent CPU overload
            # if model_type:
            #     self.cache_manager.ensure_model_loaded(model_name, model_type)
            
            # First ensure the model is available
            available_models = self.list_available_models()
            model_names = [m['name'].split(':')[0] for m in available_models.get('models', [])]

            if model_name not in model_names:
                self.logger.info(f"Model {model_name} not found locally. Attempting to pull...")
                if not self.pull_model(model_name):
                    return OllamaOutput(
                        response="",
                        error_message=f"Failed to pull model {model_name}"
                    )            # Run inference with streaming
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": True,  # Enable streaming
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                        "repeat_penalty": 1.1,  # Prevent repetition
                        "top_k": 40,
                        "top_p": 0.9,
                        "num_gpu": 1,  # Force GPU usage
                        "gpu_layers": 999,  # Force all layers on GPU
                        "use_mmap": False,  # Disable memory mapping to prevent CPU usage
                        "use_mlock": True,  # Lock GPU memory
                        "numa": False,  # Disable NUMA for GPU-only
                        "low_vram": False,  # We have sufficient VRAM
                        "stop": ["###", "```\n\n", "\n\n---", "\n\nUser:", "\n\nAssistant:"]  # Better stop conditions
                    }
                },
                timeout=120,
                stream=True  # Enable streaming response
            )
            
            if response.status_code == 200:
                # Process streaming response
                full_response = ""
                first_token_time = None
                actual_tokens = 0
                total_duration_ns = 0
                eval_count = 0
                eval_duration_ns = 0
                
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line.decode('utf-8'))
                            
                            # Check for response content
                            if 'response' in chunk_data:
                                token_text = chunk_data['response']
                                full_response += token_text
                                
                                # Record first token time
                                if first_token_time is None and token_text.strip():
                                    first_token_time = time.time()
                                
                                # Print token as it arrives (real-time streaming)
                                print(token_text, end='', flush=True)
                            
                            # Check for completion and extract ACTUAL token metrics from Ollama
                            if chunk_data.get('done', False):
                                # Use Ollama's native token metrics for accurate TPS
                                eval_count = chunk_data.get('eval_count', 0)
                                eval_duration_ns = chunk_data.get('eval_duration', 0)
                                total_duration_ns = chunk_data.get('total_duration', 0)
                                actual_tokens = eval_count  # Real token count from Ollama
                                break
                                
                        except json.JSONDecodeError:
                            continue  # Skip malformed JSON
                
                inference_time = time.time() - start_time
                
                # Use Ollama's native TPS calculation for maximum accuracy
                if eval_duration_ns > 0 and eval_count > 0:
                    # Convert nanoseconds to seconds for TPS calculation
                    eval_time_seconds = eval_duration_ns / 1e9
                    actual_tps = eval_count / eval_time_seconds
                    self.logger.debug(f"Native Ollama TPS: {actual_tps:.2f} ({eval_count} tokens in {eval_time_seconds:.2f}s)")
                else:
                    # Fallback calculation if Ollama metrics unavailable
                    if first_token_time and actual_tokens > 0:
                        generation_time = inference_time - (first_token_time - start_time)
                        actual_tps = actual_tokens / generation_time if generation_time > 0 else 0
                    else:
                        # Emergency fallback: use word count as rough estimate
                        word_count = len(full_response.split()) if full_response else 0
                        actual_tokens = max(actual_tokens, word_count)  # Use better estimate
                        actual_tps = actual_tokens / inference_time if inference_time > 0 else 0
                
                return OllamaOutput(
                    response=full_response,
                    tokens_generated=actual_tokens,
                    model_load_time=inference_time,
                    error_message=None,
                    native_tps=actual_tps  # Include the native TPS calculation
                )
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                return OllamaOutput(
                    response="",
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"Ollama inference failed: {e}"
            self.logger.error(error_msg)
            return OllamaOutput(
                response="",
                error_message=error_msg
            )
    
    def _path_to_ollama_model(self, model_path: str) -> str:
        """Convert a model file path to an Ollama model name."""
        import os
        filename = os.path.basename(model_path).lower()
        
        # Map common model patterns to Ollama model names
        if 'llama-2' in filename and '7b' in filename:
            return 'llama2'
        elif 'llama-2' in filename and '13b' in filename:
            return 'llama2:13b'
        elif 'codellama' in filename:
            if '7b' in filename:
                return 'codellama'
            elif '13b' in filename:
                return 'codellama:13b'
        elif 'mistral' in filename:
            return 'mistral'
        elif 'tinyllama' in filename:
            return 'tinyllama'
        elif 'phi' in filename:
            return 'phi'
        else:
            # Default to llama2 for unknown models
            return 'llama2'
    
    def get_model_info(self, model_name: str = "llama2") -> Dict[str, Any]:
        """Get information about a specific model."""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/show",
                json={"name": model_name}
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except requests.exceptions.RequestException:
            return {}

__all__ = ['OllamaWrapper', 'OllamaOutput']
