"""
Ollama wrapper for AURA-Engine-Core.
Adapts the llama.cpp interface to work with Ollama.
"""

from .wrapper import OllamaWrapper, OllamaOutput
from .cache_manager import ModelCacheManager

__all__ = ['OllamaWrapper', 'OllamaOutput', 'ModelCacheManager']
