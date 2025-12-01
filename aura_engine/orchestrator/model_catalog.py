"""
Tiered model catalog for automatic model selection based on performance tier.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelSpec:
    """Specification for a model including its capabilities and requirements."""
    name: str
    url: str
    size_mb: int
    category: str  # text, coding, mathematics
    description: str


class ModelCatalog:
    """
    Tiered model catalog implementing the documented model selection strategy.
    Models are organized by performance tier and category.
    """
    
    def __init__(self):
        self._catalog = self._build_catalog()
    
    def _build_catalog(self) -> Dict[str, Dict[str, List[ModelSpec]]]:
        """Build the complete model catalog with tiered organization."""
        return {
            "high-performance": {
                "text": [
                    ModelSpec(
                        name="llama-2-13b-chat.q4_K_M.gguf",
                        url="https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.q4_K_M.gguf",
                        size_mb=7323,
                        category="text",
                        description="High-performance text generation and reasoning"
                    ),
                    ModelSpec(
                        name="mistral-7b-instruct-v0.2.q5_K_M.gguf",
                        url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGML/resolve/main/mistral-7b-instruct-v0.2.q5_K_M.gguf",
                        size_mb=4785,
                        category="text",
                        description="Advanced instruction following and reasoning"
                    )
                ],
                "coding": [
                    ModelSpec(
                        name="codellama-13b-instruct.q4_K_M.gguf",
                        url="https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGML/resolve/main/codellama-13b-instruct.q4_K_M.gguf",
                        size_mb=7323,
                        category="coding",
                        description="Advanced code generation and debugging"
                    ),
                    ModelSpec(
                        name="deepseek-coder-6.7b-instruct.q5_K_M.gguf",
                        url="https://huggingface.co/TheBloke/deepseek-coder-6.7b-instruct-GGML/resolve/main/deepseek-coder-6.7b-instruct.q5_K_M.gguf",
                        size_mb=4520,
                        category="coding",
                        description="Specialized coding model with multi-language support"
                    )
                ],
                "mathematics": [
                    ModelSpec(
                        name="mathstral-7b-v0.1.q5_K_M.gguf",
                        url="https://huggingface.co/bartowski/mathstral-7B-v0.1-GGUF/resolve/main/mathstral-7B-v0.1-Q5_K_M.gguf",
                        size_mb=4785,
                        category="mathematics",
                        description="Mathematical reasoning and problem solving"
                    )
                ]
            },
            "balanced": {
                "text": [
                    ModelSpec(
                        name="llama-2-7b-chat.q4_K_M.gguf",
                        url="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.q4_K_M.gguf",
                        size_mb=3831,
                        category="text",
                        description="Balanced text generation with good performance"
                    ),
                    ModelSpec(
                        name="mistral-7b-instruct-v0.2.q4_K_M.gguf",
                        url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGML/resolve/main/mistral-7b-instruct-v0.2.q4_K_M.gguf",
                        size_mb=3829,
                        category="text",
                        description="Efficient instruction following"
                    )
                ],
                "coding": [
                    ModelSpec(
                        name="codellama-7b-instruct.q4_K_M.gguf",
                        url="https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGML/resolve/main/codellama-7b-instruct.q4_K_M.gguf",
                        size_mb=3831,
                        category="coding",
                        description="Balanced coding assistance"
                    ),
                    ModelSpec(
                        name="starcoder2-7b.q4_K_M.gguf",
                        url="https://huggingface.co/bartowski/starcoder2-7b-GGUF/resolve/main/starcoder2-7b-Q4_K_M.gguf",
                        size_mb=4100,
                        category="coding",
                        description="Efficient code generation"
                    )
                ],
                "mathematics": [
                    ModelSpec(
                        name="mathstral-7b-v0.1.q4_K_M.gguf",
                        url="https://huggingface.co/bartowski/mathstral-7B-v0.1-GGUF/resolve/main/mathstral-7B-v0.1-Q4_K_M.gguf",
                        size_mb=3829,
                        category="mathematics",
                        description="Mathematical reasoning with balanced performance"
                    )
                ]
            },
            "high-efficiency": {
                "text": [
                    ModelSpec(
                        name="tinyllama-1.1b-chat-v1.0.q4_K_M.gguf",
                        url="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.q4_K_M.gguf",
                        size_mb=669,
                        category="text",
                        description="Ultra-efficient text generation for low-resource systems"
                    ),
                    ModelSpec(
                        name="phi-2.q4_K_M.gguf",
                        url="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
                        size_mb=1568,
                        category="text",
                        description="Small but capable reasoning model"
                    )
                ],
                "coding": [
                    ModelSpec(
                        name="starcoder2-3b.q4_K_M.gguf",
                        url="https://huggingface.co/bartowski/starcoder2-3b-GGUF/resolve/main/starcoder2-3b-Q4_K_M.gguf",
                        size_mb=1700,
                        category="coding",
                        description="Lightweight coding model"
                    ),
                    ModelSpec(
                        name="codegemma-2b.q4_K_M.gguf",
                        url="https://huggingface.co/bartowski/codegemma-2b-GGUF/resolve/main/codegemma-2b-Q4_K_M.gguf",
                        size_mb=1400,
                        category="coding",
                        description="Efficient code generation for resource-constrained systems"
                    )
                ],
                "mathematics": [
                    ModelSpec(
                        name="phi-2.q4_K_M.gguf",
                        url="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
                        size_mb=1568,
                        category="mathematics",
                        description="Small model with mathematical capabilities"
                    )
                ]
            }
        }
    
    def get_models_for_tier(self, tier: str, category: str = None) -> List[ModelSpec]:
        """
        Get all models for a specific tier and optional category.
        
        Args:
            tier: Performance tier ('high-performance', 'balanced', 'high-efficiency')
            category: Optional category filter ('text', 'coding', 'mathematics')
            
        Returns:
            List of ModelSpec objects matching the criteria
        """
        if tier not in self._catalog:
            raise ValueError(f"Unknown tier: {tier}")
        
        if category is None:
            # Return all models for the tier
            models = []
            for cat_models in self._catalog[tier].values():
                models.extend(cat_models)
            return models
        
        if category not in self._catalog[tier]:
            raise ValueError(f"Unknown category: {category}")
        
        return self._catalog[tier][category]
    
    def get_default_model(self, tier: str, category: str = "text") -> Optional[ModelSpec]:
        """
        Get the default (first) model for a tier and category.
        
        Args:
            tier: Performance tier
            category: Model category (defaults to 'text')
            
        Returns:
            Default ModelSpec or None if not found
        """
        try:
            models = self.get_models_for_tier(tier, category)
            return models[0] if models else None
        except ValueError:
            return None
    
    def get_available_tiers(self) -> List[str]:
        """Get list of available performance tiers."""
        return list(self._catalog.keys())
    
    def get_available_categories(self) -> List[str]:
        """Get list of available model categories."""
        return ["text", "coding", "mathematics"]
