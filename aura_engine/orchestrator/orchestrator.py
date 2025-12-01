"""
Main model orchestrator that combines routing and model management.
"""

import logging
from typing import Optional, Dict

from ..models import ModelType, HardwareProfile, EngineConfig
from ..llama_wrapper import LlamaWrapper
from .router import PromptRouter
from .model_manager import ModelManager


class ModelOrchestrator:
    """
    Main orchestrator that handles intelligent model routing and management.
    Combines prompt analysis with efficient model switching and memory management.
    """
    
    def __init__(self, llama_wrapper: LlamaWrapper, hardware_profile: HardwareProfile, 
                 config: EngineConfig):
        self.llama_wrapper = llama_wrapper
        self.hardware_profile = hardware_profile
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.router = PromptRouter()
        self.model_manager = ModelManager(llama_wrapper, hardware_profile)
        
        # State tracking
        self.is_initialized = False
        
        self.logger.info(f"ModelOrchestrator initialized with performance tier: {hardware_profile.performance_tier}")
    
    def get_tier_info(self) -> dict:
        """
        Get information about the current performance tier and available models.
        
        Returns:
            dict: Tier information including available models
        """
        return {
            "performance_tier": self.hardware_profile.performance_tier,
            "available_categories": self.model_manager.model_catalog.get_available_categories(),
            "text_models": self.model_manager.get_available_models("text"),
            "coding_models": self.model_manager.get_available_models("coding"),
            "mathematics_models": self.model_manager.get_available_models("mathematics")
        }
    
    def get_recommended_model_for_category(self, category: str = "text"):
        """
        Get the recommended model for a specific category based on current tier.
        
        Args:
            category: Model category ('text', 'coding', 'mathematics')
            
        Returns:
            ModelSpec or None if not found
        """
        return self.model_manager.get_recommended_model(category)
    
    def auto_configure_models(self, download_dir: str = "models") -> Optional[Dict]:
        """
        Automatically configure models for the current performance tier.
        
        Args:
            download_dir: Directory to download models to
            
        Returns:
            Dict mapping categories to model paths, or None if failed
        """
        if not hasattr(self.model_manager, 'auto_configure_for_tier'):
            self.logger.warning("Auto-configuration not available in current model manager")
            return None
        
        return self.model_manager.auto_configure_for_tier(download_dir)
    
    def initialize(self, model_paths: Dict[ModelType, str]) -> None:
        """
        Initialize the orchestrator with model paths.
        
        Args:
            model_paths: Dictionary mapping model types to file paths
        """
        self.logger.info("Initializing ModelOrchestrator...")
        
        # Configure model paths
        self.model_manager.configure_model_paths(model_paths)
        
        # Validate all model files
        validation_results = self.model_manager.validate_model_paths()
        
        missing_models = [model_type.value for model_type, exists in validation_results.items() if not exists]
        if missing_models:
            raise FileNotFoundError(f"Missing model files: {missing_models}")
        
        self.is_initialized = True
        self.logger.info("ModelOrchestrator initialization complete")
    
    def process_prompt(self, prompt: str, force_model_type: Optional[ModelType] = None,
                      gpu_layers_override: Optional[int] = None) -> tuple[ModelType, str]:
        """
        Process a prompt by routing to the appropriate model and executing inference.
        
        Args:
            prompt: Input prompt to process
            force_model_type: Force a specific model type (bypass routing)
            gpu_layers_override: Override GPU layers calculation
            
        Returns:
            tuple[ModelType, str]: (model_type_used, model_path)
            
        Raises:
            RuntimeError: If orchestrator is not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("ModelOrchestrator must be initialized before processing prompts")
        
        # Determine which model to use
        if force_model_type:
            selected_model = force_model_type
            self.logger.info(f"Using forced model type: {selected_model.value}")
        else:
            selected_model = self.router.analyze_prompt(prompt)
            self.logger.info(f"Router selected model: {selected_model.value}")
        
        # Switch to the selected model if needed
        current_model = self.model_manager.get_current_model()
        
        if current_model != selected_model:
            self.logger.info(f"Model switch required: {current_model.value if current_model else 'None'} -> {selected_model.value}")
            
            # Log memory usage before switch
            memory_before = self.model_manager.get_memory_usage()
            self.logger.info(f"Memory before switch: {memory_before['current_memory_mb']}MB")
            
            # Perform model switch
            model_info = self.model_manager.switch_model(selected_model, gpu_layers_override)
            
            # Log memory usage after switch
            memory_after = self.model_manager.get_memory_usage()
            self.logger.info(f"Memory after switch: {memory_after['current_memory_mb']}MB")
            self.logger.info(f"Model switch complete: {selected_model.value} loaded in {model_info.load_time:.2f}s")
            
        else:
            self.logger.info(f"Model {selected_model.value} already loaded, no switch needed")
            model_info = self.model_manager.get_current_model_info()
        
        return selected_model, model_info.model_path
    
    def get_routing_explanation(self, prompt: str) -> Dict:
        """
        Get detailed explanation of routing decision for debugging.
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            Dict: Detailed routing analysis
        """
        return self.router.explain_routing_decision(prompt)
    
    def get_current_model_info(self) -> Optional[Dict]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Optional[Dict]: Current model information or None if no model loaded
        """
        model_info = self.model_manager.get_current_model_info()
        if not model_info:
            return None
        
        return {
            'model_type': model_info.model_type.value,
            'model_path': model_info.model_path,
            'load_time': model_info.load_time,
            'memory_usage_mb': model_info.memory_usage_mb,
            'gpu_layers': model_info.gpu_layers
        }
    
    def get_memory_usage(self) -> Dict:
        """
        Get current memory usage statistics.
        
        Returns:
            Dict: Memory usage statistics
        """
        return self.model_manager.get_memory_usage()
    
    def unload_current_model(self) -> None:
        """
        Unload the currently loaded model to free memory.
        """
        self.model_manager.unload_current_model()
    
    def add_custom_routing_rule(self, model_type: ModelType, keywords: list, 
                               patterns: list = None, weight: float = 1.0) -> None:
        """
        Add a custom routing rule to the prompt router.
        
        Args:
            model_type: Target model type
            keywords: List of keywords to match
            patterns: List of regex patterns to match
            weight: Weight for this rule
        """
        self.router.add_custom_rule(model_type, keywords, patterns, weight)
        self.logger.info(f"Added custom routing rule for {model_type.value}")
    
    def get_available_models(self) -> Dict[ModelType, str]:
        """
        Get the configured model paths.
        
        Returns:
            Dict[ModelType, str]: Available model types and their paths
        """
        return self.model_manager.model_paths.copy()
    
    def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate the orchestrator configuration.
        
        Returns:
            Dict[str, bool]: Validation results
        """
        results = {
            'initialized': self.is_initialized,
            'models_configured': len(self.model_manager.model_paths) > 0,
            'all_models_exist': True
        }
        
        if results['models_configured']:
            model_validation = self.model_manager.validate_model_paths()
            results['all_models_exist'] = all(model_validation.values())
            results['model_validation'] = model_validation
        
        return results
    
    def shutdown(self) -> None:
        """
        Shutdown the orchestrator and clean up resources.
        """
        self.logger.info("Shutting down ModelOrchestrator...")
        
        # Clean up model manager
        self.model_manager.cleanup()
        
        # Reset state
        self.is_initialized = False
        
        self.logger.info("ModelOrchestrator shutdown complete")