"""
Unit tests for model orchestrator functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from aura_engine.orchestrator.orchestrator import ModelOrchestrator
from aura_engine.orchestrator.model_manager import ModelInfo
from aura_engine.models import ModelType, HardwareProfile, EngineConfig
from aura_engine.llama_wrapper import LlamaWrapper


class TestModelOrchestrator:
    """Test suite for ModelOrchestrator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_llama_wrapper = Mock(spec=LlamaWrapper)
        self.hardware_profile = HardwareProfile(
            system_ram_mb=16384,
            gpu_vram_mb=8192,
            gpu_name="RTX 4070",
            optimal_gpu_layers=25,
            cpu_cores=8
        )
        self.config = EngineConfig()
        
        # Create orchestrator
        self.orchestrator = ModelOrchestrator(
            self.mock_llama_wrapper,
            self.hardware_profile,
            self.config
        )
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.llama_wrapper == self.mock_llama_wrapper
        assert self.orchestrator.hardware_profile == self.hardware_profile
        assert self.orchestrator.config == self.config
        assert not self.orchestrator.is_initialized
        assert self.orchestrator.router is not None
        assert self.orchestrator.model_manager is not None
    
    def test_initialize_success(self):
        """Test successful orchestrator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary model files
            model_paths = {}
            for model_type in [ModelType.CODER, ModelType.WRITER, ModelType.GENERAL]:
                path = os.path.join(temp_dir, f"{model_type.value}.gguf")
                with open(path, 'w') as f:
                    f.write("mock model")
                model_paths[model_type] = path
            
            self.orchestrator.initialize(model_paths)
            
            assert self.orchestrator.is_initialized
            assert self.orchestrator.model_manager.model_paths == model_paths
    
    def test_initialize_missing_models(self):
        """Test initialization with missing model files."""
        model_paths = {
            ModelType.CODER: "/nonexistent/coder.gguf",
            ModelType.WRITER: "/nonexistent/writer.gguf"
        }
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            self.orchestrator.initialize(model_paths)
        
        assert not self.orchestrator.is_initialized
    
    @patch.object(ModelOrchestrator, 'initialize')
    def test_process_prompt_not_initialized(self, mock_init):
        """Test processing prompt without initialization."""
        with pytest.raises(RuntimeError, match="ModelOrchestrator must be initialized"):
            self.orchestrator.process_prompt("test prompt")
    
    def test_process_prompt_with_routing(self):
        """Test prompt processing with automatic routing."""
        # Setup initialized orchestrator
        self.orchestrator.is_initialized = True
        
        # Mock router to return CODER
        with patch.object(self.orchestrator.router, 'analyze_prompt', return_value=ModelType.CODER):
            # Mock model manager
            mock_model_info = ModelInfo(
                model_type=ModelType.CODER,
                model_path="/path/to/coder.gguf",
                load_time=2.0,
                memory_usage_mb=500,
                gpu_layers=25
            )
            
            with patch.object(self.orchestrator.model_manager, 'get_current_model', return_value=None):
                with patch.object(self.orchestrator.model_manager, 'switch_model', return_value=mock_model_info):
                    with patch.object(self.orchestrator.model_manager, 'get_memory_usage', return_value={'current_memory_mb': 1500}):
                        
                        model_type, model_path = self.orchestrator.process_prompt("Write a Python function")
                        
                        assert model_type == ModelType.CODER
                        assert model_path == "/path/to/coder.gguf"
    
    def test_process_prompt_with_forced_model(self):
        """Test prompt processing with forced model type."""
        # Setup initialized orchestrator
        self.orchestrator.is_initialized = True
        
        # Mock model manager
        mock_model_info = ModelInfo(
            model_type=ModelType.WRITER,
            model_path="/path/to/writer.gguf",
            load_time=1.5,
            memory_usage_mb=400,
            gpu_layers=20
        )
        
        with patch.object(self.orchestrator.model_manager, 'get_current_model', return_value=None):
            with patch.object(self.orchestrator.model_manager, 'switch_model', return_value=mock_model_info):
                with patch.object(self.orchestrator.model_manager, 'get_memory_usage', return_value={'current_memory_mb': 1400}):
                    
                    model_type, model_path = self.orchestrator.process_prompt(
                        "Any prompt", 
                        force_model_type=ModelType.WRITER
                    )
                    
                    assert model_type == ModelType.WRITER
                    assert model_path == "/path/to/writer.gguf"
    
    def test_process_prompt_no_model_switch_needed(self):
        """Test prompt processing when target model is already loaded."""
        # Setup initialized orchestrator
        self.orchestrator.is_initialized = True
        
        # Mock router to return CODER
        with patch.object(self.orchestrator.router, 'analyze_prompt', return_value=ModelType.CODER):
            # Mock model manager - CODER already loaded
            mock_model_info = ModelInfo(
                model_type=ModelType.CODER,
                model_path="/path/to/coder.gguf",
                load_time=2.0,
                memory_usage_mb=500,
                gpu_layers=25
            )
            
            with patch.object(self.orchestrator.model_manager, 'get_current_model', return_value=ModelType.CODER):
                with patch.object(self.orchestrator.model_manager, 'get_current_model_info', return_value=mock_model_info):
                    
                    model_type, model_path = self.orchestrator.process_prompt("Write a Python function")
                    
                    assert model_type == ModelType.CODER
                    assert model_path == "/path/to/coder.gguf"
    
    def test_get_routing_explanation(self):
        """Test routing explanation functionality."""
        mock_explanation = {
            'prompt': 'test prompt',
            'selected_model': ModelType.CODER,
            'rule_scores': {'coder': 2.5, 'writer': 1.0, 'general': 0.5},
            'matched_keywords': {'coder': ['python'], 'writer': [], 'general': []},
            'matched_patterns': {'coder': [], 'writer': [], 'general': []}
        }
        
        with patch.object(self.orchestrator.router, 'explain_routing_decision', return_value=mock_explanation):
            result = self.orchestrator.get_routing_explanation("test prompt")
            
            assert result == mock_explanation
    
    def test_get_current_model_info_with_model(self):
        """Test getting current model info when model is loaded."""
        mock_model_info = ModelInfo(
            model_type=ModelType.CODER,
            model_path="/path/to/coder.gguf",
            load_time=2.0,
            memory_usage_mb=500,
            gpu_layers=25
        )
        
        with patch.object(self.orchestrator.model_manager, 'get_current_model_info', return_value=mock_model_info):
            result = self.orchestrator.get_current_model_info()
            
            expected = {
                'model_type': 'coder',
                'model_path': '/path/to/coder.gguf',
                'load_time': 2.0,
                'memory_usage_mb': 500,
                'gpu_layers': 25
            }
            assert result == expected
    
    def test_get_current_model_info_no_model(self):
        """Test getting current model info when no model is loaded."""
        with patch.object(self.orchestrator.model_manager, 'get_current_model_info', return_value=None):
            result = self.orchestrator.get_current_model_info()
            
            assert result is None
    
    def test_get_memory_usage(self):
        """Test memory usage reporting."""
        mock_usage = {
            'current_memory_mb': 1500,
            'baseline_memory_mb': 1000,
            'model_memory_mb': 500,
            'model_loaded': True
        }
        
        with patch.object(self.orchestrator.model_manager, 'get_memory_usage', return_value=mock_usage):
            result = self.orchestrator.get_memory_usage()
            
            assert result == mock_usage
    
    def test_unload_current_model(self):
        """Test unloading current model."""
        with patch.object(self.orchestrator.model_manager, 'unload_current_model') as mock_unload:
            self.orchestrator.unload_current_model()
            
            mock_unload.assert_called_once()
    
    def test_add_custom_routing_rule(self):
        """Test adding custom routing rule."""
        keywords = ["tensorflow", "pytorch"]
        patterns = [r"\\bML\\b", r"\\bAI\\b"]
        
        with patch.object(self.orchestrator.router, 'add_custom_rule') as mock_add_rule:
            self.orchestrator.add_custom_routing_rule(
                ModelType.CODER, 
                keywords, 
                patterns, 
                weight=1.5
            )
            
            mock_add_rule.assert_called_once_with(ModelType.CODER, keywords, patterns, 1.5)
    
    def test_get_available_models(self):
        """Test getting available models."""
        mock_paths = {
            ModelType.CODER: "/path/to/coder.gguf",
            ModelType.WRITER: "/path/to/writer.gguf"
        }
        
        self.orchestrator.model_manager.model_paths = mock_paths
        
        result = self.orchestrator.get_available_models()
        
        assert result == mock_paths
        # Verify it's a copy, not the original
        assert result is not self.orchestrator.model_manager.model_paths
    
    def test_validate_configuration_not_initialized(self):
        """Test configuration validation when not initialized."""
        result = self.orchestrator.validate_configuration()
        
        expected = {
            'initialized': False,
            'models_configured': False,
            'all_models_exist': True
        }
        assert result == expected
    
    def test_validate_configuration_initialized(self):
        """Test configuration validation when initialized."""
        self.orchestrator.is_initialized = True
        
        mock_validation = {
            ModelType.CODER: True,
            ModelType.WRITER: False
        }
        
        with patch.object(self.orchestrator.model_manager, 'validate_model_paths', return_value=mock_validation):
            self.orchestrator.model_manager.model_paths = {
                ModelType.CODER: "/path/to/coder.gguf",
                ModelType.WRITER: "/path/to/writer.gguf"
            }
            
            result = self.orchestrator.validate_configuration()
            
            expected = {
                'initialized': True,
                'models_configured': True,
                'all_models_exist': False,
                'model_validation': mock_validation
            }
            assert result == expected
    
    def test_shutdown(self):
        """Test orchestrator shutdown."""
        self.orchestrator.is_initialized = True
        
        with patch.object(self.orchestrator.model_manager, 'cleanup') as mock_cleanup:
            self.orchestrator.shutdown()
            
            mock_cleanup.assert_called_once()
            assert not self.orchestrator.is_initialized


if __name__ == "__main__":
    pytest.main([__file__])