"""
Integration tests for Phase 1 functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from aura_engine.engine import InferenceEngine
from aura_engine.models import EngineConfig, HardwareProfile, ModelType
from aura_engine.cli import parse_arguments, validate_inputs, create_engine_config


class TestPhase1Integration:
    """Integration test suite for Phase 1 end-to-end functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EngineConfig(
            llama_cpp_path="llama.cpp",
            max_tokens=256,
            temperature=0.7,
            enable_benchmarking=False  # Disable for testing
        )
    
    @patch('aura_engine.hardware.profiler.HardwareProfiler.get_hardware_profile')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper._validate_binary')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper.run_inference')
    @patch('os.path.isfile')
    def test_complete_inference_workflow(self, mock_isfile, mock_run_inference, 
                                       mock_validate_binary, mock_get_profile):
        """Test complete inference workflow from engine initialization to result."""
        # Mock hardware profile
        hardware_profile = HardwareProfile(
            system_ram_mb=16384,
            gpu_vram_mb=8192,
            gpu_name="RTX 4070",
            optimal_gpu_layers=25,
            cpu_cores=8
        )
        mock_get_profile.return_value = hardware_profile
        
        # Mock binary validation
        mock_validate_binary.return_value = None
        
        # Mock model file existence
        mock_isfile.return_value = True
        
        # Mock inference result
        from aura_engine.llama_wrapper.wrapper import ProcessedOutput
        mock_inference_result = ProcessedOutput(
            response="This is a test response from the model.",
            tokens_generated=12,
            model_load_time=2.5,
            inference_time=3.2,
            error_message=None
        )
        mock_run_inference.return_value = mock_inference_result
        
        # Create and initialize engine
        engine = InferenceEngine(self.config)
        engine.initialize()
        
        # Verify initialization
        assert engine.is_initialized
        assert engine.hardware_profile == hardware_profile
        
        # Process a prompt
        result = engine.process_prompt(
            prompt="Test prompt",
            model_path="test_model.gguf"
        )
        
        # Verify result structure
        assert result.response == "This is a test response from the model."
        assert result.model_used == ModelType.GENERAL
        assert result.rag_context is None
        assert result.metrics.tokens_per_second > 0
        assert result.metrics.model_load_time_s == 2.5
        
        # Verify inference was called with correct parameters
        mock_run_inference.assert_called_once()
        call_args = mock_run_inference.call_args
        assert call_args[1]['model_path'] == "test_model.gguf"
        assert call_args[1]['prompt'] == "Test prompt"
        assert call_args[1]['gpu_layers'] == 25  # From hardware profile
    
    @patch('aura_engine.hardware.profiler.HardwareProfiler.get_hardware_profile')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper._validate_binary')
    def test_engine_initialization_cpu_only(self, mock_validate_binary, mock_get_profile):
        """Test engine initialization with CPU-only system."""
        # Mock CPU-only hardware profile
        hardware_profile = HardwareProfile(
            system_ram_mb=8192,
            gpu_vram_mb=None,
            gpu_name=None,
            optimal_gpu_layers=0,
            cpu_cores=4
        )
        mock_get_profile.return_value = hardware_profile
        mock_validate_binary.return_value = None
        
        # Initialize engine
        engine = InferenceEngine(self.config)
        engine.initialize()
        
        # Verify CPU-only configuration
        assert engine.hardware_profile.gpu_vram_mb is None
        assert engine.hardware_profile.optimal_gpu_layers == 0
        assert engine.is_initialized
    
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper._validate_binary')
    def test_engine_not_initialized_error(self, mock_validate_binary):
        """Test error when processing prompt without initialization."""
        mock_validate_binary.return_value = None
        
        engine = InferenceEngine(self.config)
        
        with pytest.raises(RuntimeError, match="Engine must be initialized"):
            engine.process_prompt("test", "model.gguf")
    
    @patch('aura_engine.hardware.profiler.HardwareProfiler.get_hardware_profile')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper._validate_binary')
    def test_model_file_not_found_error(self, mock_validate_binary, mock_get_profile):
        """Test error when model file is not found."""
        mock_get_profile.return_value = HardwareProfile(8192, None, None, 0, 4)
        mock_validate_binary.return_value = None
        
        engine = InferenceEngine(self.config)
        engine.initialize()
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            engine.process_prompt("test", "nonexistent_model.gguf")
    
    @patch('sys.argv', ['main.py', 'Test prompt for CLI'])
    def test_cli_argument_parsing(self):
        """Test CLI argument parsing with basic prompt."""
        args = parse_arguments()
        
        assert args.prompt == "Test prompt for CLI"
        assert args.model_path == "models/model.gguf"  # Default
        assert args.max_tokens == 512  # Default
        assert args.temperature == 0.7  # Default
        assert args.log_level == "INFO"  # Default
    
    @patch('sys.argv', ['main.py', '--max-tokens', '1024', '--temperature', '0.5', 
                       '--gpu-layers', '30', 'Advanced prompt'])
    def test_cli_argument_parsing_advanced(self):
        """Test CLI argument parsing with advanced options."""
        args = parse_arguments()
        
        assert args.prompt == "Advanced prompt"
        assert args.max_tokens == 1024
        assert args.temperature == 0.5
        assert args.gpu_layers == 30
    
    def test_cli_input_validation_success(self):
        """Test successful CLI input validation."""
        from argparse import Namespace
        
        args = Namespace(
            prompt="Valid prompt",
            temperature=0.7,
            max_tokens=512,
            gpu_layers=25
        )
        
        assert validate_inputs(args) is True
    
    def test_cli_input_validation_empty_prompt(self):
        """Test CLI input validation with empty prompt."""
        from argparse import Namespace
        
        args = Namespace(
            prompt="",
            temperature=0.7,
            max_tokens=512,
            gpu_layers=25
        )
        
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            validate_inputs(args)
    
    def test_cli_input_validation_invalid_temperature(self):
        """Test CLI input validation with invalid temperature."""
        from argparse import Namespace
        
        args = Namespace(
            prompt="Valid prompt",
            temperature=3.0,  # Invalid: > 2.0
            max_tokens=512,
            gpu_layers=25
        )
        
        with pytest.raises(ValueError, match="Temperature must be between"):
            validate_inputs(args)
    
    def test_engine_config_creation(self):
        """Test engine configuration creation from CLI args."""
        from argparse import Namespace
        
        args = Namespace(
            llama_cpp_path="custom/llama.cpp",
            max_tokens=1024,
            temperature=0.8,
            no_benchmark=True,
            log_level="DEBUG"
        )
        
        config = create_engine_config(args)
        
        assert config.llama_cpp_path == "custom/llama.cpp"
        assert config.max_tokens == 1024
        assert config.temperature == 0.8
        assert config.enable_benchmarking is False
        assert config.log_level == "DEBUG"
    
    @patch('aura_engine.hardware.profiler.HardwareProfiler.get_hardware_profile')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper._validate_binary')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper.run_inference')
    @patch('os.path.isfile')
    @patch('os.path.getsize')
    def test_gpu_layer_calculation_with_model_size(self, mock_getsize, mock_isfile, 
                                                  mock_run_inference, mock_validate_binary, 
                                                  mock_get_profile):
        """Test GPU layer calculation based on actual model file size."""
        # Mock hardware profile with GPU
        hardware_profile = HardwareProfile(
            system_ram_mb=32768,
            gpu_vram_mb=24576,  # 24GB VRAM
            gpu_name="RTX 4090",
            optimal_gpu_layers=35,
            cpu_cores=16
        )
        mock_get_profile.return_value = hardware_profile
        mock_validate_binary.return_value = None
        mock_isfile.return_value = True
        
        # Mock large model file (13GB)
        mock_getsize.return_value = 13 * 1024 * 1024 * 1024  # 13GB in bytes
        
        # Mock inference
        from aura_engine.llama_wrapper.wrapper import ProcessedOutput
        mock_run_inference.return_value = ProcessedOutput(
            "response", 10, 2.0, 3.0, None
        )
        
        # Initialize engine and process prompt
        engine = InferenceEngine(self.config)
        engine.initialize()
        
        result = engine.process_prompt("test", "large_model.gguf")
        
        # Verify GPU layers were recalculated for large model
        call_args = mock_run_inference.call_args
        gpu_layers_used = call_args[1]['gpu_layers']
        
        # Should use fewer layers for larger model
        assert gpu_layers_used <= 20  # Should be conservative for 13GB model


if __name__ == "__main__":
    pytest.main([__file__])