"""
Integration tests for Phase 2 functionality (Model Orchestration).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from aura_engine.engine import InferenceEngine
from aura_engine.models import EngineConfig, ModelType, HardwareProfile
from aura_engine.orchestrator.model_manager import ModelInfo
from aura_engine.llama_wrapper.wrapper import ProcessedOutput


class TestPhase2Integration:
    """Integration test suite for Phase 2 model orchestration functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EngineConfig(
            llama_cpp_path="llama.cpp",
            max_tokens=256,
            temperature=0.7,
            enable_benchmarking=False  # Disable for testing
        )
        
        self.hardware_profile = HardwareProfile(
            system_ram_mb=16384,
            gpu_vram_mb=8192,
            gpu_name="RTX 4070",
            optimal_gpu_layers=25,
            cpu_cores=8
        )
    
    @patch('aura_engine.hardware.profiler.HardwareProfiler.get_hardware_profile')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper._validate_binary')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper.run_inference')
    def test_phase2_model_switching_workflow(self, mock_run_inference, mock_validate_binary, mock_get_profile):
        """Test complete Phase 2 workflow with model switching."""
        # Mock hardware profile
        mock_get_profile.return_value = self.hardware_profile
        mock_validate_binary.return_value = None
        
        # Create temporary model files
        with tempfile.TemporaryDirectory() as temp_dir:
            model_paths = {}
            for model_type in [ModelType.CODER, ModelType.WRITER, ModelType.GENERAL]:
                path = os.path.join(temp_dir, f"{model_type.value}.gguf")
                with open(path, 'w') as f:
                    f.write("mock model")
                model_paths[model_type] = path
            
            # Mock inference results
            coding_response = ProcessedOutput(
                response="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                tokens_generated=15,
                model_load_time=2.0,
                inference_time=3.0,
                error_message=None
            )
            
            writing_response = ProcessedOutput(
                response="Climate change is a pressing global issue that requires immediate attention...",
                tokens_generated=20,
                model_load_time=1.8,
                inference_time=2.5,
                error_message=None
            )
            
            mock_run_inference.side_effect = [coding_response, writing_response]
            
            # Initialize engine with Phase 2 models
            engine = InferenceEngine(self.config)
            engine.initialize(model_paths)
            
            # Verify Phase 2 initialization
            assert engine.model_orchestrator is not None
            assert engine.model_orchestrator.is_initialized
            
            # Test coding prompt
            coding_prompt = "Write a Python function to calculate fibonacci numbers"
            result1 = engine.process_prompt(coding_prompt)
            
            assert result1.model_used == ModelType.CODER
            assert "fibonacci" in result1.response
            assert result1.metrics.tokens_per_second > 0
            
            # Test writing prompt (should trigger model switch)
            writing_prompt = "Write an essay about climate change"
            result2 = engine.process_prompt(writing_prompt)
            
            assert result2.model_used == ModelType.WRITER
            assert "climate change" in result2.response.lower()
            assert result2.metrics.tokens_per_second > 0
            
            # Verify both inference calls were made
            assert mock_run_inference.call_count == 2
    
    @patch('aura_engine.hardware.profiler.HardwareProfiler.get_hardware_profile')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper._validate_binary')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper.run_inference')
    def test_phase2_forced_model_selection(self, mock_run_inference, mock_validate_binary, mock_get_profile):
        """Test Phase 2 with forced model selection."""
        # Mock setup
        mock_get_profile.return_value = self.hardware_profile
        mock_validate_binary.return_value = None
        
        mock_response = ProcessedOutput(
            response="This is a forced response from the writer model",
            tokens_generated=10,
            model_load_time=1.5,
            inference_time=2.0,
            error_message=None
        )
        mock_run_inference.return_value = mock_response
        
        # Create temporary model files
        with tempfile.TemporaryDirectory() as temp_dir:
            model_paths = {
                ModelType.CODER: os.path.join(temp_dir, "coder.gguf"),
                ModelType.WRITER: os.path.join(temp_dir, "writer.gguf")
            }
            
            for path in model_paths.values():
                with open(path, 'w') as f:
                    f.write("mock model")
            
            # Initialize engine
            engine = InferenceEngine(self.config)
            engine.initialize(model_paths)
            
            # Test forced model selection (coding prompt but force writer model)
            coding_prompt = "Write a Python function"  # Would normally route to CODER
            result = engine.process_prompt(coding_prompt, force_model_type=ModelType.WRITER)
            
            # Should use WRITER model despite coding prompt
            assert result.model_used == ModelType.WRITER
            assert result.response == "This is a forced response from the writer model"
    
    @patch('aura_engine.hardware.profiler.HardwareProfiler.get_hardware_profile')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper._validate_binary')
    def test_phase2_orchestrator_info(self, mock_validate_binary, mock_get_profile):
        """Test Phase 2 orchestrator information retrieval."""
        # Mock setup
        mock_get_profile.return_value = self.hardware_profile
        mock_validate_binary.return_value = None
        
        # Create temporary model files
        with tempfile.TemporaryDirectory() as temp_dir:
            model_paths = {
                ModelType.CODER: os.path.join(temp_dir, "coder.gguf"),
                ModelType.WRITER: os.path.join(temp_dir, "writer.gguf")
            }
            
            for path in model_paths.values():
                with open(path, 'w') as f:
                    f.write("mock model")
            
            # Initialize engine
            engine = InferenceEngine(self.config)
            engine.initialize(model_paths)
            
            # Get orchestrator info
            info = engine.get_orchestrator_info()
            
            assert info is not None
            assert 'current_model' in info
            assert 'memory_usage' in info
            assert 'available_models' in info
            assert ModelType.CODER in info['available_models']
            assert ModelType.WRITER in info['available_models']
    
    @patch('aura_engine.hardware.profiler.HardwareProfiler.get_hardware_profile')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper._validate_binary')
    def test_phase2_routing_explanation(self, mock_validate_binary, mock_get_profile):
        """Test Phase 2 routing explanation functionality."""
        # Mock setup
        mock_get_profile.return_value = self.hardware_profile
        mock_validate_binary.return_value = None
        
        # Create temporary model files
        with tempfile.TemporaryDirectory() as temp_dir:
            model_paths = {
                ModelType.CODER: os.path.join(temp_dir, "coder.gguf")
            }
            
            with open(model_paths[ModelType.CODER], 'w') as f:
                f.write("mock model")
            
            # Initialize engine
            engine = InferenceEngine(self.config)
            engine.initialize(model_paths)
            
            # Get routing explanation
            explanation = engine.get_routing_explanation("Write a Python function")
            
            assert explanation is not None
            assert 'prompt' in explanation
            assert 'selected_model' in explanation
            assert 'rule_scores' in explanation
            assert explanation['selected_model'] == ModelType.CODER
    
    @patch('aura_engine.hardware.profiler.HardwareProfiler.get_hardware_profile')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper._validate_binary')
    def test_phase1_compatibility(self, mock_validate_binary, mock_get_profile):
        """Test that Phase 1 functionality still works when no model paths provided."""
        # Mock setup
        mock_get_profile.return_value = self.hardware_profile
        mock_validate_binary.return_value = None
        
        # Initialize engine without model paths (Phase 1 mode)
        engine = InferenceEngine(self.config)
        engine.initialize()  # No model_paths parameter
        
        # Verify Phase 1 mode
        assert engine.model_orchestrator is None
        
        # Phase 1 methods should return None
        assert engine.get_orchestrator_info() is None
        assert engine.get_routing_explanation("test") is None
    
    @patch('aura_engine.hardware.profiler.HardwareProfiler.get_hardware_profile')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper._validate_binary')
    def test_phase2_memory_management_logging(self, mock_validate_binary, mock_get_profile):
        """Test that Phase 2 properly logs memory management operations."""
        # Mock setup
        mock_get_profile.return_value = self.hardware_profile
        mock_validate_binary.return_value = None
        
        # Create temporary model files
        with tempfile.TemporaryDirectory() as temp_dir:
            model_paths = {
                ModelType.CODER: os.path.join(temp_dir, "coder.gguf"),
                ModelType.WRITER: os.path.join(temp_dir, "writer.gguf")
            }
            
            for path in model_paths.values():
                with open(path, 'w') as f:
                    f.write("mock model")
            
            # Initialize engine
            engine = InferenceEngine(self.config)
            
            # Capture log messages
            with patch('aura_engine.engine.logging.getLogger') as mock_logger:
                mock_log = Mock()
                mock_logger.return_value = mock_log
                
                engine.initialize(model_paths)
                
                # Verify initialization logging (check that some orchestrator-related logs were made)
                # The exact log messages may vary, so check for key components
                log_calls = [str(call) for call in mock_log.info.call_args_list]
                orchestrator_logs = [log for log in log_calls if 'orchestrator' in log.lower() or 'model' in log.lower()]
                assert len(orchestrator_logs) > 0, f"Expected orchestrator-related logs, got: {log_calls}"
    
    @patch('aura_engine.hardware.profiler.HardwareProfiler.get_hardware_profile')
    @patch('aura_engine.llama_wrapper.wrapper.LlamaWrapper._validate_binary')
    def test_phase2_error_handling_missing_models(self, mock_validate_binary, mock_get_profile):
        """Test Phase 2 error handling when model files are missing."""
        # Mock setup
        mock_get_profile.return_value = self.hardware_profile
        mock_validate_binary.return_value = None
        
        # Create engine
        engine = InferenceEngine(self.config)
        
        # Try to initialize with missing model files
        model_paths = {
            ModelType.CODER: "/nonexistent/coder.gguf",
            ModelType.WRITER: "/nonexistent/writer.gguf"
        }
        
        with pytest.raises(FileNotFoundError):
            engine.initialize(model_paths)
        
        # Engine should not be initialized
        assert not engine.is_initialized


if __name__ == "__main__":
    pytest.main([__file__])