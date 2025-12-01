"""
Unit tests for model manager functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from aura_engine.orchestrator.model_manager import ModelManager, ModelInfo
from aura_engine.models import ModelType, HardwareProfile
from aura_engine.llama_wrapper import LlamaWrapper


class TestModelManager:
    """Test suite for ModelManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock hardware profile
        self.hardware_profile = HardwareProfile(
            system_ram_mb=16384,
            gpu_vram_mb=8192,
            gpu_name="RTX 4070",
            optimal_gpu_layers=25,
            cpu_cores=8
        )
        
        # Mock llama wrapper
        self.mock_llama_wrapper = Mock(spec=LlamaWrapper)
        
        # Create model manager
        with patch('psutil.Process'):
            self.manager = ModelManager(self.mock_llama_wrapper, self.hardware_profile)
    
    def test_initialization(self):
        """Test model manager initialization."""
        assert self.manager.llama_wrapper == self.mock_llama_wrapper
        assert self.manager.hardware_profile == self.hardware_profile
        assert self.manager.current_model is None
        assert len(self.manager.model_paths) == 0
    
    def test_configure_model_paths_success(self):
        """Test successful model path configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary model files
            coder_path = os.path.join(temp_dir, "coder.gguf")
            writer_path = os.path.join(temp_dir, "writer.gguf")
            
            with open(coder_path, 'w') as f:
                f.write("mock model")
            with open(writer_path, 'w') as f:
                f.write("mock model")
            
            model_paths = {
                ModelType.CODER: coder_path,
                ModelType.WRITER: writer_path
            }
            
            self.manager.configure_model_paths(model_paths)
            
            assert self.manager.model_paths == model_paths
    
    def test_configure_model_paths_missing_file(self):
        """Test model path configuration with missing file."""
        model_paths = {
            ModelType.CODER: "/nonexistent/path/model.gguf"
        }
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            self.manager.configure_model_paths(model_paths)
    
    @patch('os.path.isfile')
    @patch('os.path.getsize')
    @patch('time.time')
    def test_load_model_success(self, mock_time, mock_getsize, mock_isfile):
        """Test successful model loading."""
        # Setup mocks
        mock_isfile.return_value = True
        mock_getsize.return_value = 7 * 1024 * 1024 * 1024  # 7GB
        mock_time.side_effect = [100.0, 102.5]  # Load time simulation
        
        # Configure model paths
        model_paths = {ModelType.CODER: "/path/to/coder.gguf"}
        self.manager.model_paths = model_paths
        
        # Mock memory tracking
        with patch.object(self.manager, '_get_current_memory', side_effect=[1000, 1500]):
            result = self.manager.load_model(ModelType.CODER, gpu_layers=30)
        
        # Verify result
        assert isinstance(result, ModelInfo)
        assert result.model_type == ModelType.CODER
        assert result.model_path == "/path/to/coder.gguf"
        assert result.gpu_layers == 30
        assert result.load_time == 2.5
        
        # Verify state
        assert self.manager.current_model == result
        assert self.manager.is_model_loaded()
        assert self.manager.get_current_model() == ModelType.CODER
    
    def test_load_model_unconfigured_type(self):
        """Test loading unconfigured model type."""
        with pytest.raises(ValueError, match="Model type coder not configured"):
            self.manager.load_model(ModelType.CODER)
    
    @patch('os.path.isfile')
    def test_load_model_missing_file(self, mock_isfile):
        """Test loading model with missing file."""
        mock_isfile.return_value = False
        
        # Configure model paths
        self.manager.model_paths = {ModelType.CODER: "/path/to/missing.gguf"}
        
        with pytest.raises(RuntimeError, match="Model loading failed"):
            self.manager.load_model(ModelType.CODER)
    
    def test_unload_current_model_with_model(self):
        """Test unloading when a model is loaded."""
        # Setup loaded model
        model_info = ModelInfo(
            model_type=ModelType.CODER,
            model_path="/path/to/model.gguf",
            load_time=2.0,
            memory_usage_mb=500,
            gpu_layers=25
        )
        self.manager.current_model = model_info
        
        # Mock memory tracking
        with patch.object(self.manager, '_get_current_memory', side_effect=[1500, 1000]):
            self.manager.unload_current_model()
        
        # Verify model is unloaded
        assert self.manager.current_model is None
        assert not self.manager.is_model_loaded()
        assert self.manager.get_current_model() is None
    
    def test_unload_current_model_no_model(self):
        """Test unloading when no model is loaded."""
        assert self.manager.current_model is None
        
        # Should not raise an error
        self.manager.unload_current_model()
        
        assert self.manager.current_model is None
    
    @patch('os.path.isfile')
    @patch('os.path.getsize')
    @patch('time.time')
    def test_switch_model_different_type(self, mock_time, mock_getsize, mock_isfile):
        """Test switching to a different model type."""
        # Setup mocks
        mock_isfile.return_value = True
        mock_getsize.return_value = 7 * 1024 * 1024 * 1024
        mock_time.side_effect = [100.0, 102.0, 105.0, 107.5]  # Two load operations
        
        # Configure model paths
        model_paths = {
            ModelType.CODER: "/path/to/coder.gguf",
            ModelType.WRITER: "/path/to/writer.gguf"
        }
        self.manager.model_paths = model_paths
        
        # Load first model
        with patch.object(self.manager, '_get_current_memory', side_effect=[1000, 1500, 1200, 1800, 1600, 2000]):
            # Load coder model
            coder_info = self.manager.load_model(ModelType.CODER)
            assert self.manager.get_current_model() == ModelType.CODER
            
            # Switch to writer model
            writer_info = self.manager.switch_model(ModelType.WRITER)
            assert self.manager.get_current_model() == ModelType.WRITER
            assert writer_info.model_type == ModelType.WRITER
    
    @patch('os.path.isfile')
    @patch('os.path.getsize')
    def test_switch_model_same_type(self, mock_getsize, mock_isfile):
        """Test switching to the same model type (no-op)."""
        mock_isfile.return_value = True
        mock_getsize.return_value = 7 * 1024 * 1024 * 1024
        
        # Setup loaded model
        model_info = ModelInfo(
            model_type=ModelType.CODER,
            model_path="/path/to/coder.gguf",
            load_time=2.0,
            memory_usage_mb=500,
            gpu_layers=25
        )
        self.manager.current_model = model_info
        self.manager.model_paths = {ModelType.CODER: "/path/to/coder.gguf"}
        
        # Switch to same model
        result = self.manager.switch_model(ModelType.CODER)
        
        # Should return the same model info
        assert result == model_info
        assert self.manager.get_current_model() == ModelType.CODER
    
    def test_get_memory_usage(self):
        """Test memory usage reporting."""
        with patch.object(self.manager, '_get_current_memory', return_value=1500):
            self.manager.baseline_memory_mb = 1000
            
            # No model loaded
            usage = self.manager.get_memory_usage()
            expected = {
                'current_memory_mb': 1500,
                'baseline_memory_mb': 1000,
                'model_memory_mb': 0,
                'model_loaded': False
            }
            assert usage == expected
            
            # With model loaded
            self.manager.current_model = ModelInfo(
                ModelType.CODER, "/path", 2.0, 500, 25
            )
            
            usage = self.manager.get_memory_usage()
            expected['model_memory_mb'] = 500
            expected['model_loaded'] = True
            assert usage == expected
    
    @patch('psutil.Process')
    def test_get_current_memory_success(self, mock_process_class):
        """Test successful memory usage retrieval."""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1500  # 1500 MB in bytes
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process
        
        manager = ModelManager(self.mock_llama_wrapper, self.hardware_profile)
        memory_mb = manager._get_current_memory()
        
        assert memory_mb == 1500
    
    @patch('psutil.Process')
    def test_get_current_memory_failure(self, mock_process_class):
        """Test memory usage retrieval failure."""
        mock_process = Mock()
        mock_process.memory_info.side_effect = Exception("Memory error")
        mock_process_class.return_value = mock_process
        
        manager = ModelManager(self.mock_llama_wrapper, self.hardware_profile)
        memory_mb = manager._get_current_memory()
        
        assert memory_mb == 0  # Should return 0 on failure
    
    @patch('os.path.getsize')
    def test_calculate_gpu_layers(self, mock_getsize):
        """Test GPU layer calculation."""
        mock_getsize.return_value = 7 * 1024 * 1024 * 1024  # 7GB
        
        with patch('aura_engine.hardware.HardwareProfiler') as mock_profiler_class:
            mock_profiler = Mock()
            mock_profiler.calculate_gpu_layers.return_value = 30
            mock_profiler_class.return_value = mock_profiler
            
            gpu_layers = self.manager._calculate_gpu_layers("/path/to/model.gguf")
            
            assert gpu_layers == 30
            mock_profiler.calculate_gpu_layers.assert_called_once_with(
                model_size_mb=7168,  # 7GB in MB
                available_vram_mb=8192
            )
    
    def test_validate_model_paths(self):
        """Test model path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create one valid file
            valid_path = os.path.join(temp_dir, "valid.gguf")
            with open(valid_path, 'w') as f:
                f.write("mock")
            
            invalid_path = os.path.join(temp_dir, "invalid.gguf")
            
            self.manager.model_paths = {
                ModelType.CODER: valid_path,
                ModelType.WRITER: invalid_path
            }
            
            results = self.manager.validate_model_paths()
            
            assert results[ModelType.CODER] is True
            assert results[ModelType.WRITER] is False
    
    def test_cleanup(self):
        """Test cleanup functionality."""
        # Setup loaded model and paths
        self.manager.current_model = ModelInfo(
            ModelType.CODER, "/path", 2.0, 500, 25
        )
        self.manager.model_paths = {ModelType.CODER: "/path/to/model.gguf"}
        
        with patch.object(self.manager, 'unload_current_model') as mock_unload:
            self.manager.cleanup()
            
            mock_unload.assert_called_once()
            assert len(self.manager.model_paths) == 0


if __name__ == "__main__":
    pytest.main([__file__])