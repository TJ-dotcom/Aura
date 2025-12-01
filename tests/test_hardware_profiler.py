"""
Unit tests for hardware profiler functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import subprocess
import xml.etree.ElementTree as ET

from aura_engine.hardware.profiler import HardwareProfiler
from aura_engine.models import HardwareProfile


class TestHardwareProfiler:
    """Test suite for HardwareProfiler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = HardwareProfiler()
    
    @patch('psutil.virtual_memory')
    def test_detect_system_memory_success(self, mock_memory):
        """Test successful system memory detection."""
        # Mock psutil response
        mock_memory.return_value = Mock(total=16 * 1024 * 1024 * 1024)  # 16GB
        
        result = self.profiler.detect_system_memory()
        
        assert result == 16384  # 16GB in MB
        mock_memory.assert_called_once()
    
    @patch('psutil.virtual_memory')
    def test_detect_system_memory_failure(self, mock_memory):
        """Test system memory detection with psutil failure."""
        mock_memory.side_effect = Exception("psutil error")
        
        result = self.profiler.detect_system_memory()
        
        assert result == 8192  # Fallback value
    
    @patch('subprocess.run')
    def test_detect_gpu_memory_success(self, mock_run):
        """Test successful GPU detection with nvidia-smi."""
        # Mock nvidia-smi XML output
        xml_output = '''<?xml version="1.0" ?>
        <nvidia_smi_log>
            <gpu id="00000000:01:00.0">
                <product_name>NVIDIA GeForce RTX 4080</product_name>
                <fb_memory_usage>
                    <total>16384 MiB</total>
                    <used>1024 MiB</used>
                    <free>15360 MiB</free>
                </fb_memory_usage>
            </gpu>
        </nvidia_smi_log>'''
        
        mock_run.return_value = Mock(returncode=0, stdout=xml_output)
        
        vram_mb, gpu_name = self.profiler.detect_gpu_memory()
        
        assert vram_mb == 16384
        assert gpu_name == "NVIDIA GeForce RTX 4080"
        mock_run.assert_called_once_with(
            ['nvidia-smi', '-q', '-x'],
            capture_output=True,
            text=True,
            timeout=10
        )
    
    @patch('subprocess.run')
    def test_detect_gpu_memory_nvidia_smi_not_found(self, mock_run):
        """Test GPU detection when nvidia-smi is not found."""
        mock_run.side_effect = FileNotFoundError()
        
        vram_mb, gpu_name = self.profiler.detect_gpu_memory()
        
        assert vram_mb is None
        assert gpu_name is None
    
    @patch('subprocess.run')
    def test_detect_gpu_memory_nvidia_smi_fails(self, mock_run):
        """Test GPU detection when nvidia-smi command fails."""
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Error")
        
        vram_mb, gpu_name = self.profiler.detect_gpu_memory()
        
        assert vram_mb is None
        assert gpu_name is None
    
    @patch('subprocess.run')
    def test_detect_gpu_memory_timeout(self, mock_run):
        """Test GPU detection with nvidia-smi timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(['nvidia-smi'], 10)
        
        vram_mb, gpu_name = self.profiler.detect_gpu_memory()
        
        assert vram_mb is None
        assert gpu_name is None
    
    @patch('subprocess.run')
    def test_detect_gpu_memory_invalid_xml(self, mock_run):
        """Test GPU detection with invalid XML output."""
        mock_run.return_value = Mock(returncode=0, stdout="invalid xml")
        
        vram_mb, gpu_name = self.profiler.detect_gpu_memory()
        
        assert vram_mb is None
        assert gpu_name is None
    
    @patch('subprocess.run')
    def test_detect_gpu_memory_no_gpu_in_xml(self, mock_run):
        """Test GPU detection with XML containing no GPU information."""
        xml_output = '''<?xml version="1.0" ?>
        <nvidia_smi_log>
        </nvidia_smi_log>'''
        
        mock_run.return_value = Mock(returncode=0, stdout=xml_output)
        
        vram_mb, gpu_name = self.profiler.detect_gpu_memory()
        
        assert vram_mb is None
        assert gpu_name is None
    
    def test_calculate_gpu_layers_no_vram(self):
        """Test GPU layer calculation with no VRAM available."""
        result = self.profiler.calculate_gpu_layers(7000, 0)
        assert result == 0
        
        result = self.profiler.calculate_gpu_layers(7000, None)
        assert result == 0
    
    def test_calculate_gpu_layers_insufficient_vram(self):
        """Test GPU layer calculation with insufficient VRAM."""
        result = self.profiler.calculate_gpu_layers(7000, 512)  # Only 512MB VRAM
        assert result == 0
    
    def test_calculate_gpu_layers_small_model(self):
        """Test GPU layer calculation for small model."""
        result = self.profiler.calculate_gpu_layers(3000, 8192)  # 3GB model, 8GB VRAM
        assert result > 0
        assert result <= 35  # Should use more layers for smaller models
    
    def test_calculate_gpu_layers_medium_model(self):
        """Test GPU layer calculation for medium model."""
        result = self.profiler.calculate_gpu_layers(7000, 16384)  # 7GB model, 16GB VRAM
        assert result > 0
        assert result <= 30
    
    def test_calculate_gpu_layers_large_model(self):
        """Test GPU layer calculation for large model."""
        result = self.profiler.calculate_gpu_layers(12000, 24576)  # 12GB model, 24GB VRAM
        assert result > 0
        assert result <= 20  # Should use fewer layers for larger models
    
    def test_calculate_gpu_layers_vram_limit(self):
        """Test GPU layer calculation respects VRAM limits."""
        # Very limited VRAM after reservation
        result = self.profiler.calculate_gpu_layers(7000, 1536)  # 1.5GB VRAM
        assert result >= 0
        assert result <= 3  # Should be very conservative
    
    @patch('psutil.cpu_count')
    def test_get_cpu_cores_success(self, mock_cpu_count):
        """Test successful CPU core detection."""
        mock_cpu_count.return_value = 8
        
        result = self.profiler.get_cpu_cores()
        
        assert result == 8
        mock_cpu_count.assert_called_with(logical=False)
    
    @patch('psutil.cpu_count')
    def test_get_cpu_cores_fallback_logical(self, mock_cpu_count):
        """Test CPU core detection with logical fallback."""
        mock_cpu_count.side_effect = [None, 16]  # Physical cores None, logical cores 16
        
        result = self.profiler.get_cpu_cores()
        
        assert result == 16
    
    @patch('psutil.cpu_count')
    def test_get_cpu_cores_failure(self, mock_cpu_count):
        """Test CPU core detection failure."""
        mock_cpu_count.side_effect = Exception("CPU detection error")
        
        result = self.profiler.get_cpu_cores()
        
        assert result == 4  # Fallback value
    
    @patch.object(HardwareProfiler, 'detect_system_memory')
    @patch.object(HardwareProfiler, 'detect_gpu_memory')
    @patch.object(HardwareProfiler, 'get_cpu_cores')
    @patch.object(HardwareProfiler, 'calculate_gpu_layers')
    def test_get_hardware_profile_complete(self, mock_calc_layers, mock_cpu_cores, 
                                         mock_gpu_memory, mock_system_memory):
        """Test complete hardware profile generation."""
        # Mock all component methods
        mock_system_memory.return_value = 32768  # 32GB RAM
        mock_gpu_memory.return_value = (24576, "RTX 4090")  # 24GB VRAM
        mock_cpu_cores.return_value = 16
        mock_calc_layers.return_value = 25
        
        profile = self.profiler.get_hardware_profile(7000)
        
        assert isinstance(profile, HardwareProfile)
        assert profile.system_ram_mb == 32768
        assert profile.gpu_vram_mb == 24576
        assert profile.gpu_name == "RTX 4090"
        assert profile.cpu_cores == 16
        assert profile.optimal_gpu_layers == 25
        
        # Verify all methods were called
        mock_system_memory.assert_called_once()
        mock_gpu_memory.assert_called_once()
        mock_cpu_cores.assert_called_once()
        mock_calc_layers.assert_called_once_with(7000, 24576)
    
    @patch.object(HardwareProfiler, 'detect_system_memory')
    @patch.object(HardwareProfiler, 'detect_gpu_memory')
    @patch.object(HardwareProfiler, 'get_cpu_cores')
    @patch.object(HardwareProfiler, 'calculate_gpu_layers')
    def test_get_hardware_profile_no_gpu(self, mock_calc_layers, mock_cpu_cores, 
                                       mock_gpu_memory, mock_system_memory):
        """Test hardware profile generation with no GPU."""
        # Mock CPU-only system
        mock_system_memory.return_value = 16384  # 16GB RAM
        mock_gpu_memory.return_value = (None, None)  # No GPU
        mock_cpu_cores.return_value = 8
        mock_calc_layers.return_value = 0  # CPU-only
        
        profile = self.profiler.get_hardware_profile()
        
        assert profile.system_ram_mb == 16384
        assert profile.gpu_vram_mb is None
        assert profile.gpu_name is None
        assert profile.cpu_cores == 8
        assert profile.optimal_gpu_layers == 0
        
        mock_calc_layers.assert_called_once_with(7000, 0)  # Default model size, no VRAM


if __name__ == "__main__":
    pytest.main([__file__])