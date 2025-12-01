"""
Hardware profiling and detection for optimal inference configuration.
"""

import logging
import subprocess
import xml.etree.ElementTree as ET
from typing import Optional
import psutil

from ..models import HardwareProfile


logger = logging.getLogger(__name__)


class HardwareProfiler:
    """
    Hardware profiler for detecting system capabilities and calculating
    optimal inference settings.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_system_memory(self) -> int:
        """
        Detect available system RAM in megabytes.
        
        Returns:
            int: Available system RAM in MB
        """
        try:
            memory = psutil.virtual_memory()
            ram_mb = int(memory.total / (1024 * 1024))
            self.logger.info(f"Detected system RAM: {ram_mb}MB")
            return ram_mb
        except Exception as e:
            self.logger.error(f"Failed to detect system memory: {e}")
            # Fallback to conservative estimate
            return 8192  # 8GB default
    
    def detect_gpu_memory(self) -> tuple[Optional[int], Optional[str]]:
        """
        Detect GPU VRAM using nvidia-smi XML parsing.
        
        Returns:
            tuple: (VRAM in MB, GPU name) or (None, None) if no GPU detected
        """
        try:
            # Try to run nvidia-smi with XML output
            result = subprocess.run(
                ['nvidia-smi', '-q', '-x'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.warning("nvidia-smi command failed, falling back to CPU-only")
                return None, None
            
            # Parse XML output
            root = ET.fromstring(result.stdout)
            
            # Find the first GPU
            gpu = root.find('.//gpu')
            if gpu is None:
                self.logger.warning("No GPU found in nvidia-smi output")
                return None, None
            
            # Extract GPU name
            gpu_name = gpu.find('.//product_name')
            gpu_name_str = gpu_name.text if gpu_name is not None else "Unknown GPU"
            
            # Extract memory information
            memory_info = gpu.find('.//fb_memory_usage')
            if memory_info is None:
                self.logger.warning("No memory information found for GPU")
                return None, None
            
            total_memory = memory_info.find('.//total')
            if total_memory is None:
                self.logger.warning("No total memory information found")
                return None, None
            
            # Parse memory value (format: "XXXX MiB")
            memory_text = total_memory.text.strip()
            if memory_text.endswith(' MiB'):
                vram_mb = int(memory_text.replace(' MiB', ''))
                self.logger.info(f"Detected GPU: {gpu_name_str} with {vram_mb}MB VRAM")
                return vram_mb, gpu_name_str
            else:
                self.logger.warning(f"Unexpected memory format: {memory_text}")
                return None, None
                
        except subprocess.TimeoutExpired:
            self.logger.warning("nvidia-smi command timed out")
            return None, None
        except FileNotFoundError:
            self.logger.info("nvidia-smi not found, assuming CPU-only system")
            return None, None
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse nvidia-smi XML output: {e}")
            return None, None
        except Exception as e:
            self.logger.error(f"Unexpected error during GPU detection: {e}")
            return None, None
    
    def calculate_gpu_layers(self, model_size_mb: int, available_vram_mb: int) -> int:
        """
        Calculate optimal number of GPU layers based on model size and available VRAM.
        
        Args:
            model_size_mb: Estimated model size in MB
            available_vram_mb: Available GPU VRAM in MB
            
        Returns:
            int: Optimal number of GPU layers (0 for CPU-only)
        """
        if available_vram_mb is None or available_vram_mb <= 0:
            return 0
        
        # Reserve 1GB for GPU operations and other processes
        reserved_vram_mb = 1024
        usable_vram_mb = max(0, available_vram_mb - reserved_vram_mb)
        
        if usable_vram_mb < 512:  # Less than 512MB usable
            self.logger.warning("Insufficient VRAM for GPU acceleration")
            return 0
        
        # Estimate layers based on VRAM capacity
        # Rough estimate: each layer uses ~100-200MB for 7B models
        estimated_mb_per_layer = 150
        
        # Calculate maximum layers that fit in VRAM
        max_layers = min(40, usable_vram_mb // estimated_mb_per_layer)  # Cap at 40 layers
        
        # For smaller models, use more layers
        if model_size_mb < 4000:  # < 4GB model
            gpu_layers = min(max_layers, 35)
        elif model_size_mb < 8000:  # < 8GB model
            gpu_layers = min(max_layers, 30)
        else:  # Larger models
            gpu_layers = min(max_layers, 20)
        
        self.logger.info(f"Calculated optimal GPU layers: {gpu_layers} (VRAM: {available_vram_mb}MB, Model: {model_size_mb}MB)")
        return gpu_layers
    
    def get_cpu_cores(self) -> int:
        """
        Get the number of CPU cores available.
        
        Returns:
            int: Number of CPU cores
        """
        try:
            cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
            self.logger.info(f"Detected CPU cores: {cores}")
            return cores
        except Exception as e:
            self.logger.error(f"Failed to detect CPU cores: {e}")
            return 4  # Conservative fallback
    
    def determine_performance_tier(self, gpu_vram_mb: Optional[int]) -> str:
        """
        Determine performance tier based on available VRAM.
        
        Args:
            gpu_vram_mb: Available GPU VRAM in MB
            
        Returns:
            str: Performance tier ('high-performance', 'balanced', 'high-efficiency')
        """
        if gpu_vram_mb is None or gpu_vram_mb < 8000:  # < 8GB VRAM (with some margin)
            tier = "high-efficiency"
        elif gpu_vram_mb < 10000:  # 8-10GB VRAM (with some margin)
            tier = "balanced"
        else:  # >= 10GB VRAM
            tier = "high-performance"
        
        self.logger.info(f"Determined performance tier: {tier} (VRAM: {gpu_vram_mb}MB)")
        return tier

    def get_hardware_profile(self, estimated_model_size_mb: int = 7000) -> HardwareProfile:
        """
        Generate a complete hardware profile for the system.
        
        Args:
            estimated_model_size_mb: Estimated model size in MB for GPU layer calculation
            
        Returns:
            HardwareProfile: Complete hardware configuration
        """
        self.logger.info("Starting hardware profiling...")
        
        # Detect system components
        system_ram_mb = self.detect_system_memory()
        gpu_vram_mb, gpu_name = self.detect_gpu_memory()
        cpu_cores = self.get_cpu_cores()
        
        # Determine performance tier
        performance_tier = self.determine_performance_tier(gpu_vram_mb)
        
        # Calculate optimal GPU layers
        optimal_gpu_layers = self.calculate_gpu_layers(estimated_model_size_mb, gpu_vram_mb or 0)
        
        profile = HardwareProfile(
            system_ram_mb=system_ram_mb,
            gpu_vram_mb=gpu_vram_mb,
            gpu_name=gpu_name,
            optimal_gpu_layers=optimal_gpu_layers,
            cpu_cores=cpu_cores,
            performance_tier=performance_tier
        )
        
        self.logger.info(f"Hardware profile complete: {profile}")
        return profile