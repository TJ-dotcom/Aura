"""
Performance monitoring and benchmarking for AURA-Engine-Core.
"""

import logging
import time
import psutil
import os
from typing import Optional
from datetime import datetime

from ..models import EngineConfig, PerformanceMetrics, BenchmarkEntry, HardwareProfile


class PerformanceMonitor:
    """
    Performance monitoring system for tracking inference metrics
    and generating benchmark reports.
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Timing state
        self.inference_start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.completion_time: Optional[float] = None
        
        # Memory tracking
        self.initial_memory: Optional[int] = None
        self.peak_memory: int = 0
        self.peak_vram: int = 0
        
        # Process reference for memory monitoring
        self.process = psutil.Process()
    
    def start_inference_timer(self) -> None:
        """
        Start timing for inference operation.
        """
        self.inference_start_time = time.time()
        self.first_token_time = None
        self.completion_time = None
        
        # Record initial memory state
        self.initial_memory = self._get_current_memory()
        self.peak_memory = self.initial_memory
        self.peak_vram = self._get_current_vram()
        
        self.logger.debug(f"Started inference timer at {self.inference_start_time}")
    
    def record_first_token(self) -> None:
        """
        Record the time when the first token is generated.
        """
        if self.inference_start_time is None:
            self.logger.warning("Cannot record first token: inference timer not started")
            return
        
        self.first_token_time = time.time()
        self.logger.debug(f"First token recorded at {self.first_token_time}")
    
    def record_completion(self, token_count: int, override_tps: float = None) -> None:
        """
        Record inference completion and final token count.
        
        Args:
            token_count: Total number of tokens generated
            override_tps: Optional pre-calculated TPS to use instead of recalculating
        """
        if self.inference_start_time is None:
            self.logger.warning("Cannot record completion: inference timer not started")
            return
        
        self.completion_time = time.time()
        self.token_count = token_count
        self.override_tps = override_tps  # Store the pre-calculated TPS
        
        # Update peak memory usage
        self._update_peak_memory()
        
        self.logger.debug(f"Inference completed at {self.completion_time} with {token_count} tokens")
    
    def get_metrics(self, model_load_time: float = 0.0) -> PerformanceMetrics:
        """
        Calculate and return performance metrics.
        
        Args:
            model_load_time: Time taken to load the model (seconds)
            
        Returns:
            PerformanceMetrics: Calculated performance metrics
        """
        if self.inference_start_time is None or self.completion_time is None:
            raise RuntimeError("Cannot calculate metrics: timing data incomplete")
        
        # Calculate TTFT (Time to First Token)
        if self.first_token_time is not None:
            ttft_ms = (self.first_token_time - self.inference_start_time) * 1000
        else:
            # Estimate TTFT as 10% of total time if not recorded
            total_time = self.completion_time - self.inference_start_time
            ttft_ms = total_time * 100  # Convert to ms and estimate
        
        # Calculate total inference time
        total_inference_time = self.completion_time - self.inference_start_time
        
        # Calculate tokens per second
        token_count = getattr(self, 'token_count', 0)
        override_tps = getattr(self, 'override_tps', None)
        
        if override_tps is not None:
            # Use the pre-calculated TPS (e.g., from Ollama native metrics)
            tokens_per_second = override_tps
            self.logger.debug(f"Using override TPS: {override_tps:.2f}")
        elif total_inference_time > 0 and token_count > 0:
            tokens_per_second = token_count / total_inference_time
        else:
            tokens_per_second = 0.0
        
        # Update final memory readings
        self._update_peak_memory()
        
        metrics = PerformanceMetrics(
            ttft_ms=ttft_ms,
            tokens_per_second=tokens_per_second,
            peak_ram_mb=self.peak_memory,
            peak_vram_mb=self.peak_vram,
            model_load_time_s=model_load_time
        )
        
        self.logger.info(f"Performance metrics: {metrics}")
        return metrics
    
    def log_benchmark(self, metrics: PerformanceMetrics, hardware_profile: HardwareProfile,
                     phase: int, scenario: str) -> None:
        """
        Log benchmark data to BENCHMARKS.md file.
        
        Args:
            metrics: Performance metrics to log
            hardware_profile: Hardware configuration
            phase: Project phase number
            scenario: Benchmark scenario name
        """
        if not self.config.enable_benchmarking:
            return
        
        try:
            # Create benchmark entry
            entry = BenchmarkEntry.from_metrics(
                metrics=metrics,
                hardware_profile=hardware_profile,
                phase=phase,
                scenario=scenario
            )
            
            # Ensure BENCHMARKS.md exists with header (use absolute path)
            import os.path
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            benchmark_file = os.path.join(script_dir, "markdown", "BENCHMARKS.md")
            if not os.path.exists(benchmark_file):
                self._create_benchmark_file(benchmark_file)
            
            # Append benchmark data
            with open(benchmark_file, 'a', encoding='utf-8') as f:
                f.write(entry.to_table_row() + '\\n')
            
            self.logger.info(f"Benchmark data logged: {scenario} - Phase {phase}")
            
        except Exception as e:
            self.logger.error(f"Failed to log benchmark data: {e}")
    
    def _get_current_memory(self) -> int:
        """
        Get current RAM usage in MB.
        
        Returns:
            int: Current RAM usage in MB
        """
        try:
            memory_info = self.process.memory_info()
            return int(memory_info.rss / (1024 * 1024))  # Convert to MB
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0
    
    def _get_current_vram(self) -> int:
        """
        Get current VRAM usage in MB (best effort).
        
        Returns:
            int: Current VRAM usage in MB (0 if unavailable)
        """
        try:
            # This is a simplified approach - in a real implementation,
            # you might use nvidia-ml-py or similar for accurate VRAM tracking
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                vram_mb = int(result.stdout.strip())
                return vram_mb
            
        except Exception:
            pass  # Silently fail for VRAM detection
        
        return 0
    
    def _update_peak_memory(self) -> None:
        """
        Update peak memory usage tracking.
        """
        current_ram = self._get_current_memory()
        current_vram = self._get_current_vram()
        
        # Ensure peak values are initialized
        if self.peak_memory is None:
            self.peak_memory = 0
        if self.peak_vram is None:
            self.peak_vram = 0
        
        self.peak_memory = max(self.peak_memory or 0, current_ram or 0)
        self.peak_vram = max(self.peak_vram or 0, current_vram or 0)
    
    def _create_benchmark_file(self, filename: str) -> None:
        """
        Create BENCHMARKS.md file with proper header.
        
        Args:
            filename: Path to benchmark file
        """
        header = """# AURA-Engine-Core Benchmarks

Performance benchmarks for the AURA-Engine-Core system across all development phases.

| Date       | Hardware          | Phase | Scenario | TTFT (ms) | TPS    | Peak VRAM (MB) | Peak RAM (MB) |
|------------|-------------------|-------|----------|-----------|--------|----------------|---------------|
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(header)