"""
Core data models and enums for the AURA-Engine-Core system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime


class ModelType(Enum):
    """Enumeration of available model types for specialized inference."""
    CODER = "coder"
    WRITER = "writer"
    GENERAL = "general"
    MATH = "math"


@dataclass
class HardwareProfile:
    """Hardware configuration profile for optimal inference settings."""
    system_ram_mb: int
    gpu_vram_mb: Optional[int]
    gpu_name: Optional[str]
    optimal_gpu_layers: int
    cpu_cores: int
    performance_tier: str = "high-efficiency"  # Default tier
    
    def __str__(self) -> str:
        gpu_info = f"{self.gpu_name} ({self.gpu_vram_mb}MB VRAM)" if self.gpu_name else "CPU Only"
        return f"Hardware: {gpu_info}, RAM: {self.system_ram_mb}MB, CPU Cores: {self.cpu_cores}, GPU Layers: {self.optimal_gpu_layers}, Tier: {self.performance_tier}"


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during inference operations."""
    ttft_ms: float
    tokens_per_second: float
    peak_ram_mb: int
    peak_vram_mb: int
    model_load_time_s: float
    
    def __str__(self) -> str:
        return f"TTFT: {self.ttft_ms:.1f}ms, TPS: {self.tokens_per_second:.1f}, Peak RAM: {self.peak_ram_mb}MB, Peak VRAM: {self.peak_vram_mb}MB"


@dataclass
class InferenceResult:
    """Complete result of an inference operation including response and metrics."""
    response: str
    metrics: PerformanceMetrics
    model_used: ModelType
    rag_context: Optional[str] = None
    
    def __str__(self) -> str:
        context_info = f" (RAG: {len(self.rag_context)} chars)" if self.rag_context else ""
        return f"Model: {self.model_used.value}{context_info}, {self.metrics}"


@dataclass
class BenchmarkEntry:
    """Structured benchmark data for performance tracking."""
    date: str
    hardware: str
    phase: int
    scenario: str
    ttft_ms: float
    tps: float
    peak_vram_mb: int
    peak_ram_mb: int
    
    @classmethod
    def from_metrics(cls, metrics: PerformanceMetrics, hardware_profile: HardwareProfile, 
                     phase: int, scenario: str) -> 'BenchmarkEntry':
        """Create a benchmark entry from performance metrics and hardware profile."""
        hardware_desc = f"{hardware_profile.gpu_name or 'CPU'} / {hardware_profile.system_ram_mb//1024}GB"
        return cls(
            date=datetime.now().strftime("%Y-%m-%d"),
            hardware=hardware_desc,
            phase=phase,
            scenario=scenario,
            ttft_ms=metrics.ttft_ms,
            tps=metrics.tokens_per_second,
            peak_vram_mb=metrics.peak_vram_mb,
            peak_ram_mb=metrics.peak_ram_mb
        )
    
    def to_table_row(self) -> str:
        """Convert benchmark entry to markdown table row format."""
        return f"| {self.date} | {self.hardware} | {self.phase} | {self.scenario} | {self.ttft_ms:.1f} | {self.tps:.1f} | {self.peak_vram_mb} | {self.peak_ram_mb} |"


@dataclass
class EngineConfig:
    """Configuration settings for the inference engine."""
    models_dir: str = "models"
    llama_cpp_path: str = "llama.cpp"
    faiss_index_path: Optional[str] = None
    enable_benchmarking: bool = True
    log_level: str = "INFO"
    max_tokens: int = 512
    temperature: float = 0.7
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return True