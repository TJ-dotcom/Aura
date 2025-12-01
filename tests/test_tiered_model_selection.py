"""
Test suite for tiered model selection functionality.
"""

import unittest
from unittest.mock import Mock, patch
import logging

from aura_engine.models import HardwareProfile
from aura_engine.orchestrator.model_catalog import ModelCatalog, ModelSpec
from aura_engine.orchestrator.model_manager import ModelManager
from aura_engine.orchestrator.orchestrator import ModelOrchestrator
from aura_engine.hardware.profiler import HardwareProfiler


class TestModelCatalog(unittest.TestCase):
    """Test the model catalog functionality."""
    
    def setUp(self):
        self.catalog = ModelCatalog()
    
    def test_get_available_tiers(self):
        """Test getting available performance tiers."""
        tiers = self.catalog.get_available_tiers()
        expected_tiers = ["high-performance", "balanced", "high-efficiency"]
        self.assertEqual(sorted(tiers), sorted(expected_tiers))
    
    def test_get_available_categories(self):
        """Test getting available model categories."""
        categories = self.catalog.get_available_categories()
        expected_categories = ["text", "coding", "mathematics"]
        self.assertEqual(sorted(categories), sorted(expected_categories))
    
    def test_get_models_for_tier_text(self):
        """Test getting text models for each tier."""
        # High-performance tier
        hp_models = self.catalog.get_models_for_tier("high-performance", "text")
        self.assertGreater(len(hp_models), 0)
        self.assertTrue(all(isinstance(m, ModelSpec) for m in hp_models))
        self.assertTrue(all(m.category == "text" for m in hp_models))
        
        # Balanced tier
        balanced_models = self.catalog.get_models_for_tier("balanced", "text")
        self.assertGreater(len(balanced_models), 0)
        
        # High-efficiency tier
        efficient_models = self.catalog.get_models_for_tier("high-efficiency", "text")
        self.assertGreater(len(efficient_models), 0)
    
    def test_get_models_for_tier_coding(self):
        """Test getting coding models for each tier."""
        for tier in ["high-performance", "balanced", "high-efficiency"]:
            models = self.catalog.get_models_for_tier(tier, "coding")
            self.assertGreater(len(models), 0)
            self.assertTrue(all(m.category == "coding" for m in models))
    
    def test_get_models_for_tier_mathematics(self):
        """Test getting mathematics models for each tier."""
        for tier in ["high-performance", "balanced", "high-efficiency"]:
            models = self.catalog.get_models_for_tier(tier, "mathematics")
            self.assertGreater(len(models), 0)
            self.assertTrue(all(m.category == "mathematics" for m in models))
    
    def test_get_default_model(self):
        """Test getting default models for each tier and category."""
        for tier in ["high-performance", "balanced", "high-efficiency"]:
            for category in ["text", "coding", "mathematics"]:
                default_model = self.catalog.get_default_model(tier, category)
                self.assertIsInstance(default_model, ModelSpec)
                self.assertEqual(default_model.category, category)
    
    def test_invalid_tier(self):
        """Test handling of invalid tier names."""
        with self.assertRaises(ValueError):
            self.catalog.get_models_for_tier("invalid-tier", "text")
    
    def test_invalid_category(self):
        """Test handling of invalid category names."""
        with self.assertRaises(ValueError):
            self.catalog.get_models_for_tier("balanced", "invalid-category")
    
    def test_model_size_progression(self):
        """Test that models are appropriately sized for their tiers."""
        # Get default text models for comparison
        hp_model = self.catalog.get_default_model("high-performance", "text")
        balanced_model = self.catalog.get_default_model("balanced", "text")
        efficient_model = self.catalog.get_default_model("high-efficiency", "text")
        
        # High-efficiency should have smallest models
        self.assertLessEqual(efficient_model.size_mb, balanced_model.size_mb)
        
        # Generally, high-performance models should be larger or equal
        # (though this may not always be true due to quantization differences)
        self.assertGreaterEqual(hp_model.size_mb, efficient_model.size_mb)


class TestHardwareProfilerTiering(unittest.TestCase):
    """Test hardware profiler tier determination."""
    
    def setUp(self):
        self.profiler = HardwareProfiler()
    
    def test_determine_performance_tier_high_performance(self):
        """Test tier determination for high-performance systems."""
        # >= 10GB VRAM
        tier = self.profiler.determine_performance_tier(12288)  # 12GB
        self.assertEqual(tier, "high-performance")
        
        tier = self.profiler.determine_performance_tier(10240)  # Exactly 10GB
        self.assertEqual(tier, "high-performance")
    
    def test_determine_performance_tier_balanced(self):
        """Test tier determination for balanced systems."""
        # 8-10GB VRAM
        tier = self.profiler.determine_performance_tier(9216)  # 9GB
        self.assertEqual(tier, "balanced")
        
        tier = self.profiler.determine_performance_tier(8192)  # Exactly 8GB
        self.assertEqual(tier, "balanced")
    
    def test_determine_performance_tier_high_efficiency(self):
        """Test tier determination for high-efficiency systems."""
        # < 8GB VRAM or no GPU
        tier = self.profiler.determine_performance_tier(4096)  # 4GB
        self.assertEqual(tier, "high-efficiency")
        
        tier = self.profiler.determine_performance_tier(None)  # No GPU
        self.assertEqual(tier, "high-efficiency")
    
    @patch('psutil.virtual_memory')
    @patch('subprocess.run')
    def test_hardware_profile_with_tier(self, mock_subprocess, mock_memory):
        """Test that hardware profile includes performance tier."""
        # Mock system memory
        mock_memory.return_value.total = 16 * 1024 * 1024 * 1024  # 16GB
        
        # Mock nvidia-smi output for 8GB GPU
        mock_subprocess.return_value.stdout = '''<?xml version="1.0" ?>
        <nvidia_smi_log>
        <gpu id="00000000:01:00.0">
        <product_name>GeForce RTX 3070</product_name>
        <fb_memory_usage>
        <total>8192 MiB</total>
        </fb_memory_usage>
        </gpu>
        </nvidia_smi_log>'''
        mock_subprocess.return_value.returncode = 0
        
        profile = self.profiler.get_hardware_profile()
        
        self.assertEqual(profile.performance_tier, "balanced")
        self.assertEqual(profile.gpu_vram_mb, 8192)


class TestModelManagerTiering(unittest.TestCase):
    """Test model manager tiered selection functionality."""
    
    def setUp(self):
        # Create hardware profile for testing
        self.hardware_profile = HardwareProfile(
            system_ram_mb=16384,
            gpu_vram_mb=8192,
            gpu_name="Test GPU",
            optimal_gpu_layers=20,
            cpu_cores=8,
            performance_tier="balanced"
        )
        
        # Mock llama wrapper
        self.mock_llama_wrapper = Mock()
        
        # Create model manager
        self.model_manager = ModelManager(self.mock_llama_wrapper, self.hardware_profile)
    
    def test_get_recommended_model(self):
        """Test getting recommended models for current tier."""
        # Test text category
        text_model = self.model_manager.get_recommended_model("text")
        self.assertIsInstance(text_model, ModelSpec)
        self.assertEqual(text_model.category, "text")
        
        # Test coding category
        coding_model = self.model_manager.get_recommended_model("coding")
        self.assertIsInstance(coding_model, ModelSpec)
        self.assertEqual(coding_model.category, "coding")
        
        # Test mathematics category
        math_model = self.model_manager.get_recommended_model("mathematics")
        self.assertIsInstance(math_model, ModelSpec)
        self.assertEqual(math_model.category, "mathematics")
    
    def test_get_available_models(self):
        """Test getting all available models for current tier."""
        # All models
        all_models = self.model_manager.get_available_models()
        self.assertGreater(len(all_models), 0)
        
        # Text models only
        text_models = self.model_manager.get_available_models("text")
        self.assertGreater(len(text_models), 0)
        self.assertTrue(all(m.category == "text" for m in text_models))
    
    def test_tier_consistency(self):
        """Test that model manager respects hardware profile tier."""
        # All recommended models should be from the balanced tier
        # (we can't easily verify this without exposing the catalog structure)
        text_model = self.model_manager.get_recommended_model("text")
        coding_model = self.model_manager.get_recommended_model("coding")
        math_model = self.model_manager.get_recommended_model("mathematics")
        
        # At least verify they're valid ModelSpec objects
        self.assertIsInstance(text_model, ModelSpec)
        self.assertIsInstance(coding_model, ModelSpec)
        self.assertIsInstance(math_model, ModelSpec)


class TestModelOrchestratorTiering(unittest.TestCase):
    """Test model orchestrator tiered selection functionality."""
    
    def setUp(self):
        # Create hardware profile
        self.hardware_profile = HardwareProfile(
            system_ram_mb=16384,
            gpu_vram_mb=10240,  # 10GB - high-performance tier
            gpu_name="Test GPU",
            optimal_gpu_layers=25,
            cpu_cores=8,
            performance_tier="high-performance"
        )
        
        # Mock dependencies
        self.mock_llama_wrapper = Mock()
        self.mock_config = Mock()
        
        # Create orchestrator
        self.orchestrator = ModelOrchestrator(
            self.mock_llama_wrapper, 
            self.hardware_profile, 
            self.mock_config
        )
    
    def test_get_tier_info(self):
        """Test getting comprehensive tier information."""
        tier_info = self.orchestrator.get_tier_info()
        
        # Check basic structure
        self.assertEqual(tier_info["performance_tier"], "high-performance")
        self.assertIn("available_categories", tier_info)
        self.assertIn("text_models", tier_info)
        self.assertIn("coding_models", tier_info)
        self.assertIn("mathematics_models", tier_info)
        
        # Check categories
        expected_categories = ["text", "coding", "mathematics"]
        self.assertEqual(sorted(tier_info["available_categories"]), sorted(expected_categories))
        
        # Check that models are present
        self.assertGreater(len(tier_info["text_models"]), 0)
        self.assertGreater(len(tier_info["coding_models"]), 0)
        self.assertGreater(len(tier_info["mathematics_models"]), 0)
    
    def test_get_recommended_model_for_category(self):
        """Test getting recommended models by category."""
        # Test each category
        text_model = self.orchestrator.get_recommended_model_for_category("text")
        coding_model = self.orchestrator.get_recommended_model_for_category("coding")
        math_model = self.orchestrator.get_recommended_model_for_category("mathematics")
        
        # Verify types and categories
        self.assertIsInstance(text_model, ModelSpec)
        self.assertEqual(text_model.category, "text")
        
        self.assertIsInstance(coding_model, ModelSpec)
        self.assertEqual(coding_model.category, "coding")
        
        self.assertIsInstance(math_model, ModelSpec)
        self.assertEqual(math_model.category, "mathematics")


class TestTierIntegration(unittest.TestCase):
    """Integration tests for the complete tiered model selection system."""
    
    @patch('psutil.virtual_memory')
    @patch('subprocess.run')
    def test_end_to_end_tier_selection(self, mock_subprocess, mock_memory):
        """Test complete flow from hardware detection to model selection."""
        # Mock system with 16GB RAM and 12GB GPU (high-performance)
        mock_memory.return_value.total = 16 * 1024 * 1024 * 1024
        
        mock_subprocess.return_value.stdout = '''<?xml version="1.0" ?>
        <nvidia_smi_log>
        <gpu id="00000000:01:00.0">
        <product_name>GeForce RTX 4090</product_name>
        <fb_memory_usage>
        <total>12288 MiB</total>
        </fb_memory_usage>
        </gpu>
        </nvidia_smi_log>'''
        mock_subprocess.return_value.returncode = 0
        
        # Create hardware profiler and get profile
        profiler = HardwareProfiler()
        hardware_profile = profiler.get_hardware_profile()
        
        # Verify tier assignment
        self.assertEqual(hardware_profile.performance_tier, "high-performance")
        
        # Create model manager and test recommendations
        mock_llama_wrapper = Mock()
        model_manager = ModelManager(mock_llama_wrapper, hardware_profile)
        
        # Get recommended models
        text_model = model_manager.get_recommended_model("text")
        coding_model = model_manager.get_recommended_model("coding")
        
        # Verify we get valid models
        self.assertIsInstance(text_model, ModelSpec)
        self.assertIsInstance(coding_model, ModelSpec)
        self.assertEqual(text_model.category, "text")
        self.assertEqual(coding_model.category, "coding")
    
    def test_tier_boundaries(self):
        """Test tier boundary conditions."""
        profiler = HardwareProfiler()
        
        # Test exact boundaries
        tier_7gb = profiler.determine_performance_tier(7168)  # 7GB
        self.assertEqual(tier_7gb, "high-efficiency")
        
        tier_8gb = profiler.determine_performance_tier(8192)  # 8GB
        self.assertEqual(tier_8gb, "balanced")
        
        tier_10gb = profiler.determine_performance_tier(10240)  # 10GB
        self.assertEqual(tier_10gb, "high-performance")
        
        tier_none = profiler.determine_performance_tier(None)  # No GPU
        self.assertEqual(tier_none, "high-efficiency")


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main()
