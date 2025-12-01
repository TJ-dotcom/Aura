"""
Command-line interface for AURA-Engine-Core.
"""

import argparse
import logging
import sys
from typing import Optional

from .models import EngineConfig
from .engine import InferenceEngine


def setup_logging(log_level: str) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='AURA-Engine-Core: Hardware-Aware AI Inference Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Write a Python function to calculate fibonacci numbers"
  %(prog)s --model-path models/llama-7b.gguf "Explain quantum computing"
  %(prog)s --gpu-layers 25 --max-tokens 1024 "Tell me a story"
  %(prog)s --log-level DEBUG "Debug this code: print('hello')"
        """
    )
    
    # Required arguments (optional when showing tier info)
    parser.add_argument(
        'prompt',
        type=str,
        nargs='?',  # Make prompt optional
        help='Input prompt for the AI model'
    )
    
    # Model configuration
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/model.gguf',
        help='Path to the GGUF model file (Phase 1 only, default: models/model.gguf)'
    )
    
    # Phase 2 model configuration
    parser.add_argument(
        '--coder-model',
        type=str,
        help='Path to the coding specialist model (Phase 2)'
    )
    
    parser.add_argument(
        '--writer-model',
        type=str,
        help='Path to the writing specialist model (Phase 2)'
    )
    
    parser.add_argument(
        '--general-model',
        type=str,
        help='Path to the general purpose model (Phase 2)'
    )
    
    parser.add_argument(
        '--force-model',
        type=str,
        choices=['coder', 'writer', 'general'],
        help='Force a specific model type (Phase 2 only)'
    )
    
    parser.add_argument(
        '--llama-cpp-path',
        type=str,
        default='llama.cpp',
        help='Path to llama.cpp binary (default: llama.cpp)'
    )
    
    # Inference parameters
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum number of tokens to generate (default: 512)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (0.0 to 2.0, default: 0.7)'
    )
    
    parser.add_argument(
        '--gpu-layers',
        type=int,
        default=None,
        help='Number of GPU layers (auto-detected if not specified)'
    )
    
    # System configuration
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--no-benchmark',
        action='store_true',
        help='Disable benchmark data collection'
    )
    
    # Future flags (placeholders for Phase 2 and 3)
    parser.add_argument(
        '--rag',
        action='store_true',
        help='Enable RAG (Retrieval-Augmented Generation) - Phase 3 feature'
    )
    
    parser.add_argument(
        '--show-tier-info',
        action='store_true',
        help='Display detailed performance tier information and exit'
    )
    
    parser.add_argument(
        '--auto-download',
        action='store_true',
        help='Automatically download recommended models for current performance tier'
    )
    
    parser.add_argument(
        '--download-category',
        choices=['text', 'coding', 'mathematics'],
        help='Download model for specific category (requires --auto-download)'
    )
    
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        bool: True if inputs are valid
        
    Raises:
        ValueError: If validation fails
    """
    # Validate prompt (only required if not showing tier info or auto-downloading)
    if not args.show_tier_info and not args.auto_download:
        if not args.prompt or not args.prompt.strip():
            raise ValueError("Prompt cannot be empty (unless using --show-tier-info or --auto-download)")
    
    # Validate download category dependency
    if args.download_category and not args.auto_download:
        raise ValueError("--download-category requires --auto-download flag")
    
    # Validate temperature
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("Temperature must be between 0.0 and 2.0")
    
    # Validate max_tokens
    if args.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    
    # Validate gpu_layers if specified
    if args.gpu_layers is not None and args.gpu_layers < 0:
        raise ValueError("gpu_layers must be non-negative")
    
    return True


def create_engine_config(args: argparse.Namespace) -> EngineConfig:
    """
    Create engine configuration from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        EngineConfig: Engine configuration object
    """
    return EngineConfig(
        models_dir="models",  # Default models directory
        llama_cpp_path=args.llama_cpp_path,
        faiss_index_path=None,  # Will be set in Phase 3
        enable_benchmarking=not args.no_benchmark,
        log_level=args.log_level,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )


def main() -> int:
    """
    Main entry point for the CLI application.
    
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_inputs(args)
        
        # Setup logging
        setup_logging(args.log_level)
        logger = logging.getLogger(__name__)
        
        logger.info("AURA-Engine-Core starting...")
        if args.prompt:
            logger.info(f"Prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
        else:
            logger.info("Mode: Tier information display")
        
        # Create engine configuration
        config = create_engine_config(args)
        
        # Initialize inference engine (for --show-tier-info, we don't need llama.cpp)
        if args.show_tier_info:
            # Create a minimal engine just for hardware profiling
            logger.info("Initializing hardware profiling for tier information...")
            from .hardware import HardwareProfiler
            profiler = HardwareProfiler()
            hardware_profile = profiler.get_hardware_profile()
            
            print(f"\n🎯 PERFORMANCE TIER: {hardware_profile.performance_tier.upper()}")
            print(f"📊 Hardware Profile: {hardware_profile}")
            print("\n📦 AVAILABLE MODELS BY CATEGORY:")
            
            from .orchestrator.model_catalog import ModelCatalog
            catalog = ModelCatalog()
            
            for category in catalog.get_available_categories():
                models = catalog.get_models_for_tier(hardware_profile.performance_tier, category)
                print(f"\n  {category.title()} Models ({len(models)} available):")
                for i, model in enumerate(models, 1):
                    marker = "⭐" if i == 1 else "  "
                    print(f"    {marker} {model.name}")
                    print(f"      Size: {model.size_mb}MB | {model.description}")
                    
            print(f"\n⭐ Default models are automatically selected for each category")
            print(f"💡 Your tier is determined by available VRAM: {hardware_profile.gpu_vram_mb}MB")
            
            return 0  # Exit after showing tier info
        
        # Handle --auto-download flag
        if args.auto_download:
            logger.info("Auto-downloading recommended models...")
            
            # Get hardware profile for downloads
            from .hardware import HardwareProfiler
            profiler = HardwareProfiler()
            hardware_profile = profiler.get_hardware_profile()
            
            from .orchestrator.model_manager import ModelManager
            from .llama_wrapper import LlamaWrapper
            
            # Create minimal components for model download
            mock_wrapper = None  # We don't need llama wrapper for downloading
            model_manager = ModelManager(mock_wrapper, hardware_profile)
            
            if args.download_category:
                # Download specific category
                print(f"\n📥 Auto-downloading {args.download_category} model for {hardware_profile.performance_tier} tier...")
                model_path = model_manager.auto_select_model(args.download_category)
                if model_path:
                    print(f"✅ Downloaded: {model_path}")
                else:
                    print(f"❌ Failed to download {args.download_category} model")
            else:
                # Download all categories
                print(f"\n📥 Auto-downloading all recommended models for {hardware_profile.performance_tier} tier...")
                configured_models = model_manager.auto_configure_for_tier()
                
                if configured_models:
                    print(f"\n✅ Successfully configured {len(configured_models)} models:")
                    for category, path in configured_models.items():
                        print(f"   {category.title()}: {path}")
                else:
                    print("❌ Failed to download any models")
            
            return 0  # Exit after downloading
        
        # Regular engine initialization
        logger.info("Initializing inference engine...")
        engine = InferenceEngine(config)
        
        # Check if Phase 2 models are provided
        model_paths = None
        if args.coder_model or args.writer_model or args.general_model:
            from .models import ModelType
            model_paths = {}
            if args.coder_model:
                model_paths[ModelType.CODER] = args.coder_model
            if args.writer_model:
                model_paths[ModelType.WRITER] = args.writer_model
            if args.general_model:
                model_paths[ModelType.GENERAL] = args.general_model
            
            logger.info("Phase 2 mode: Using model orchestration")
        
        engine.initialize(model_paths)
        
        # Display performance tier information for regular runs
        if hasattr(engine, 'orchestrator') and engine.orchestrator:
            tier_info = engine.orchestrator.get_tier_info()
            print(f"\n🎯 Performance Tier: {tier_info['performance_tier'].upper()}")
            print(f"📊 Hardware: {engine.hardware_profile}")
            
            # Show recommended models for each category
            for category in tier_info['available_categories']:
                recommended = engine.orchestrator.get_recommended_model_for_category(category)
                if recommended:
                    print(f"📦 Recommended {category.title()} Model: {recommended.name} ({recommended.size_mb}MB)")
            print()  # Add spacing
        else:
            print(f"\n📊 Hardware Profile: {engine.hardware_profile}")
            print(f"🎯 Performance Tier: {engine.hardware_profile.performance_tier.upper()}")
            print()  # Add spacing
        
        # Run inference
        logger.info("Starting inference...")
        
        # Determine force model type if specified
        force_model_type = None
        if args.force_model:
            from .models import ModelType
            force_model_type = ModelType(args.force_model)
        
        # Process prompt with or without RAG
        result = engine.process_prompt(
            prompt=args.prompt,
            model_path=args.model_path,
            gpu_layers_override=args.gpu_layers,
            force_model_type=force_model_type,
            enable_rag=args.rag
        )
        
        # Output results
        print("\\n" + "="*80)
        print("INFERENCE RESULT")
        print("="*80)
        print(result.response)
        print("\\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)
        print(f"Model: {result.model_used.value}")
        print(f"Time to First Token: {result.metrics.ttft_ms:.1f}ms")
        print(f"Tokens per Second: {result.metrics.tokens_per_second:.1f}")
        print(f"Peak RAM Usage: {result.metrics.peak_ram_mb}MB")
        print(f"Peak VRAM Usage: {result.metrics.peak_vram_mb}MB")
        print(f"Model Load Time: {result.metrics.model_load_time_s:.2f}s")
        print("="*80)
        
        # Display RAG context if available
        if result.rag_context:
            print("RAG Context:", result.rag_context)
        
        logger.info("Inference completed successfully")
        return 0
        
    except KeyboardInterrupt:
        print("\\nOperation cancelled by user")
        return 130  # Standard exit code for SIGINT
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
        
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        return 2
        
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        logging.getLogger(__name__).exception("Unexpected error occurred")
        return 3


if __name__ == "__main__":
    sys.exit(main())