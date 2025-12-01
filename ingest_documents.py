#!/usr/bin/env python3
"""
Document ingestion script for AURA-Engine-Core RAG pipeline.
Creates a local FAISS vector index from documents.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from aura_engine.rag.rag_pipeline import RAGPipeline


def setup_logging(log_level: str) -> None:
    """Configure logging for the application."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='AURA-Engine-Core: Document Ingestion for RAG Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.txt
  %(prog)s document1.txt document2.txt document3.txt
  %(prog)s --index-dir custom_rag_data document.txt
  %(prog)s --log-level DEBUG document.txt
        """
    )
    
    # Required arguments
    parser.add_argument(
        'documents',
        nargs='+',
        help='Document files to ingest into the RAG pipeline'
    )
    
    # Optional arguments
    parser.add_argument(
        '--index-dir',
        type=str,
        default='rag_data',
        help='Directory to store the RAG index (default: rag_data)'
    )
    
    parser.add_argument(
        '--embedding-dimension',
        type=int,
        default=384,
        help='Dimension of embeddings (default: 384)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show RAG pipeline statistics after ingestion'
    )
    
    return parser.parse_args()


def validate_documents(document_paths: list) -> list:
    """
    Validate that document files exist and are readable.
    
    Args:
        document_paths: List of document file paths
        
    Returns:
        List of valid document paths
        
    Raises:
        FileNotFoundError: If any document is not found
    """
    valid_paths = []
    
    for doc_path in document_paths:
        path = Path(doc_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {doc_path}")
        
        # Check if file is readable
        try:
            with open(path, 'r', encoding='utf-8') as f:
                f.read(100)  # Read first 100 characters as test
        except UnicodeDecodeError:
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    f.read(100)
            except Exception as e:
                raise ValueError(f"Cannot read document {doc_path}: {e}")
        except Exception as e:
            raise ValueError(f"Cannot read document {doc_path}: {e}")
        
        valid_paths.append(str(path.resolve()))
    
    return valid_paths


def main() -> int:
    """Main entry point for the document ingestion script."""
    try:
        # Parse arguments
        args = parse_arguments()
        setup_logging(args.log_level)
        
        logger = logging.getLogger(__name__)
        logger.info("Starting AURA-Engine-Core document ingestion...")
        
        # Validate documents
        logger.info(f"Validating {len(args.documents)} document(s)...")
        valid_documents = validate_documents(args.documents)
        logger.info(f"All {len(valid_documents)} document(s) are valid")
        
        # Initialize RAG pipeline
        logger.info(f"Initializing RAG pipeline (index dir: {args.index_dir})...")
        rag_pipeline = RAGPipeline(
            index_dir=args.index_dir,
            embedding_dimension=args.embedding_dimension
        )
        
        # Ingest each document
        success_count = 0
        for doc_path in valid_documents:
            try:
                logger.info(f"Processing: {Path(doc_path).name}")
                rag_pipeline.ingest_document(doc_path)
                success_count += 1
                logger.info(f"✓ Successfully ingested: {Path(doc_path).name}")
            
            except Exception as e:
                logger.error(f"✗ Failed to ingest {Path(doc_path).name}: {e}")
                continue
        
        # Report results
        logger.info(f"Ingestion complete: {success_count}/{len(valid_documents)} documents processed")
        
        if success_count == 0:
            logger.error("No documents were successfully ingested")
            return 1
        
        # Show statistics if requested
        if args.stats:
            stats = rag_pipeline.get_stats()
            logger.info("RAG Pipeline Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("Document ingestion completed successfully!")
        return 0
    
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user")
        return 130
    
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Fatal error during ingestion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
