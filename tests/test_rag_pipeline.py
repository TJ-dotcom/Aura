"""
Tests for RAG (Retrieval-Augmented Generation) functionality - Phase 3.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import os

from aura_engine.rag import VectorStore, RAGPipeline


class TestVectorStore:
    """Test the VectorStore component."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.temp_dir, "test.index")
        self.metadata_path = os.path.join(self.temp_dir, "test_metadata.pkl")
        self.store = VectorStore(self.index_path, self.metadata_path)
    
    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_index_and_add_documents(self):
        """Test creating index and adding documents with metadata."""
        dimension = 128
        self.store.create_index(dimension)
        
        # Test data
        embeddings = np.random.random((3, dimension)).astype('float32')
        documents = ["Document 1", "Document 2", "Document 3"]
        
        self.store.add_documents(embeddings, documents)
        
        assert self.store.index.ntotal == 3
        assert len(self.store.documents) == 3
        assert self.store.documents[0] == "Document 1"
    
    def test_search_with_documents(self):
        """Test search functionality returning documents."""
        dimension = 64
        self.store.create_index(dimension)
        
        # Known embeddings and documents
        embeddings = np.array([
            [1.0] * dimension,
            [0.5] * dimension,
            [0.0] * dimension
        ], dtype='float32')
        
        documents = [
            "This is the first document",
            "This is the second document", 
            "This is the third document"
        ]
        
        self.store.add_documents(embeddings, documents)
        
        # Search
        query = np.array([1.0] * dimension, dtype='float32')
        distances, retrieved_docs = self.store.search(query, top_k=2)
        
        assert len(retrieved_docs) == 2
        assert retrieved_docs[0] == "This is the first document"
    
    def test_save_and_load_complete(self):
        """Test saving and loading both index and metadata."""
        dimension = 128
        self.store.create_index(dimension)
        
        # Add test data
        embeddings = np.random.random((2, dimension)).astype('float32')
        documents = ["Test doc 1", "Test doc 2"]
        
        self.store.add_documents(embeddings, documents)
        self.store.save()
        
        # Verify files exist
        assert os.path.exists(self.index_path)
        assert os.path.exists(self.metadata_path)
        
        # Load in new store
        new_store = VectorStore(self.index_path, self.metadata_path)
        new_store.load()
        
        assert new_store.index.ntotal == 2
        assert len(new_store.documents) == 2
        assert new_store.documents == documents
    
    def test_is_empty(self):
        """Test empty store detection."""
        assert self.store.is_empty()
        
        self.store.create_index(128)
        assert self.store.is_empty()  # Still empty, no documents added
        
        embeddings = np.random.random((1, 128)).astype('float32')
        documents = ["Test"]
        self.store.add_documents(embeddings, documents)
        
        assert not self.store.is_empty()


class TestRAGPipeline:
    """Test the complete RAG pipeline."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = RAGPipeline(index_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_text_chunking(self):
        """Test document chunking functionality."""
        text = "This is a test document. " * 100  # Create long text
        
        chunks = self.pipeline.chunk_text(text, chunk_size=200, overlap=50)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 250 for chunk in chunks)  # Allow some variance
        
        # Test short text
        short_text = "Short text."
        chunks = self.pipeline.chunk_text(short_text)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."
    
    def test_simple_embedding(self):
        """Test simple embedding generation."""
        texts = ["This is test text", "Another test document"]
        
        embeddings = self.pipeline.simple_embedding(texts)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == self.pipeline.embedding_dimension
        assert embeddings.dtype == np.float32
    
    def test_document_ingestion(self):
        """Test document ingestion from file."""
        # Create test document file
        test_file = os.path.join(self.temp_dir, "test_doc.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("This is a test document for RAG ingestion. " * 20)
        
        # Ingest document
        self.pipeline.ingest_document(test_file)
        
        # Verify index was created and populated
        assert not self.pipeline.vector_store.is_empty()
        stats = self.pipeline.get_stats()
        assert stats['total_chunks'] > 0
        assert stats['index_exists']
    
    def test_context_retrieval(self):
        """Test context retrieval functionality."""
        # Create test document
        test_file = os.path.join(self.temp_dir, "knowledge.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("""
            The AURA-Engine-Core has three main phases.
            Phase 1 focuses on hardware-aware inference.
            Phase 2 implements model orchestration.
            Phase 3 adds RAG capabilities with FAISS.
            The system is designed for elite engineering demonstrations.
            """)
        
        # Ingest and test retrieval
        self.pipeline.ingest_document(test_file)
        
        context = self.pipeline.retrieve_context("What is Phase 1?", top_k=2)
        
        assert context is not None
        assert len(context) > 0
        assert "Phase 1" in context or "hardware" in context.lower()
    
    def test_empty_index_handling(self):
        """Test handling of empty index."""
        context = self.pipeline.retrieve_context("Any query")
        assert context == ""
        
        stats = self.pipeline.get_stats()
        assert stats['total_chunks'] == 0
        assert not stats['index_exists']
    
    def test_file_not_found_error(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            self.pipeline.ingest_document("nonexistent_file.txt")
    
    def test_stats_reporting(self):
        """Test statistics reporting."""
        # Test empty pipeline
        stats = self.pipeline.get_stats()
        expected_keys = {'total_documents', 'total_chunks', 'index_exists'}
        assert all(key in stats for key in expected_keys)
        
        # Test with documents
        test_file = os.path.join(self.temp_dir, "stats_test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Test content for statistics.")
        
        self.pipeline.ingest_document(test_file)
        stats = self.pipeline.get_stats()
        
        assert stats['total_chunks'] > 0
        assert stats['index_exists']
        assert 'embedding_dimension' in stats


class TestRAGIntegration:
    """Integration tests for RAG functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = RAGPipeline(index_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_rag_workflow(self):
        """Test complete RAG workflow from ingestion to retrieval."""
        # Create a knowledge base
        knowledge_file = os.path.join(self.temp_dir, "knowledge_base.txt")
        knowledge_content = """
        The AURA-Engine-Core is a sophisticated AI inference system.
        
        Phase 1 Implementation:
        - Hardware profiling with RAM and GPU detection
        - Dynamic GPU layer calculation
        - Integration with llama.cpp
        - Performance monitoring and benchmarking
        
        Phase 2 Implementation:
        - Intelligent prompt routing
        - Model orchestration with memory management
        - Support for specialized models (coder, writer, general)
        
        Phase 3 Implementation:
        - FAISS vector store integration
        - Document chunking and embedding
        - Context-aware retrieval
        - RAG-augmented inference
        
        Key Features:
        - CLI-first approach
        - Comprehensive error handling
        - 100% test coverage
        - Performance benchmarking
        """
        
        with open(knowledge_file, 'w', encoding='utf-8') as f:
            f.write(knowledge_content)
        
        # Ingest the document
        self.pipeline.ingest_document(knowledge_file)
        
        # Test various queries
        test_cases = [
            ("What is Phase 1?", ["Phase 1", "hardware", "GPU"]),
            ("How does model orchestration work?", ["Phase 2", "orchestration", "model"]),
            ("What are the key features?", ["features", "CLI", "test"]),
            ("Tell me about FAISS", ["Phase 3", "FAISS", "vector"])
        ]
        
        for query, expected_terms in test_cases:
            context = self.pipeline.retrieve_context(query, top_k=2)
            
            # Verify we got some context
            assert context is not None
            assert len(context) > 0
            
            # Verify at least one expected term appears (case insensitive)
            context_lower = context.lower()
            assert any(term.lower() in context_lower for term in expected_terms), \
                f"None of {expected_terms} found in context for query: {query}"
    
    def test_multiple_document_ingestion(self):
        """Test ingesting multiple documents."""
        # Create multiple test documents
        docs = {
            "doc1.txt": "This document covers Phase 1 implementation details.",
            "doc2.txt": "This document explains Phase 2 model orchestration.",
            "doc3.txt": "This document describes Phase 3 RAG integration."
        }
        
        for filename, content in docs.items():
            filepath = os.path.join(self.temp_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.pipeline.ingest_document(filepath)
        
        # Verify all documents are indexed
        stats = self.pipeline.get_stats()
        assert stats['total_chunks'] >= 3  # At least one chunk per document
        
        # Test retrieval across documents
        phase1_context = self.pipeline.retrieve_context("Phase 1")
        phase2_context = self.pipeline.retrieve_context("Phase 2") 
        phase3_context = self.pipeline.retrieve_context("Phase 3")
        
        # Each should retrieve different relevant context
        assert "Phase 1" in phase1_context
        assert "Phase 2" in phase2_context  
        assert "Phase 3" in phase3_context


if __name__ == "__main__":
    pytest.main([__file__])
