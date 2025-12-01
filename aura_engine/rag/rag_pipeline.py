"""
Complete RAG (Retrieval-Augmented Generation) pipeline.
"""

import os
import re
import numpy as np
from typing import List, Optional, Tuple
import logging
from pathlib import Path

from .vector_store import VectorStore


class RAGPipeline:
    """
    Complete RAG pipeline for document ingestion, indexing, and retrieval.
    """
    
    def __init__(self, index_dir: str = "rag_data", embedding_dimension: int = 384):
        """
        Initialize the RAG pipeline.
        
        Args:
            index_dir: Directory to store index and metadata files
            embedding_dimension: Dimension of embeddings (384 for sentence-transformers)
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        self.embedding_dimension = embedding_dimension
        self.vector_store = VectorStore(
            index_path=str(self.index_dir / "documents.index"),
            metadata_path=str(self.index_dir / "documents_metadata.pkl")
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Try to load existing index
        try:
            self.vector_store.load()
            self.logger.info("Loaded existing RAG index")
        except FileNotFoundError:
            self.logger.info("No existing RAG index found")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_end = -1
                for i in range(max(0, end - 100), end):
                    if text[i] in '.!?':
                        sentence_end = i + 1
                        break
                
                if sentence_end > start:
                    end = sentence_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        self.logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def simple_embedding(self, texts: List[str]) -> np.ndarray:
        """
        Generate simple embeddings using character frequencies.
        This is a placeholder for actual embedding models like sentence-transformers.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embeddings as numpy array
        """
        embeddings = []
        
        for text in texts:
            # Simple character frequency-based embedding
            char_counts = np.zeros(256, dtype=np.float32)
            
            for char in text.lower():
                char_counts[ord(char) % 256] += 1
            
            # Normalize
            if char_counts.sum() > 0:
                char_counts = char_counts / char_counts.sum()
            
            # Pad or truncate to desired dimension
            if len(char_counts) > self.embedding_dimension:
                embedding = char_counts[:self.embedding_dimension]
            else:
                embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
                embedding[:len(char_counts)] = char_counts
            
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def ingest_document(self, file_path: str) -> None:
        """
        Ingest a single document into the RAG pipeline.
        
        Args:
            file_path: Path to the document file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Read document
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        self.logger.info(f"Ingesting document: {file_path}")
        
        # Chunk the document
        chunks = self.chunk_text(content)
        self.logger.info(f"Created {len(chunks)} chunks from document")
        
        # Generate embeddings
        embeddings = self.simple_embedding(chunks)
        
        # Create or load index if needed
        if self.vector_store.is_empty():
            self.vector_store.create_index(self.embedding_dimension)
        
        # Add to vector store
        self.vector_store.add_documents(embeddings, chunks)
        
        # Save the updated index
        self.vector_store.save()
        
        self.logger.info(f"Successfully ingested document: {os.path.basename(file_path)}")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Query string
            top_k: Number of top chunks to retrieve
            
        Returns:
            Combined context string
        """
        if self.vector_store.is_empty():
            self.logger.warning("No documents in RAG index")
            return ""
        
        # Generate query embedding
        query_embedding = self.simple_embedding([query])[0]
        
        # Search for similar chunks
        distances, retrieved_docs = self.vector_store.search(
            query_embedding, top_k=top_k
        )
        
        # Combine retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            if doc.strip():  # Skip empty documents
                context_parts.append(f"[Context {i+1}]: {doc}")
        
        context = "\n\n".join(context_parts)
        
        self.logger.info(f"Retrieved {len(context_parts)} context chunks for query")
        return context
    
    def get_stats(self) -> dict:
        """Get statistics about the RAG pipeline."""
        try:
            if self.vector_store.is_empty():
                return {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "index_exists": False
                }
            else:
                return {
                    "total_documents": "unknown",  # We don't track original document count
                    "total_chunks": len(self.vector_store.documents),
                    "index_exists": True,
                    "embedding_dimension": self.embedding_dimension
                }
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
