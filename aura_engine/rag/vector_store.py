"""
FAISS-based vector store for document retrieval.
"""

import os
import faiss
import numpy as np
import pickle
from typing import List, Tuple, Optional
import logging


class VectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    """
    
    def __init__(self, index_path: str, metadata_path: str):
        """
        Initialize the vector store.
        
        Args:
            index_path: Path to save/load the FAISS index
            metadata_path: Path to save/load document metadata
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self.logger = logging.getLogger(__name__)
    
    def create_index(self, dimension: int) -> None:
        """
        Create a new FAISS index.
        
        Args:
            dimension: Dimension of the embeddings
        """
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.logger.info(f"Created new FAISS index with dimension {dimension}")
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str]) -> None:
        """
        Add document embeddings and their text to the index.
        
        Args:
            embeddings: Document embeddings as numpy array
            documents: List of document texts corresponding to embeddings
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        
        self.logger.info(f"Added {len(documents)} documents to the index")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[str]]:
        """
        Search for the most similar documents.
        
        Args:
            query_embedding: Query embedding as numpy array
            top_k: Number of top results to return
            
        Returns:
            Tuple of (distances, retrieved_documents)
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Ensure query is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get corresponding documents
        retrieved_docs = []
        for idx in indices[0]:  # indices is 2D, we want first row
            if 0 <= idx < len(self.documents):
                retrieved_docs.append(self.documents[idx])
            else:
                retrieved_docs.append("")
        
        self.logger.debug(f"Retrieved {len(retrieved_docs)} documents with distances: {distances[0]}")
        return distances[0], retrieved_docs
    
    def save(self) -> None:
        """Save the index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save document metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        self.logger.info(f"Saved index to {self.index_path} and metadata to {self.metadata_path}")
    
    def load(self) -> None:
        """Load the index and metadata from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load document metadata
        with open(self.metadata_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        self.logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.documents)} documents")
    
    def is_empty(self) -> bool:
        """Check if the vector store is empty."""
        return self.index is None or self.index.ntotal == 0
