import os
import faiss
import numpy as np
from typing import List

class DocumentProcessor:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.index = None

    def create_index(self, dimension: int):
        """Create a new FAISS index."""
        self.index = faiss.IndexFlatL2(dimension)

    def add_documents(self, embeddings: np.ndarray):
        """Add document embeddings to the FAISS index."""
        if self.index is None:
            raise ValueError("Index has not been created. Call create_index first.")
        self.index.add(embeddings)

    def save_index(self):
        """Save the FAISS index to disk."""
        if self.index is None:
            raise ValueError("Index has not been created.")
        faiss.write_index(self.index, self.index_path)

    def load_index(self):
        """Load a FAISS index from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found at {self.index_path}")
        self.index = faiss.read_index(self.index_path)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """Search the FAISS index for the nearest neighbors of a query embedding."""
        if self.index is None:
            raise ValueError("Index has not been loaded or created.")
        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor(index_path="vector_index.faiss")
    processor.create_index(dimension=128)

    # Example embeddings (randomly generated for demonstration purposes)
    embeddings = np.random.random((10, 128)).astype('float32')
    processor.add_documents(embeddings)
    processor.save_index()

    # Load and search
    processor.load_index()
    query = np.random.random((1, 128)).astype('float32')
    distances, indices = processor.search(query)
    print("Distances:", distances)
    print("Indices:", indices)
