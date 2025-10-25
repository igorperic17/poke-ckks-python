"""
Approximate Nearest Neighbor (ANN) search for efficient similarity search.

This module provides fast vector similarity search using FAISS indexing,
suitable for large-scale non-encrypted search scenarios.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Optional dependency - gracefully handle if not installed
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None  # type: ignore


@dataclass
class IndexedVector:
    """Container for a vector with metadata."""
    vector_id: int
    identifier: str
    metadata: Optional[dict] = None


class ANNSearchIndex:
    """
    Approximate Nearest Neighbor search index using FAISS.
    
    This class provides efficient similarity search for high-dimensional vectors
    using approximate nearest neighbor algorithms. Unlike the CKKS homomorphic
    search, this operates on plaintext vectors for maximum speed.
    
    Features:
    - Fast insertion and search using FAISS indexing
    - Support for inner product (cosine similarity for normalized vectors)
    - Automatic dimension detection
    - Metadata storage for vectors
    
    Example:
        >>> index = ANNSearchIndex(dimension=128)
        >>> # Insert vectors
        >>> index.insert(vector1, identifier="doc_1")
        >>> index.insert(vector2, identifier="doc_2")
        >>> # Search for top-3 similar vectors
        >>> results = index.search(query_vector, k=3)
        >>> for idx, score in results:
        ...     print(f"ID: {idx}, Score: {score}")
    """
    
    def __init__(
        self,
        dimension: int,
        metric: str = "inner_product",
        index_type: str = "flat"
    ):
        """
        Initialize the ANN search index.
        
        Args:
            dimension: Dimensionality of the vectors to index.
            metric: Distance metric to use. Options:
                   - "inner_product": Inner product (for normalized vectors, this is cosine similarity)
                   - "l2": Euclidean distance
            index_type: Type of FAISS index to use. Options:
                       - "flat": Exact search (brute force), guaranteed exact results
                       - "ivf": Inverted file index (approximate, faster for large datasets)
                       - "hnsw": Hierarchical Navigable Small World (fast approximate search)
        
        Raises:
            ImportError: If faiss is not installed.
        """
        if not HAS_FAISS:
            raise ImportError(
                "faiss is required for ANN search. "
                "Install it with: pip install faiss-cpu (or faiss-gpu for GPU support)"
            )
        
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        
        # Create FAISS index based on metric and type
        self._index = self._create_index()
        
        # Storage for vector metadata
        self._vectors: List[IndexedVector] = []
        self._id_to_position: dict[str, int] = {}
        self._next_id = 0
    
    def _create_index(self) -> "faiss.Index":
        """Create the appropriate FAISS index based on configuration."""
        if self.metric == "inner_product":
            if self.index_type == "flat":
                # Exact inner product search
                index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "ivf":
                # Approximate search with inverted file
                quantizer = faiss.IndexFlatIP(self.dimension)
                # Use sqrt(n) clusters as a rule of thumb, but start with 100
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, faiss.METRIC_INNER_PRODUCT)
            elif self.index_type == "hnsw":
                # HNSW for fast approximate search
                index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            else:
                raise ValueError(f"Unknown index_type: {self.index_type}")
        
        elif self.metric == "l2":
            if self.index_type == "flat":
                index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, 100, faiss.METRIC_L2)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_L2)
            else:
                raise ValueError(f"Unknown index_type: {self.index_type}")
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return index
    
    def insert(
        self,
        vector: np.ndarray,
        identifier: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> int:
        """
        Insert a vector into the index.
        
        Args:
            vector: The vector to insert (will be normalized if using inner_product metric).
            identifier: Optional string identifier for the vector.
                       If not provided, will use auto-incrementing integer as string.
            metadata: Optional dictionary of metadata to associate with the vector.
        
        Returns:
            The integer ID assigned to this vector in the index.
        
        Example:
            >>> index = ANNSearchIndex(dimension=128)
            >>> vid = index.insert(vector, identifier="doc_42", metadata={"title": "Example"})
        """
        # Validate dimension
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match index dimension {self.dimension}"
            )
        
        # Auto-generate identifier if not provided
        if identifier is None:
            identifier = f"vec_{self._next_id}"
        
        # Check for duplicate identifiers
        if identifier in self._id_to_position:
            raise ValueError(f"Vector with identifier '{identifier}' already exists in index")
        
        # Prepare vector for insertion
        vec = vector.astype(np.float32).reshape(1, -1)
        
        # Normalize if using inner product (for cosine similarity)
        if self.metric == "inner_product":
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        
        # For IVF indices, train if needed
        if self.index_type == "ivf" and not self._index.is_trained:
            # Train with first 100 vectors or when we have enough
            if len(self._vectors) >= 100:
                training_vectors = np.array([v.vector for v in self._vectors[-100:]], dtype=np.float32)
                self._index.train(training_vectors)
        
        # Add to FAISS index
        self._index.add(vec)
        
        # Store metadata
        vector_id = self._next_id
        self._next_id += 1
        
        indexed_vec = IndexedVector(
            vector_id=vector_id,
            identifier=identifier,
            metadata=metadata
        )
        self._vectors.append(indexed_vec)
        self._id_to_position[identifier] = len(self._vectors) - 1
        
        return vector_id
    
    def insert_batch(
        self,
        vectors: np.ndarray,
        identifiers: Optional[List[str]] = None,
        metadata: Optional[List[dict]] = None
    ) -> List[int]:
        """
        Insert multiple vectors at once (more efficient than individual inserts).
        
        Args:
            vectors: Array of shape (n_vectors, dimension).
            identifiers: Optional list of string identifiers (must match length of vectors).
            metadata: Optional list of metadata dicts (must match length of vectors).
        
        Returns:
            List of integer IDs assigned to the vectors.
        
        Example:
            >>> vectors = np.random.randn(100, 128)
            >>> ids = index.insert_batch(vectors)
        """
        n_vectors = len(vectors)
        
        # Prepare identifiers
        if identifiers is None:
            identifiers = [f"vec_{self._next_id + i}" for i in range(n_vectors)]
        elif len(identifiers) != n_vectors:
            raise ValueError("Length of identifiers must match number of vectors")
        
        # Prepare metadata
        if metadata is None:
            metadata = [None] * n_vectors
        elif len(metadata) != n_vectors:
            raise ValueError("Length of metadata must match number of vectors")
        
        # Check for duplicates
        for ident in identifiers:
            if ident in self._id_to_position:
                raise ValueError(f"Vector with identifier '{ident}' already exists in index")
        
        # Prepare vectors
        vecs = vectors.astype(np.float32)
        
        # Validate dimensions
        if vecs.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vecs.shape[1]} does not match index dimension {self.dimension}"
            )
        
        # Normalize if using inner product
        if self.metric == "inner_product":
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = np.where(norms > 0, vecs / norms, vecs)
        
        # Train IVF index if needed
        if self.index_type == "ivf" and not self._index.is_trained:
            if n_vectors >= 100:
                self._index.train(vecs)
            elif len(self._vectors) + n_vectors >= 100:
                # Combine with existing vectors for training
                existing = np.array([v.vector for v in self._vectors], dtype=np.float32)
                training_data = np.vstack([existing, vecs])
                self._index.train(training_data)
        
        # Add to FAISS index
        self._index.add(vecs)
        
        # Store metadata
        vector_ids = []
        for i, (ident, meta) in enumerate(zip(identifiers, metadata)):
            vector_id = self._next_id + i
            indexed_vec = IndexedVector(
                vector_id=vector_id,
                identifier=ident,
                metadata=meta
            )
            self._vectors.append(indexed_vec)
            self._id_to_position[ident] = len(self._vectors) - 1
            vector_ids.append(vector_id)
        
        self._next_id += n_vectors
        
        return vector_ids
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for the k most similar vectors.
        
        Args:
            query: Query vector to search for.
            k: Number of nearest neighbors to return.
        
        Returns:
            List of (identifier, score) tuples, ordered by descending similarity.
            For inner_product metric, higher scores are better.
            For l2 metric, lower scores are better (distances).
        
        Example:
            >>> results = index.search(query_vector, k=5)
            >>> for identifier, score in results:
            ...     print(f"{identifier}: {score:.4f}")
        """
        if len(self._vectors) == 0:
            return []
        
        # Validate dimension
        if len(query) != self.dimension:
            raise ValueError(
                f"Query dimension {len(query)} does not match index dimension {self.dimension}"
            )
        
        # Prepare query
        q = query.astype(np.float32).reshape(1, -1)
        
        # Normalize if using inner product
        if self.metric == "inner_product":
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm
        
        # Limit k to available vectors
        k = min(k, len(self._vectors))
        
        # Search
        scores, indices = self._index.search(q, k)
        
        # Convert to list of (identifier, score) tuples
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._vectors):
                # FAISS can return -1 for missing results
                continue
            
            vec_info = self._vectors[idx]
            results.append((vec_info.identifier, float(score)))
        
        return results
    
    def get_vector_info(self, identifier: str) -> Optional[IndexedVector]:
        """
        Retrieve information about a vector by its identifier.
        
        Args:
            identifier: The string identifier of the vector.
        
        Returns:
            IndexedVector object with the vector's metadata, or None if not found.
        """
        pos = self._id_to_position.get(identifier)
        if pos is None:
            return None
        return self._vectors[pos]
    
    @property
    def size(self) -> int:
        """Get the number of vectors in the index."""
        return len(self._vectors)
    
    def __len__(self) -> int:
        """Get the number of vectors in the index."""
        return self.size
    
    def __repr__(self) -> str:
        return (
            f"ANNSearchIndex(dimension={self.dimension}, "
            f"metric={self.metric}, "
            f"index_type={self.index_type}, "
            f"size={self.size})"
        )
