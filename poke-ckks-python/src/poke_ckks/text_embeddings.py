"""
Text embedding utilities for converting text to vectors.

This module provides functionality to convert text into numerical vectors
suitable for homomorphic encryption and similarity search.
"""

import numpy as np
from typing import Optional

# Optional dependency - gracefully handle if not installed
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore


class TextEmbedder:
    """
    Text embedding utility using state-of-the-art sentence transformers.
    
    This class provides a convenient interface for converting text into
    semantic vector embeddings using pre-trained transformer models.
    
    Example:
        >>> embedder = TextEmbedder(model_name='all-MiniLM-L6-v2', target_dimension=128)
        >>> vector = embedder.embed("Example text for embedding")
        >>> vector.shape
        (128,)
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        target_dimension: Optional[int] = None,
        cache_folder: Optional[str] = None
    ):
        """
        Initialize the text embedder.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is 'all-MiniLM-L6-v2' which produces 384-dim embeddings.
            target_dimension: Optional dimension to reduce/expand embeddings to.
                            If None, uses the model's native dimension.
            cache_folder: Optional folder to cache downloaded models.
        
        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required for text embeddings. "
                "Install it with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.target_dimension = target_dimension
        self._model: Optional[SentenceTransformer] = None
        self._cache_folder = cache_folder
    
    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy-load the sentence transformer model.
        
        The model is only loaded on first use to avoid unnecessary initialization.
        """
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=self._cache_folder
            )
        return self._model
    
    @property
    def native_dimension(self) -> int:
        """Get the native embedding dimension of the model."""
        # Load model to get dimension
        return self.model.get_sentence_embedding_dimension()
    
    def embed(self, text: str) -> np.ndarray:
        """
        Convert text to a normalized vector embedding.
        
        Args:
            text: Input text to embed.
        
        Returns:
            Normalized numpy array of shape (dimension,).
            If target_dimension is set, the embedding will be adjusted to that size.
        
        Example:
            >>> embedder = TextEmbedder()
            >>> vector = embedder.embed("Machine learning is fascinating")
            >>> np.linalg.norm(vector)  # Should be close to 1.0
            1.0
        """
        # Get embedding from model (already normalized by default)
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Adjust dimension if needed
        if self.target_dimension is not None:
            embedding = self._adjust_dimension(embedding, self.target_dimension)
        
        return embedding
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Convert multiple texts to vector embeddings efficiently.
        
        Args:
            texts: List of input texts to embed.
        
        Returns:
            Normalized numpy array of shape (len(texts), dimension).
        
        Example:
            >>> embedder = TextEmbedder()
            >>> vectors = embedder.embed_batch(["First text", "Second text"])
            >>> vectors.shape
            (2, 384)
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10  # Show progress for large batches
        )
        
        # Adjust dimensions if needed
        if self.target_dimension is not None:
            embeddings = np.array([
                self._adjust_dimension(emb, self.target_dimension)
                for emb in embeddings
            ])
        
        return embeddings
    
    @staticmethod
    def _adjust_dimension(embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Adjust embedding dimension by truncation or padding.
        
        Args:
            embedding: Input embedding vector.
            target_dim: Desired output dimension.
        
        Returns:
            Adjusted and re-normalized embedding vector.
        """
        native_dim = len(embedding)
        
        if target_dim < native_dim:
            # Truncate to target dimension
            adjusted = embedding[:target_dim]
        elif target_dim > native_dim:
            # Pad with zeros
            adjusted = np.zeros(target_dim, dtype=embedding.dtype)
            adjusted[:native_dim] = embedding
        else:
            # Already correct dimension
            return embedding
        
        # Re-normalize after adjustment
        norm = np.linalg.norm(adjusted)
        if norm > 0:
            adjusted = adjusted / norm
        
        return adjusted


def create_embedder(
    model_name: str = 'all-MiniLM-L6-v2',
    target_dimension: Optional[int] = None
) -> TextEmbedder:
    """
    Factory function to create a TextEmbedder instance.
    
    This is a convenience function for quick embedder creation.
    
    Args:
        model_name: Name of the sentence-transformers model.
        target_dimension: Optional dimension to adjust embeddings to.
    
    Returns:
        Configured TextEmbedder instance.
    
    Example:
        >>> embedder = create_embedder(target_dimension=128)
        >>> vector = embedder.embed("Hello world")
    """
    return TextEmbedder(model_name=model_name, target_dimension=target_dimension)
