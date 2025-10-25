"""
Distance-preserving encrypted search using Scale-and-Perturb (SAP) with FAISS.

This module implements privacy-preserving similarity search by encrypting vectors
in a way that preserves their relative distances, allowing FAISS to index and
search over encrypted data.

The encryption scheme uses:
1. Random orthogonal rotation (preserves distances exactly)
2. Scaling factor (amplifies values)
3. Random noise (provides security)

This provides much better performance than fully homomorphic encryption while
still offering reasonable privacy guarantees for similarity search applications.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import pickle

from .ann_search import ANNSearchIndex, IndexedVector

# Optional dependency check
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None  # type: ignore


@dataclass
class SAPEncryptionKey:
    """
    Encryption key for Scale-and-Perturb scheme.
    
    Attributes:
        rotation_matrix: Orthogonal matrix for random rotation (d x d)
        scale_factor: Scaling multiplier
        noise_std: Standard deviation of Gaussian noise
        dimension: Vector dimension
    """
    rotation_matrix: np.ndarray
    scale_factor: float
    noise_std: float
    dimension: int
    
    def save(self, filepath: str) -> None:
        """Save encryption key to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str) -> 'SAPEncryptionKey':
        """Load encryption key from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class EncryptedANNSearch:
    """
    Privacy-preserving ANN search using Scale-and-Perturb encryption.
    
    This class allows building a searchable encrypted index where:
    - Vectors are encrypted before insertion
    - FAISS can still find approximate nearest neighbors
    - The database server never sees plaintext vectors
    - Search results preserve relative ranking
    
    Security properties:
    - Individual vector values are obfuscated by rotation + noise
    - Exact distances are hidden (only approximate ordering preserved)
    - Statistical attacks are mitigated by rotation
    
    Performance:
    - Much faster than fully homomorphic encryption (CKKS)
    - Scales to millions of vectors
    - Search speed similar to plaintext FAISS
    
    Example:
        >>> # Generate encryption key
        >>> key = generate_sap_key(dimension=128, scale_factor=10.0, noise_std=0.1)
        >>> 
        >>> # Server-side: Create encrypted index
        >>> enc_index = EncryptedANNSearch(key=key, index_type="flat")
        >>> 
        >>> # Client-side: Encrypt and insert vectors
        >>> encrypted_vec = enc_index.encrypt_vector(my_vector)
        >>> enc_index.insert(encrypted_vec, identifier="doc_1")
        >>> 
        >>> # Client-side: Encrypt query and search
        >>> encrypted_query = enc_index.encrypt_vector(query_vector)
        >>> results = enc_index.search(encrypted_query, k=5)
    """
    
    def __init__(
        self,
        key: SAPEncryptionKey,
        metric: str = "inner_product",
        index_type: str = "flat"
    ):
        """
        Initialize encrypted ANN search index.
        
        Args:
            key: SAP encryption key.
            metric: Distance metric ("inner_product" or "l2").
            index_type: FAISS index type ("flat", "ivf", "hnsw").
        
        Raises:
            ImportError: If faiss is not installed.
        """
        if not HAS_FAISS:
            raise ImportError(
                "faiss is required for ANN search. "
                "Install it with: pip install faiss-cpu"
            )
        
        self.key = key
        self.metric = metric
        self.index_type = index_type
        
        # Create underlying ANN index for encrypted vectors
        self._index = ANNSearchIndex(
            dimension=key.dimension,
            metric=metric,
            index_type=index_type
        )
    
    def encrypt_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Encrypt a vector using Scale-and-Perturb scheme.
        
        The encryption: E(x) = scale * (M @ x) + noise
        where M is a random orthogonal matrix.
        
        Args:
            vector: Plaintext vector to encrypt.
        
        Returns:
            Encrypted vector with same dimension.
        
        Example:
            >>> encrypted = enc_index.encrypt_vector(my_vector)
        """
        if len(vector) != self.key.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match key dimension {self.key.dimension}"
            )
        
        # Ensure vector is float64 for precision
        vec = vector.astype(np.float64)
        
        # Step 1: Rotate with orthogonal matrix (preserves norm and angles)
        rotated = self.key.rotation_matrix @ vec
        
        # Step 2: Scale
        scaled = self.key.scale_factor * rotated
        
        # Step 3: Add Gaussian noise
        noise = np.random.normal(0, self.key.noise_std, self.key.dimension)
        encrypted = scaled + noise
        
        return encrypted
    
    def encrypt_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encrypt multiple vectors efficiently.
        
        Args:
            vectors: Array of shape (n_vectors, dimension).
        
        Returns:
            Encrypted vectors of same shape.
        """
        if vectors.shape[1] != self.key.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match key dimension {self.key.dimension}"
            )
        
        # Vectorized operations for efficiency
        vecs = vectors.astype(np.float64)
        
        # Rotate all vectors: (n x d) @ (d x d)^T = (n x d)
        rotated = vecs @ self.key.rotation_matrix.T
        
        # Scale
        scaled = self.key.scale_factor * rotated
        
        # Add noise to each vector
        noise = np.random.normal(0, self.key.noise_std, vectors.shape)
        encrypted = scaled + noise
        
        return encrypted
    
    def insert(
        self,
        encrypted_vector: np.ndarray,
        identifier: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> int:
        """
        Insert an encrypted vector into the index.
        
        Args:
            encrypted_vector: Pre-encrypted vector (use encrypt_vector()).
            identifier: Optional identifier for the vector.
            metadata: Optional metadata (stored in plaintext!).
        
        Returns:
            Integer ID of the inserted vector.
        
        Example:
            >>> encrypted = enc_index.encrypt_vector(vector)
            >>> vid = enc_index.insert(encrypted, identifier="doc_1")
        """
        return self._index.insert(encrypted_vector, identifier, metadata)
    
    def insert_batch(
        self,
        encrypted_vectors: np.ndarray,
        identifiers: Optional[List[str]] = None,
        metadata: Optional[List[dict]] = None
    ) -> List[int]:
        """
        Insert multiple encrypted vectors at once.
        
        Args:
            encrypted_vectors: Pre-encrypted vectors (use encrypt_batch()).
            identifiers: Optional list of identifiers.
            metadata: Optional list of metadata dicts.
        
        Returns:
            List of integer IDs.
        """
        return self._index.insert_batch(encrypted_vectors, identifiers, metadata)
    
    def search(
        self,
        encrypted_query: np.ndarray,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for k most similar encrypted vectors.
        
        Args:
            encrypted_query: Pre-encrypted query vector (use encrypt_vector()).
            k: Number of results to return.
        
        Returns:
            List of (identifier, score) tuples.
            
        Note:
            Scores are computed on encrypted vectors, so they differ from
            plaintext scores by approximately scale_factor^2. The ranking
            is preserved (within noise tolerance).
        
        Example:
            >>> encrypted_query = enc_index.encrypt_vector(query)
            >>> results = enc_index.search(encrypted_query, k=5)
        """
        return self._index.search(encrypted_query, k)
    
    def get_vector_info(self, identifier: str) -> Optional[IndexedVector]:
        """Get metadata for a vector by identifier."""
        return self._index.get_vector_info(identifier)
    
    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._index.size
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        return (
            f"EncryptedANNSearch(dimension={self.key.dimension}, "
            f"scale={self.key.scale_factor}, "
            f"noise_std={self.key.noise_std}, "
            f"size={self.size})"
        )


def generate_sap_key(
    dimension: int,
    scale_factor: float = 10.0,
    noise_std: float = 0.1,
    seed: Optional[int] = None
) -> SAPEncryptionKey:
    """
    Generate a Scale-and-Perturb encryption key.
    
    Args:
        dimension: Dimensionality of vectors to encrypt.
        scale_factor: Scaling multiplier (larger = stronger obfuscation).
                     Typical values: 5.0 - 20.0
        noise_std: Standard deviation of Gaussian noise.
                   Larger noise = more privacy but less accuracy.
                   Typical values: 0.01 - 0.5
        seed: Random seed for reproducibility (optional).
    
    Returns:
        SAPEncryptionKey that can be used for encryption/decryption.
    
    Security notes:
        - scale_factor affects the magnitude of encrypted vectors
        - noise_std controls the privacy-accuracy tradeoff
        - The rotation matrix should be kept secret
        - For high security, use noise_std ≥ 0.1 and scale_factor ≥ 10
    
    Example:
        >>> key = generate_sap_key(dimension=128, scale_factor=15.0, noise_std=0.2)
        >>> key.save("my_encryption_key.pkl")
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random orthogonal matrix using QR decomposition
    # This preserves distances and angles exactly
    random_matrix = np.random.randn(dimension, dimension)
    rotation_matrix, _ = np.linalg.qr(random_matrix)
    
    # Verify it's orthogonal (should be I)
    verification = rotation_matrix.T @ rotation_matrix
    if not np.allclose(verification, np.eye(dimension), atol=1e-10):
        raise RuntimeError("Failed to generate orthogonal rotation matrix")
    
    return SAPEncryptionKey(
        rotation_matrix=rotation_matrix,
        scale_factor=scale_factor,
        noise_std=noise_std,
        dimension=dimension
    )


def compute_distance_preservation_error(
    vectors: np.ndarray,
    encrypted_vectors: np.ndarray,
    key: SAPEncryptionKey,
    sample_size: int = 100
) -> dict:
    """
    Analyze how well the encryption preserves distances.
    
    Args:
        vectors: Original plaintext vectors (n x d).
        encrypted_vectors: Encrypted versions (n x d).
        key: Encryption key used.
        sample_size: Number of random pairs to sample.
    
    Returns:
        Dictionary with error statistics:
        - mean_relative_error: Average relative error in distances
        - max_relative_error: Maximum relative error observed
        - correlation: Correlation between original and encrypted distances
    
    Example:
        >>> stats = compute_distance_preservation_error(plain, encrypted, key)
        >>> print(f"Mean error: {stats['mean_relative_error']:.2%}")
    """
    n = len(vectors)
    if n < 2:
        raise ValueError("Need at least 2 vectors to compute distances")
    
    sample_size = min(sample_size, n * (n - 1) // 2)
    
    # Sample random pairs
    pairs = []
    for _ in range(sample_size):
        i, j = np.random.choice(n, size=2, replace=False)
        pairs.append((i, j))
    
    # Compute original and encrypted inner products
    original_products = []
    encrypted_products = []
    
    for i, j in pairs:
        # Original inner product
        orig_ip = np.dot(vectors[i], vectors[j])
        original_products.append(orig_ip)
        
        # Encrypted inner product
        enc_ip = np.dot(encrypted_vectors[i], encrypted_vectors[j])
        encrypted_products.append(enc_ip)
    
    original_products = np.array(original_products)
    encrypted_products = np.array(encrypted_products)
    
    # Expected scaling: encrypted ≈ scale^2 * original (plus noise effects)
    expected_scale = key.scale_factor ** 2
    scaled_encrypted = encrypted_products / expected_scale
    
    # Compute errors
    relative_errors = np.abs(scaled_encrypted - original_products) / (np.abs(original_products) + 1e-10)
    
    # Compute correlation
    correlation = np.corrcoef(original_products, encrypted_products)[0, 1]
    
    return {
        'mean_relative_error': float(np.mean(relative_errors)),
        'max_relative_error': float(np.max(relative_errors)),
        'std_relative_error': float(np.std(relative_errors)),
        'correlation': float(correlation),
        'expected_scale': expected_scale,
    }
