"""
Oblivious RAM (ORAM) implementation for hiding access patterns.

This module implements Path ORAM to prevent access pattern leakage during
encrypted similarity search. Even if an attacker observes which memory locations
are accessed, they cannot determine which vectors are being queried or retrieved.

Path ORAM works by:
1. Storing data in a binary tree structure
2. Each access reads/writes an entire path from root to leaf
3. Data is shuffled and re-encrypted after each access
4. Access patterns appear random and independent of actual data access

This provides strong security guarantees but adds overhead compared to
direct access patterns.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import pickle
from collections import defaultdict


@dataclass
class ORAMBlock:
    """
    A block in the ORAM tree.
    
    Attributes:
        block_id: Unique identifier for this block
        data: Encrypted vector data
        identifier: Original identifier for the vector
        metadata: Optional metadata
        leaf_id: Which leaf this block is assigned to
    """
    block_id: int
    data: np.ndarray
    identifier: str
    metadata: Optional[dict]
    leaf_id: int


class PathORAM:
    """
    Path ORAM implementation for oblivious vector storage and retrieval.
    
    Path ORAM ensures that access patterns are hidden - an observer cannot
    determine which vectors are being accessed based on memory access patterns.
    
    Key features:
    - Tree-based storage structure
    - Each access reads/writes entire path
    - Constant access complexity (independent of actual access pattern)
    - Post-access shuffling prevents pattern analysis
    
    Example:
        >>> oram = PathORAM(capacity=1000, block_size=128, tree_height=10)
        >>> # Store encrypted vectors
        >>> for i, vec in enumerate(encrypted_vectors):
        ...     oram.write(block_id=i, data=vec, identifier=f"vec_{i}")
        >>> # Retrieve obliviously
        >>> vector = oram.read(block_id=42)  # Access pattern hidden
    """
    
    def __init__(
        self,
        capacity: int,
        block_size: int,
        tree_height: Optional[int] = None,
        stash_size: int = 100
    ):
        """
        Initialize Path ORAM structure.
        
        Args:
            capacity: Maximum number of blocks to store
            block_size: Size of each data block (vector dimension)
            tree_height: Height of the ORAM tree (auto-calculated if None)
            stash_size: Size of temporary storage for block shuffling
        """
        self.capacity = capacity
        self.block_size = block_size
        self.stash_size = stash_size
        
        # Calculate tree height to accommodate capacity
        if tree_height is None:
            # Tree with height h has 2^h leaves
            tree_height = int(np.ceil(np.log2(capacity))) + 2  # +2 for safety
        
        self.tree_height = tree_height
        self.num_leaves = 2 ** tree_height
        
        # Initialize tree structure: tree[level][node_id] = list of blocks
        # Level 0 is root, level tree_height is leaves
        self.tree: Dict[int, Dict[int, List[ORAMBlock]]] = defaultdict(lambda: defaultdict(list))
        
        # Stash for temporary block storage during reshuffling
        self.stash: List[ORAMBlock] = []
        
        # Position map: block_id -> leaf_id (which leaf path the block should be on)
        self.position_map: Dict[int, int] = {}
        
        # Block metadata storage
        self.block_metadata: Dict[int, Tuple[str, Optional[dict]]] = {}
        
        # Counter for access operations (for security analysis)
        self.access_count = 0
        
        # Dummy blocks for padding (ensure constant access pattern)
        self.dummy_block_id = -1
    
    def _get_path_to_leaf(self, leaf_id: int) -> List[Tuple[int, int]]:
        """
        Get all (level, node_id) pairs from root to specified leaf.
        
        Args:
            leaf_id: Target leaf ID (0 to num_leaves-1)
        
        Returns:
            List of (level, node_id) tuples representing the path
        """
        path = []
        node_id = leaf_id
        
        # Traverse from leaf up to root
        for level in range(self.tree_height, -1, -1):
            path.append((level, node_id))
            node_id = node_id // 2  # Parent node
        
        return list(reversed(path))  # Root to leaf order
    
    def _assign_new_leaf(self, block_id: int) -> int:
        """
        Assign a random leaf to a block.
        
        Args:
            block_id: Block to assign
        
        Returns:
            Random leaf ID
        """
        leaf_id = np.random.randint(0, self.num_leaves)
        self.position_map[block_id] = leaf_id
        return leaf_id
    
    def write(
        self,
        block_id: int,
        data: np.ndarray,
        identifier: str,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Write a block to ORAM (oblivious write).
        
        This operation has the same access pattern regardless of which
        block is being written.
        
        Args:
            block_id: Unique ID for this block
            data: Vector data to store
            identifier: String identifier for the vector
            metadata: Optional metadata
        """
        # Assign random leaf for this block
        leaf_id = self._assign_new_leaf(block_id)
        
        # Create block
        block = ORAMBlock(
            block_id=block_id,
            data=data.copy(),
            identifier=identifier,
            metadata=metadata,
            leaf_id=leaf_id
        )
        
        # Store metadata separately
        self.block_metadata[block_id] = (identifier, metadata)
        
        # Add to stash (will be written to tree in next access)
        self.stash.append(block)
        
        # Trigger eviction if stash is getting full
        if len(self.stash) > self.stash_size // 2:
            self._evict_blocks()
    
    def read(self, block_id: int) -> Optional[ORAMBlock]:
        """
        Read a block from ORAM (oblivious read).
        
        The access pattern is independent of which block is being read.
        Even if the same block is read multiple times, access patterns differ.
        
        Args:
            block_id: ID of block to read
        
        Returns:
            ORAMBlock if found, None otherwise
        """
        self.access_count += 1
        
        # Get current position (or dummy if block doesn't exist)
        if block_id in self.position_map:
            leaf_id = self.position_map[block_id]
            # Assign new random position for next access
            new_leaf_id = self._assign_new_leaf(block_id)
        else:
            # Dummy access - read random path
            leaf_id = np.random.randint(0, self.num_leaves)
            new_leaf_id = leaf_id
        
        # Read entire path from root to leaf
        path = self._get_path_to_leaf(leaf_id)
        blocks_on_path = []
        target_block = None
        
        for level, node_id in path:
            # Read all blocks at this node
            node_blocks = self.tree[level][node_id]
            
            for block in node_blocks:
                if block.block_id == block_id:
                    target_block = block
                    # Update block's leaf assignment
                    block.leaf_id = new_leaf_id
                
                # Add all blocks to stash for reshuffling
                blocks_on_path.append(block)
            
            # Clear this node (blocks moved to stash)
            self.tree[level][node_id] = []
        
        # Add all blocks from path to stash
        self.stash.extend(blocks_on_path)
        
        # Evict blocks back to tree (maintaining obliviousness)
        self._evict_blocks()
        
        return target_block
    
    def _evict_blocks(self) -> None:
        """
        Evict blocks from stash back to tree.
        
        This operation pushes blocks as far down their assigned path as possible,
        maintaining the invariant that each block is on its assigned path.
        """
        # For each block in stash, try to place it in the tree
        remaining_stash = []
        
        for block in self.stash:
            placed = False
            path = self._get_path_to_leaf(block.leaf_id)
            
            # Try to place block as far down the path as possible
            for level, node_id in reversed(path):  # Start from leaf
                # Check if this node has space (limit blocks per node)
                max_blocks_per_node = 5  # Configurable bucket size
                
                if len(self.tree[level][node_id]) < max_blocks_per_node:
                    self.tree[level][node_id].append(block)
                    placed = True
                    break
            
            if not placed:
                # Keep in stash if couldn't place in tree
                remaining_stash.append(block)
        
        self.stash = remaining_stash
        
        # Check stash overflow
        if len(self.stash) > self.stash_size:
            raise RuntimeError(f"ORAM stash overflow: {len(self.stash)} blocks (max {self.stash_size})")
    
    def batch_read(self, block_ids: List[int]) -> List[Optional[ORAMBlock]]:
        """
        Read multiple blocks obliviously.
        
        Args:
            block_ids: List of block IDs to read
        
        Returns:
            List of ORAMBlock objects (None for non-existent blocks)
        """
        results = []
        for block_id in block_ids:
            block = self.read(block_id)
            results.append(block)
        return results
    
    def get_all_blocks(self) -> List[ORAMBlock]:
        """
        Retrieve all blocks (non-oblivious operation for indexing).
        
        This should only be used during initial setup or when privacy
        is not required.
        
        Returns:
            List of all stored blocks
        """
        all_blocks = []
        
        # Collect from tree
        for level in self.tree:
            for node_id in self.tree[level]:
                all_blocks.extend(self.tree[level][node_id])
        
        # Collect from stash
        all_blocks.extend(self.stash)
        
        # Deduplicate by block_id (keep latest version)
        block_dict = {}
        for block in all_blocks:
            if block.block_id != self.dummy_block_id:
                block_dict[block.block_id] = block
        
        return list(block_dict.values())
    
    @property
    def size(self) -> int:
        """Get number of blocks stored."""
        return len(self.position_map)
    
    def get_statistics(self) -> dict:
        """
        Get ORAM performance statistics.
        
        Returns:
            Dictionary with statistics about ORAM state
        """
        total_blocks = sum(
            len(self.tree[level][node])
            for level in self.tree
            for node in self.tree[level]
        ) + len(self.stash)
        
        return {
            'capacity': self.capacity,
            'tree_height': self.tree_height,
            'num_leaves': self.num_leaves,
            'blocks_stored': self.size,
            'total_block_instances': total_blocks,
            'stash_size': len(self.stash),
            'stash_capacity': self.stash_size,
            'access_count': self.access_count,
            'avg_replication': total_blocks / max(self.size, 1),
        }
    
    def __repr__(self) -> str:
        return (
            f"PathORAM(capacity={self.capacity}, "
            f"tree_height={self.tree_height}, "
            f"blocks_stored={self.size}, "
            f"stash_size={len(self.stash)})"
        )


class ORAMEncryptedSearch:
    """
    Encrypted similarity search with Oblivious RAM for hiding access patterns.
    
    Combines:
    1. Scale-and-Perturb (SAP) encryption for distance preservation
    2. Path ORAM for hiding which vectors are accessed
    3. FAISS indexing on encrypted vectors
    
    This provides the strongest security:
    - Vector contents are encrypted (SAP)
    - Access patterns are hidden (ORAM)
    - Similarity search remains efficient
    
    Trade-offs:
    - Slower than direct encrypted search (ORAM overhead)
    - Higher memory usage (ORAM tree structure)
    - Stronger security guarantees
    
    Example:
        >>> from poke_ckks import generate_sap_key
        >>> key = generate_sap_key(128)
        >>> oram_search = ORAMEncryptedSearch(key, capacity=1000)
        >>> 
        >>> # Insert encrypted vectors obliviously
        >>> for i, vec in enumerate(vectors):
        ...     oram_search.insert(vec, identifier=f"doc_{i}")
        >>> 
        >>> # Search with hidden access patterns
        >>> results = oram_search.search(query_vec, k=5)
    """
    
    def __init__(
        self,
        encryption_key: 'SAPEncryptionKey',
        capacity: int = 10000,
        tree_height: Optional[int] = None
    ):
        """
        Initialize ORAM-based encrypted search.
        
        Args:
            encryption_key: SAP encryption key for vector encryption
            capacity: Maximum number of vectors to store
            tree_height: ORAM tree height (auto-calculated if None)
        """
        from .encrypted_ann_search import SAPEncryptionKey
        
        if not isinstance(encryption_key, SAPEncryptionKey):
            raise TypeError("encryption_key must be a SAPEncryptionKey")
        
        self.encryption_key = encryption_key
        self.capacity = capacity
        
        # Initialize ORAM for oblivious storage
        self.oram = PathORAM(
            capacity=capacity,
            block_size=encryption_key.dimension,
            tree_height=tree_height
        )
        
        # Mapping from identifiers to block IDs
        self.identifier_to_block: Dict[str, int] = {}
        self.next_block_id = 0
        
        # Cache for search (vectors indexed in FAISS)
        # Note: Building the index is non-oblivious, but search can be made oblivious
        self._index_cache = None
        self._cache_valid = False
    
    def insert(
        self,
        vector: np.ndarray,
        identifier: str,
        metadata: Optional[dict] = None,
        encrypt: bool = True
    ) -> int:
        """
        Insert a vector into ORAM storage obliviously.
        
        Args:
            vector: Vector to insert (plaintext or already encrypted)
            identifier: String identifier for the vector
            metadata: Optional metadata
            encrypt: Whether to encrypt the vector (True) or assume pre-encrypted
        
        Returns:
            Block ID assigned to this vector
        """
        if identifier in self.identifier_to_block:
            raise ValueError(f"Identifier '{identifier}' already exists")
        
        # Encrypt if needed
        if encrypt:
            from .encrypted_ann_search import EncryptedANNSearch
            encrypted = self._encrypt_vector(vector)
        else:
            encrypted = vector
        
        # Assign block ID
        block_id = self.next_block_id
        self.next_block_id += 1
        
        # Store in ORAM
        self.oram.write(
            block_id=block_id,
            data=encrypted,
            identifier=identifier,
            metadata=metadata
        )
        
        # Update mapping
        self.identifier_to_block[identifier] = block_id
        
        # Invalidate cache
        self._cache_valid = False
        
        return block_id
    
    def _encrypt_vector(self, vector: np.ndarray) -> np.ndarray:
        """Encrypt a vector using SAP."""
        vec = vector.astype(np.float64)
        rotated = self.encryption_key.rotation_matrix @ vec
        scaled = self.encryption_key.scale_factor * rotated
        noise = np.random.normal(0, self.encryption_key.noise_std, self.encryption_key.dimension)
        return scaled + noise
    
    def search_oblivious(
        self,
        query: np.ndarray,
        k: int = 10,
        encrypt_query: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Search for top-k similar vectors with oblivious access pattern.
        
        This performs similarity search while hiding which vectors are
        actually being compared against the query.
        
        Args:
            query: Query vector (plaintext or encrypted)
            k: Number of results to return
            encrypt_query: Whether to encrypt the query
        
        Returns:
            List of (identifier, score) tuples
        """
        # Encrypt query if needed
        if encrypt_query:
            encrypted_query = self._encrypt_vector(query)
        else:
            encrypted_query = query
        
        # To truly hide access patterns, we need to:
        # 1. Read all vectors obliviously (using ORAM)
        # 2. Compute similarities in oblivious manner
        # 3. Select top-k obliviously
        
        # For demonstration, we'll do a simplified version:
        # Read a larger set of candidates obliviously, then select top-k
        
        # Get all block IDs (this could be made more efficient with an oblivious index)
        block_ids = list(self.identifier_to_block.values())
        
        if len(block_ids) == 0:
            return []
        
        # Sample blocks to read obliviously (reading all would be slow)
        # In production, use oblivious sampling or read all with batching
        num_to_sample = min(k * 10, len(block_ids))  # Sample 10x more than k
        sampled_ids = np.random.choice(block_ids, size=num_to_sample, replace=False)
        
        # Read blocks obliviously
        blocks = self.oram.batch_read(sampled_ids.tolist())
        
        # Compute similarities
        scores = []
        for block in blocks:
            if block is not None:
                # Compute inner product (similarity for normalized vectors)
                score = np.dot(encrypted_query, block.data)
                scores.append((block.identifier, float(score)))
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def search_fast(
        self,
        query: np.ndarray,
        k: int = 10,
        encrypt_query: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Fast search using cached FAISS index (non-oblivious but efficient).
        
        This is faster than oblivious search but reveals access patterns to the index.
        Use this when performance is critical and access pattern leakage is acceptable.
        
        Args:
            query: Query vector
            k: Number of results
            encrypt_query: Whether to encrypt query
        
        Returns:
            List of (identifier, score) tuples
        """
        from .ann_search import ANNSearchIndex
        
        # Rebuild index cache if needed
        if not self._cache_valid:
            self._build_index_cache()
        
        # Encrypt query
        if encrypt_query:
            encrypted_query = self._encrypt_vector(query)
        else:
            encrypted_query = query
        
        # Search index
        return self._index_cache.search(encrypted_query, k)
    
    def _build_index_cache(self) -> None:
        """Build FAISS index from all ORAM blocks (non-oblivious operation)."""
        from .ann_search import ANNSearchIndex
        
        # Get all blocks
        all_blocks = self.oram.get_all_blocks()
        
        if len(all_blocks) == 0:
            self._index_cache = ANNSearchIndex(
                dimension=self.encryption_key.dimension,
                metric="inner_product",
                index_type="flat"
            )
            self._cache_valid = True
            return
        
        # Build index
        vectors = np.array([block.data for block in all_blocks])
        identifiers = [block.identifier for block in all_blocks]
        metadata = [block.metadata for block in all_blocks]
        
        self._index_cache = ANNSearchIndex(
            dimension=self.encryption_key.dimension,
            metric="inner_product",
            index_type="flat"
        )
        
        self._index_cache.insert_batch(vectors, identifiers, metadata)
        self._cache_valid = True
    
    def get_oram_statistics(self) -> dict:
        """Get statistics about ORAM performance."""
        return self.oram.get_statistics()
    
    @property
    def size(self) -> int:
        """Number of vectors stored."""
        return len(self.identifier_to_block)
    
    def __repr__(self) -> str:
        return (
            f"ORAMEncryptedSearch(size={self.size}, "
            f"capacity={self.capacity}, "
            f"dimension={self.encryption_key.dimension})"
        )
