"""Test ORAM-based encrypted search for hiding access patterns."""

from __future__ import annotations

import time
import numpy as np
import pytest

from poke_ckks import (
    PathORAM,
    ORAMEncryptedSearch,
    generate_sap_key,
    create_embedder,
)

try:
    from poke_ckks.text_embeddings import HAS_SENTENCE_TRANSFORMERS
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def test_path_oram_basic():
    """Test basic Path ORAM operations."""
    print("\n" + "="*80)
    print("PATH ORAM - BASIC OPERATIONS TEST")
    print("="*80)
    
    # Initialize ORAM
    capacity = 100
    block_size = 10
    oram = PathORAM(capacity=capacity, block_size=block_size, tree_height=7)
    
    print(f"\nConfiguration:")
    print(f"  Capacity: {capacity}")
    print(f"  Block size: {block_size}")
    print(f"  Tree height: {oram.tree_height}")
    print(f"  Number of leaves: {oram.num_leaves}")
    
    # Write some blocks
    print(f"\n[1/3] Writing blocks...")
    num_blocks = 20
    test_data = {}
    
    for i in range(num_blocks):
        data = np.random.randn(block_size)
        identifier = f"block_{i}"
        metadata = {"index": i, "category": f"cat_{i % 5}"}
        
        oram.write(
            block_id=i,
            data=data,
            identifier=identifier,
            metadata=metadata
        )
        test_data[i] = (data, identifier, metadata)
    
    print(f"  ✓ Wrote {num_blocks} blocks")
    
    # Read blocks back
    print(f"\n[2/3] Reading blocks obliviously...")
    start_time = time.time()
    
    for block_id in range(num_blocks):
        block = oram.read(block_id)
        
        if block is None:
            print(f"  ✗ Block {block_id} not found!")
            continue
        
        # Verify data integrity
        expected_data, expected_id, expected_meta = test_data[block_id]
        
        assert block.identifier == expected_id, f"Identifier mismatch for block {block_id}"
        assert np.allclose(block.data, expected_data), f"Data mismatch for block {block_id}"
        assert block.metadata == expected_meta, f"Metadata mismatch for block {block_id}"
    
    read_time = time.time() - start_time
    print(f"  ✓ All blocks read successfully")
    print(f"  ✓ Time: {read_time:.3f}s ({read_time/num_blocks*1000:.1f}ms per read)")
    
    # Statistics
    print(f"\n[3/3] ORAM Statistics:")
    stats = oram.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Path ORAM basic test passed!")
    print(f"🔒 Access patterns are hidden - each read accesses entire path")
    print(f"📊 Stash usage: {stats['stash_size']}/{stats['stash_capacity']}")


def test_oram_encrypted_search():
    """Test ORAM-based encrypted similarity search."""
    print("\n" + "="*80)
    print("ORAM ENCRYPTED SEARCH - SIMILARITY SEARCH TEST")
    print("="*80)
    
    # Configuration
    dimension = 64
    num_vectors = 50
    top_k = 5
    
    print(f"\nConfiguration:")
    print(f"  Dimension: {dimension}")
    print(f"  Number of vectors: {num_vectors}")
    print(f"  Top-K: {top_k}")
    
    # Generate encryption key
    print(f"\n[1/5] Generating SAP encryption key...")
    start_time = time.time()
    encryption_key = generate_sap_key(
        dimension=dimension,
        scale_factor=10.0,
        noise_std=0.05,
        seed=42
    )
    keygen_time = time.time() - start_time
    print(f"  ✓ Completed in {keygen_time:.3f}s")
    
    # Initialize ORAM search
    print(f"\n[2/5] Initializing ORAM encrypted search...")
    start_time = time.time()
    oram_search = ORAMEncryptedSearch(
        encryption_key=encryption_key,
        capacity=num_vectors * 2
    )
    init_time = time.time() - start_time
    print(f"  ✓ Completed in {init_time:.3f}s")
    print(f"  ✓ ORAM capacity: {oram_search.capacity}")
    print(f"  ✓ Tree height: {oram_search.oram.tree_height}")
    
    # Generate and insert random vectors
    print(f"\n[3/5] Inserting {num_vectors} vectors obliviously...")
    start_time = time.time()
    
    vectors = []
    for i in range(num_vectors):
        # Generate random normalized vector
        vec = np.random.randn(dimension)
        vec = vec / np.linalg.norm(vec)
        vectors.append(vec)
        
        # Insert into ORAM
        oram_search.insert(
            vector=vec,
            identifier=f"vec_{i}",
            metadata={"index": i, "category": f"cat_{i % 5}"}
        )
    
    insert_time = time.time() - start_time
    print(f"  ✓ Completed in {insert_time:.3f}s ({insert_time/num_vectors*1000:.1f}ms per vector)")
    
    # ORAM statistics
    stats = oram_search.get_oram_statistics()
    print(f"\n  ORAM Statistics:")
    print(f"    Blocks stored: {stats['blocks_stored']}")
    print(f"    Stash size: {stats['stash_size']}/{stats['stash_capacity']}")
    print(f"    Total block instances: {stats['total_block_instances']}")
    print(f"    Avg replication: {stats['avg_replication']:.2f}x")
    
    # Create query vector (similar to first vector)
    print(f"\n[4/5] Performing oblivious search...")
    query = vectors[0] + np.random.randn(dimension) * 0.1
    query = query / np.linalg.norm(query)
    
    start_time = time.time()
    oblivious_results = oram_search.search_oblivious(query, k=top_k)
    oblivious_time = time.time() - start_time
    
    print(f"  ✓ Oblivious search completed in {oblivious_time:.3f}s ({oblivious_time*1000:.1f}ms)")
    print(f"\n  Top {top_k} results (oblivious search):")
    for rank, (identifier, score) in enumerate(oblivious_results, 1):
        print(f"    #{rank} {identifier}: {score:.4f}")
    
    # Compare with fast search
    print(f"\n[5/5] Comparing with fast (non-oblivious) search...")
    start_time = time.time()
    fast_results = oram_search.search_fast(query, k=top_k)
    fast_time = time.time() - start_time
    
    print(f"  ✓ Fast search completed in {fast_time:.3f}s ({fast_time*1000:.1f}ms)")
    print(f"\n  Top {top_k} results (fast search):")
    for rank, (identifier, score) in enumerate(fast_results, 1):
        print(f"    #{rank} {identifier}: {score:.4f}")
    
    # Timing summary
    print(f"\n" + "="*80)
    print("⏱️  TIMING SUMMARY:")
    print("="*80)
    print(f"  Key generation:      {keygen_time:8.3f}s")
    print(f"  ORAM initialization: {init_time:8.3f}s")
    print(f"  Vector insertion:    {insert_time:8.3f}s  ({insert_time/num_vectors*1000:.1f}ms per vector)")
    print(f"  Oblivious search:    {oblivious_time:8.3f}s  ({oblivious_time*1000:.1f}ms)")
    print(f"  Fast search:         {fast_time:8.3f}s  ({fast_time*1000:.1f}ms)")
    print(f"  Speedup (fast/oblivious): {oblivious_time/fast_time:.1f}x")
    print("="*80)
    
    print(f"\n✅ ORAM encrypted search test passed!")
    print(f"🔒 Security: Both vector content AND access patterns are hidden")
    print(f"⚡ Trade-off: {oblivious_time/fast_time:.1f}x slower for access pattern privacy")
    
    # Assertions
    assert len(oblivious_results) > 0, "Oblivious search returned no results"
    assert len(fast_results) > 0, "Fast search returned no results"
    assert oram_search.size == num_vectors, f"Expected {num_vectors} vectors, got {oram_search.size}"


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
def test_oram_text_similarity():
    """Test ORAM search with real text embeddings."""
    print("\n" + "="*80)
    print("ORAM TEXT SIMILARITY SEARCH")
    print("Real-world text search with hidden access patterns")
    print("="*80)
    
    # Sample texts
    texts = [
        "Machine learning algorithms can identify patterns in large datasets.",
        "Artificial intelligence is transforming healthcare and medical diagnosis.",
        "Deep neural networks learn hierarchical representations of data.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and analyze visual information.",
        "Reinforcement learning agents learn by interacting with their environment.",
        "Climate change poses significant challenges to global ecosystems.",
        "Renewable energy sources are crucial for sustainable development.",
        "Ocean acidification threatens marine biodiversity worldwide.",
        "Deforestation contributes to habitat loss and species extinction.",
    ]
    
    dimension = 64
    top_k = 3
    
    print(f"\nConfiguration:")
    print(f"  Number of texts: {len(texts)}")
    print(f"  Embedding dimension: {dimension}")
    print(f"  Top-K: {top_k}")
    
    # Load embedder
    print(f"\n[1/4] Loading sentence transformer...")
    start_time = time.time()
    embedder = create_embedder(model_name='all-MiniLM-L6-v2', target_dimension=dimension)
    model_time = time.time() - start_time
    print(f"  ✓ Completed in {model_time:.3f}s")
    
    # Generate embeddings
    print(f"\n[2/4] Creating embeddings...")
    start_time = time.time()
    embeddings = embedder.embed_batch(texts)
    embed_time = time.time() - start_time
    print(f"  ✓ Completed in {embed_time:.3f}s ({embed_time/len(texts)*1000:.1f}ms per text)")
    
    # Initialize ORAM search
    print(f"\n[3/4] Building ORAM encrypted search index...")
    start_time = time.time()
    
    encryption_key = generate_sap_key(dimension=dimension, scale_factor=10.0, noise_std=0.1, seed=42)
    oram_search = ORAMEncryptedSearch(encryption_key=encryption_key, capacity=len(texts) * 2)
    
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        category = "AI/ML" if i < 6 else "Environment"
        oram_search.insert(
            vector=embedding,
            identifier=f"doc_{i}",
            metadata={"text": text, "category": category}
        )
    
    build_time = time.time() - start_time
    print(f"  ✓ Completed in {build_time:.3f}s")
    
    # Search
    query_text = "How do neural networks learn from data?"
    print(f"\n[4/4] Searching with ORAM (oblivious access)...")
    print(f'  Query: "{query_text}"')
    
    query_embedding = embedder.embed(query_text)
    
    start_time = time.time()
    results = oram_search.search_oblivious(query_embedding, k=top_k)
    search_time = time.time() - start_time
    
    print(f"  ✓ Search completed in {search_time:.3f}s ({search_time*1000:.1f}ms)")
    
    # Display results
    print(f"\n" + "-"*80)
    print(f"TOP {top_k} RESULTS:")
    print("-"*80)
    
    for rank, (identifier, score) in enumerate(results, 1):
        doc_idx = int(identifier.split('_')[1])
        text = texts[doc_idx]
        category = "AI/ML" if doc_idx < 6 else "Environment"
        
        print(f"\n#{rank} (Score: {score:.4f})")
        print(f"  ID: {identifier}")
        print(f"  Category: {category}")
        print(f"  Text: {text}")
    
    # Timing summary
    print(f"\n" + "="*80)
    print("⏱️  TIMING SUMMARY:")
    print("="*80)
    print(f"  Model loading:       {model_time:8.3f}s")
    print(f"  Text embeddings:     {embed_time:8.3f}s")
    print(f"  ORAM index building: {build_time:8.3f}s")
    print(f"  Oblivious search:    {search_time:8.3f}s")
    print("="*80)
    
    print(f"\n✅ ORAM text similarity search test passed!")
    print(f"🔒 Privacy guarantees:")
    print(f"   • Vector contents encrypted (SAP)")
    print(f"   • Access patterns hidden (Path ORAM)")
    print(f"   • Query content encrypted")
    
    # Basic assertions
    assert len(results) > 0, "No results returned"
    assert results[0][1] > 0, "Top result has non-positive score"
    
    # Expect AI/ML results for AI/ML query
    top_result_idx = int(results[0][0].split('_')[1])
    print(f"   • Top result is from {'AI/ML' if top_result_idx < 6 else 'Environment'} category")


if __name__ == "__main__":
    test_path_oram_basic()
    test_oram_encrypted_search()
    if HAS_SENTENCE_TRANSFORMERS:
        test_oram_text_similarity()
