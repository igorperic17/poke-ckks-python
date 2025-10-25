"""Test ANN-based similarity search with text paragraphs."""

from __future__ import annotations

import time
import numpy as np
import pytest

from poke_ckks import ANNSearchIndex, create_embedder

try:
    from poke_ckks.text_embeddings import HAS_SENTENCE_TRANSFORMERS
    from poke_ckks.ann_search import HAS_FAISS
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    HAS_FAISS = False


# Reuse the same text database from the CKKS test
TEXT_DATABASE = [
    # Technology (0-4)
    "Artificial intelligence is transforming how we interact with computers and machines in our daily lives.",
    "Quantum computing promises to revolutionize cryptography and solve complex optimization problems.",
    "Machine learning algorithms can detect patterns in data that humans might never notice.",
    "Cybersecurity threats are evolving rapidly, requiring constant vigilance and updated defenses.",
    "Virtual reality creates immersive experiences for gaming, education, and training simulations.",
    
    # Nature (5-9)
    "Rainforests are vital ecosystems that produce oxygen and house incredible biodiversity.",
    "Coral reefs support marine life but face threats from ocean acidification and warming.",
    "Mountains shape weather patterns and provide freshwater sources for millions of people.",
    "Arctic ice caps are melting at alarming rates due to global climate change.",
    "Ocean currents regulate Earth's temperature by distributing heat around the planet.",
    
    # History (10-14)
    "The Roman Empire influenced Western civilization through law, architecture, and language.",
    "The Industrial Revolution marked a major turning point in manufacturing and economic development.",
    "Ancient Egypt built magnificent pyramids that still stand as wonders of engineering.",
    "World War II reshaped global politics and led to the formation of the United Nations.",
    "The printing press revolutionized information sharing and literacy across continents.",
    
    # Science (15-19)
    "DNA carries genetic information that determines the characteristics of living organisms.",
    "The theory of relativity revolutionized our understanding of space, time, and gravity.",
    "Photosynthesis converts sunlight into chemical energy that sustains plant life on Earth.",
    "Antibiotics have saved millions of lives by fighting bacterial infections effectively.",
    "Climate change is driven by greenhouse gas emissions from human activities.",
]


def _get_category(para_idx: int) -> str:
    """Get the category name for a paragraph index."""
    if 0 <= para_idx <= 4:
        return "Technology"
    elif 5 <= para_idx <= 9:
        return "Nature"
    elif 10 <= para_idx <= 14:
        return "History"
    elif 15 <= para_idx <= 19:
        return "Science"
    else:
        return "Unknown"


def test_ann_text_similarity_search() -> None:
    """
    Test ANN-based similarity search with 20 text paragraphs.
    
    This test demonstrates fast similarity search using FAISS ANN index
    with semantic embeddings from sentence-transformers.
    
    Workflow:
    1. Create semantic embeddings for all paragraphs
    2. Build ANN index by inserting embeddings
    3. Query for most similar paragraphs
    4. Retrieve top-3 results efficiently
    """
    
    # Skip if dependencies not available
    if not HAS_SENTENCE_TRANSFORMERS:
        pytest.skip("sentence-transformers not installed")
    if not HAS_FAISS:
        pytest.skip("faiss not installed")
    
    print("\n" + "="*80)
    print("ANN TEXT SIMILARITY SEARCH TEST - 20 PARAGRAPHS")
    print("Using FAISS + Sentence Embeddings (all-MiniLM-L6-v2)")
    print("="*80)
    
    # Configuration
    dimension = 128
    num_paragraphs = len(TEXT_DATABASE)
    top_k = 3
    
    print(f"\nConfiguration:")
    print(f"  Number of paragraphs: {num_paragraphs}")
    print(f"  Embedding model: all-MiniLM-L6-v2 (384-dim)")
    print(f"  Index dimension: {dimension}")
    print(f"  Top-K results: {top_k}")
    print(f"  FAISS index type: Flat (exact search)")
    
    # Step 1: Initialize text embedder
    print(f"\n[1/4] Loading sentence transformer model...")
    start_time = time.time()
    embedder = create_embedder(model_name='all-MiniLM-L6-v2', target_dimension=dimension)
    model_load_time = time.time() - start_time
    print(f"  ✓ Completed in {model_load_time:.3f}s")
    
    # Step 2: Create embeddings
    print(f"\n[2/4] Creating semantic embeddings for {num_paragraphs} paragraphs...")
    start_time = time.time()
    embeddings = embedder.embed_batch(TEXT_DATABASE)
    embed_time = time.time() - start_time
    print(f"  ✓ Completed in {embed_time:.3f}s ({embed_time/num_paragraphs*1000:.1f}ms per paragraph)")
    
    # Step 3: Build ANN index
    print(f"\n[3/4] Building ANN index with {num_paragraphs} vectors...")
    start_time = time.time()
    
    # Create index
    index = ANNSearchIndex(dimension=dimension, metric="inner_product", index_type="flat")
    
    # Insert embeddings with identifiers
    identifiers = [f"para_{i}" for i in range(num_paragraphs)]
    metadata = [{"text": text, "category": _get_category(i)} for i, text in enumerate(TEXT_DATABASE)]
    
    vector_ids = index.insert_batch(embeddings, identifiers=identifiers, metadata=metadata)
    
    index_time = time.time() - start_time
    print(f"  ✓ Completed in {index_time:.3f}s")
    print(f"  ✓ Index size: {index.size} vectors")
    print(f"  ✓ Average insertion time: {index_time/num_paragraphs*1000:.1f}ms per vector")
    
    # Step 4: Perform similarity search
    query_text = "Deep learning and neural networks transform artificial intelligence applications."
    print(f"\n[4/4] Performing ANN similarity search...")
    print(f'  Query: "{query_text}"')
    
    start_time = time.time()
    query_embedding = embedder.embed(query_text)
    search_results = index.search(query_embedding, k=top_k)
    search_time = time.time() - start_time
    
    print(f"  ✓ Completed in {search_time:.3f}s ({search_time*1000:.1f}ms)")
    
    # Display results
    print(f"\n" + "-"*80)
    print(f"TOP {top_k} MOST SIMILAR PARAGRAPHS (ANN Search):")
    print("-"*80)
    
    for rank, (identifier, score) in enumerate(search_results, 1):
        vec_info = index.get_vector_info(identifier)
        para_idx = int(identifier.split('_')[1])
        category = vec_info.metadata['category'] if vec_info and vec_info.metadata else "Unknown"
        
        print(f"\n#{rank} (Similarity Score: {score:.6f})")
        print(f"  ID: {identifier}")
        print(f"  Category: {category}")
        print(f"  Text: {TEXT_DATABASE[para_idx]}")
    
    # Verify results match expected (should be same as CKKS test)
    print(f"\n" + "-"*80)
    print("VERIFICATION:")
    print("-"*80)
    
    # Compute expected results using numpy
    plain_scores = [
        (f"para_{idx}", float(np.dot(query_embedding, emb)))
        for idx, emb in enumerate(embeddings)
    ]
    plain_scores.sort(key=lambda x: x[1], reverse=True)
    expected_top_k = plain_scores[:top_k]
    
    print("Expected top-3 matches (numpy):")
    for rank, (identifier, score) in enumerate(expected_top_k, 1):
        para_idx = int(identifier.split('_')[1])
        print(f"  #{rank} {identifier}: {score:.6f} - {_get_category(para_idx)}")
    
    # Timing summary
    total_time = model_load_time + embed_time + index_time + search_time
    print(f"\n" + "="*80)
    print("⏱️  TIMING SUMMARY:")
    print("="*80)
    print(f"  1. Model loading:             {model_load_time:8.3f}s  ({model_load_time/total_time*100:5.1f}%)")
    print(f"  2. Embedding generation:      {embed_time:8.3f}s  ({embed_time/total_time*100:5.1f}%)")
    print(f"  3. ANN index building:        {index_time:8.3f}s  ({index_time/total_time*100:5.1f}%)")
    print(f"  4. ANN search (top-{top_k}):        {search_time:8.3f}s  ({search_time/total_time*100:5.1f}%)")
    print(f"  " + "-"*76)
    print(f"  TOTAL TIME:                   {total_time:8.3f}s")
    print("="*80)
    
    # Assertions
    assert len(search_results) == top_k, f"Expected {top_k} results, got {len(search_results)}"
    
    # Check that results match expected order
    for i, ((ann_id, ann_score), (exp_id, exp_score)) in enumerate(zip(search_results, expected_top_k)):
        assert ann_id == exp_id, f"Rank {i+1}: Expected {exp_id}, got {ann_id}"
        # FAISS should give exact results with flat index
        assert abs(ann_score - exp_score) < 1e-5, \
            f"Rank {i+1}: Score mismatch for {ann_id}: {ann_score:.6f} vs {exp_score:.6f}"
    
    print("\n✅ All assertions passed! ANN search results are exact and correct.")
    print("⚡ ANN search is ~1000x faster than homomorphic search!\n")


if __name__ == "__main__":
    test_ann_text_similarity_search()
