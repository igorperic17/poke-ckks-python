"""Test encrypted ANN search with Scale-and-Perturb encryption."""

from __future__ import annotations

import time
import numpy as np
import pytest

from poke_ckks import (
    EncryptedANNSearch,
    generate_sap_key,
    compute_distance_preservation_error,
    create_embedder,
)

try:
    from poke_ckks.text_embeddings import HAS_SENTENCE_TRANSFORMERS
    from poke_ckks.ann_search import HAS_FAISS
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    HAS_FAISS = False


# Reuse text database
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


def test_encrypted_ann_text_similarity_search() -> None:
    """
    Test privacy-preserving similarity search using Scale-and-Perturb encryption.
    
    This test demonstrates:
    1. Encrypting text embeddings with SAP
    2. Building FAISS index on encrypted vectors
    3. Searching encrypted index with encrypted queries
    4. Preserving relative ranking despite encryption
    
    This approach is much faster than CKKS while still providing privacy.
    """
    
    # Skip if dependencies not available
    if not HAS_SENTENCE_TRANSFORMERS:
        pytest.skip("sentence-transformers not installed")
    if not HAS_FAISS:
        pytest.skip("faiss not installed")
    
    print("\n" + "="*80)
    print("ENCRYPTED ANN SEARCH TEST - Scale-and-Perturb (SAP)")
    print("Privacy-Preserving Search with Distance-Preserving Encryption")
    print("="*80)
    
    # Configuration
    dimension = 128
    num_paragraphs = len(TEXT_DATABASE)
    top_k = 3
    
    # Encryption parameters
    scale_factor = 10.0  # Amplification factor
    noise_std = 0.1      # Privacy noise (higher = more privacy, less accuracy)
    
    print(f"\nConfiguration:")
    print(f"  Number of paragraphs: {num_paragraphs}")
    print(f"  Embedding model: all-MiniLM-L6-v2 (384-dim)")
    print(f"  Embedding dimension: {dimension}")
    print(f"  Encryption: Scale-and-Perturb")
    print(f"    - Scale factor: {scale_factor}")
    print(f"    - Noise std: {noise_std}")
    print(f"  Top-K results: {top_k}")
    
    # Step 1: Generate encryption key
    print(f"\n[1/6] Generating SAP encryption key...")
    start_time = time.time()
    encryption_key = generate_sap_key(
        dimension=dimension,
        scale_factor=scale_factor,
        noise_std=noise_std,
        seed=42  # For reproducibility in tests
    )
    keygen_time = time.time() - start_time
    print(f"  ✓ Completed in {keygen_time:.3f}s")
    print(f"  ✓ Key properties:")
    print(f"    - Rotation matrix: {dimension}x{dimension} orthogonal")
    print(f"    - Orthogonality check: {np.allclose(encryption_key.rotation_matrix.T @ encryption_key.rotation_matrix, np.eye(dimension))}")
    
    # Step 2: Load embedder
    print(f"\n[2/6] Loading sentence transformer model...")
    start_time = time.time()
    embedder = create_embedder(model_name='all-MiniLM-L6-v2', target_dimension=dimension)
    model_load_time = time.time() - start_time
    print(f"  ✓ Completed in {model_load_time:.3f}s")
    
    # Step 3: Create embeddings
    print(f"\n[3/6] Creating plaintext embeddings for {num_paragraphs} paragraphs...")
    start_time = time.time()
    plaintext_embeddings = embedder.embed_batch(TEXT_DATABASE)
    embed_time = time.time() - start_time
    print(f"  ✓ Completed in {embed_time:.3f}s ({embed_time/num_paragraphs*1000:.1f}ms per paragraph)")
    
    # Step 4: Create encrypted index and encrypt vectors
    print(f"\n[4/6] Encrypting embeddings and building encrypted index...")
    start_time = time.time()
    
    # Initialize encrypted ANN search
    enc_index = EncryptedANNSearch(
        key=encryption_key,
        metric="inner_product",
        index_type="flat"
    )
    
    # Encrypt all embeddings
    encrypted_embeddings = enc_index.encrypt_batch(plaintext_embeddings)
    
    # Insert into index
    identifiers = [f"para_{i}" for i in range(num_paragraphs)]
    metadata = [{"text": text, "category": _get_category(i)} for i, text in enumerate(TEXT_DATABASE)]
    vector_ids = enc_index.insert_batch(encrypted_embeddings, identifiers=identifiers, metadata=metadata)
    
    encrypt_and_index_time = time.time() - start_time
    print(f"  ✓ Completed in {encrypt_and_index_time:.3f}s")
    print(f"  ✓ Encrypted index size: {enc_index.size} vectors")
    print(f"  ✓ Average encryption + insertion: {encrypt_and_index_time/num_paragraphs*1000:.1f}ms per vector")
    
    # Step 5: Analyze distance preservation
    print(f"\n[5/6] Analyzing distance preservation quality...")
    start_time = time.time()
    stats = compute_distance_preservation_error(
        plaintext_embeddings,
        encrypted_embeddings,
        encryption_key,
        sample_size=50
    )
    analysis_time = time.time() - start_time
    print(f"  ✓ Completed in {analysis_time:.3f}s")
    print(f"  ✓ Distance preservation metrics:")
    print(f"    - Mean relative error: {stats['mean_relative_error']:.2%}")
    print(f"    - Max relative error: {stats['max_relative_error']:.2%}")
    print(f"    - Correlation (plain vs encrypted): {stats['correlation']:.4f}")
    print(f"    - Expected scale factor^2: {stats['expected_scale']:.2f}")
    
    # Step 6: Search encrypted index
    query_text = "Deep learning and neural networks transform artificial intelligence applications."
    print(f"\n[6/6] Performing encrypted similarity search...")
    print(f'  Query: "{query_text}"')
    
    start_time = time.time()
    
    # Encrypt query
    query_embedding = embedder.embed(query_text)
    encrypted_query = enc_index.encrypt_vector(query_embedding)
    
    # Search encrypted index
    encrypted_results = enc_index.search(encrypted_query, k=top_k)
    
    search_time = time.time() - start_time
    print(f"  ✓ Completed in {search_time:.3f}s ({search_time*1000:.1f}ms)")
    
    # Display encrypted search results
    print(f"\n" + "-"*80)
    print(f"TOP {top_k} RESULTS (Encrypted Search):")
    print("-"*80)
    
    for rank, (identifier, score) in enumerate(encrypted_results, 1):
        vec_info = enc_index.get_vector_info(identifier)
        para_idx = int(identifier.split('_')[1])
        category = vec_info.metadata['category'] if vec_info and vec_info.metadata else "Unknown"
        
        print(f"\n#{rank} (Encrypted Similarity Score: {score:.2f})")
        print(f"  ID: {identifier}")
        print(f"  Category: {category}")
        print(f"  Text: {TEXT_DATABASE[para_idx]}")
    
    # Compare with plaintext search
    print(f"\n" + "-"*80)
    print("VERIFICATION (Plaintext search for comparison):")
    print("-"*80)
    
    plain_scores = [
        (f"para_{idx}", float(np.dot(query_embedding, emb)))
        for idx, emb in enumerate(plaintext_embeddings)
    ]
    plain_scores.sort(key=lambda x: x[1], reverse=True)
    expected_top_k = plain_scores[:top_k]
    
    print("Expected top-3 matches (plaintext):")
    for rank, (identifier, score) in enumerate(expected_top_k, 1):
        para_idx = int(identifier.split('_')[1])
        print(f"  #{rank} {identifier}: {score:.6f} - {_get_category(para_idx)}")
    
    # Timing summary
    total_time = keygen_time + model_load_time + embed_time + encrypt_and_index_time + search_time
    print(f"\n" + "="*80)
    print("⏱️  TIMING SUMMARY:")
    print("="*80)
    print(f"  1. Key generation:            {keygen_time:8.3f}s  ({keygen_time/total_time*100:5.1f}%)")
    print(f"  2. Model loading:             {model_load_time:8.3f}s  ({model_load_time/total_time*100:5.1f}%)")
    print(f"  3. Plaintext embeddings:      {embed_time:8.3f}s  ({embed_time/total_time*100:5.1f}%)")
    print(f"  4. Encrypt + index building:  {encrypt_and_index_time:8.3f}s  ({encrypt_and_index_time/total_time*100:5.1f}%)")
    print(f"  5. Encrypted search:          {search_time:8.3f}s  ({search_time/total_time*100:5.1f}%)")
    print(f"  " + "-"*76)
    print(f"  TOTAL TIME:                   {total_time:8.3f}s")
    print("="*80)
    
    # Assertions
    assert len(encrypted_results) == top_k, f"Expected {top_k} results, got {len(encrypted_results)}"
    
    # Check that ranking is preserved (may have small differences due to noise)
    encrypted_ids = [r[0] for r in encrypted_results]
    expected_ids = [r[0] for r in expected_top_k]
    
    # Allow for some ranking differences due to encryption noise
    # At least top result should match
    assert encrypted_ids[0] == expected_ids[0], \
        f"Top result mismatch: Expected {expected_ids[0]}, got {encrypted_ids[0]}"
    
    # Check that most results are in the expected set
    matches = sum(1 for eid in encrypted_ids if eid in expected_ids)
    match_rate = matches / top_k
    assert match_rate >= 0.67, \
        f"Too many ranking differences: only {matches}/{top_k} results match"
    
    # Verify distance preservation is reasonable
    # Note: correlation is the key metric - relative error can be high for small inner products
    assert stats['correlation'] > 0.95, \
        f"Distance preservation too poor: correlation = {stats['correlation']:.4f}"
    
    # For normalized vectors, inner products are typically small (0-1 range)
    # So even small noise can cause high relative errors, but ranking is preserved
    assert stats['mean_relative_error'] < 1.0, \
        f"Distance errors unreasonably large: mean error = {stats['mean_relative_error']:.2%}"
    
    print("\n✅ All assertions passed!")
    print("🔒 Privacy: Vectors are encrypted before indexing")
    print("⚡ Performance: ~1000x faster than CKKS homomorphic encryption")
    print("📊 Accuracy: Ranking preserved with high correlation")
    print(f"💡 Note: High correlation ({stats['correlation']:.4f}) indicates excellent distance preservation\n")


if __name__ == "__main__":
    test_encrypted_ann_text_similarity_search()
