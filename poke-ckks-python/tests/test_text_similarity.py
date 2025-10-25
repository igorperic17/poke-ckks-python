"""Test CKKS similarity search with text paragraphs."""

from __future__ import annotations

import time
import hashlib
import numpy as np
import pytest

from poke_ckks import CKKSDotProductSearch


# 20 hardcoded text paragraphs across various topics
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


def text_to_vector(text: str, dimension: int = 128) -> np.ndarray:
    """
    Convert text to a fixed-dimension vector using a simple hashing approach.
    
    In production, you would use proper embeddings (e.g., sentence-transformers),
    but for testing purposes, we create a deterministic vector from text.
    """
    # Create a deterministic hash-based vector
    vector = np.zeros(dimension, dtype=np.float64)
    
    # Hash the text to get a seed
    text_hash = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    rng = np.random.RandomState(text_hash % (2**32))
    
    # Generate a random vector based on the hash
    base_vector = rng.randn(dimension)
    
    # Normalize to unit length
    vector = base_vector / np.linalg.norm(base_vector)
    
    # Add some structure based on word frequency
    words = text.lower().split()
    for i, word in enumerate(words[:dimension]):
        # Add word-specific features
        word_val = sum(ord(c) for c in word) / len(word) if word else 0
        vector[i % dimension] += word_val * 0.01
    
    # Re-normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector


def test_encrypted_paragraph_similarity_search() -> None:
    """
    Test homomorphic similarity search with 20 text paragraphs.
    
    Simulates:
    1. Private insertion of paragraph embeddings into encrypted database
    2. Query for most similar paragraphs using homomorphic computation
    3. Retrieval of top-3 results without decrypting the database
    """
    
    print("\n" + "="*80)
    print("HOMOMORPHIC TEXT SIMILARITY SEARCH TEST - 20 PARAGRAPHS")
    print("="*80)
    
    # Configuration
    dimension = 128
    num_paragraphs = len(TEXT_DATABASE)
    top_k = 3
    
    print(f"\nConfiguration:")
    print(f"  Number of paragraphs: {num_paragraphs}")
    print(f"  Embedding dimension: {dimension}")
    print(f"  Top-K results: {top_k}")
    
    # Step 1: Create embeddings
    print(f"\n[1/5] Creating embeddings for {num_paragraphs} paragraphs...")
    start_time = time.time()
    embeddings = [text_to_vector(text, dimension) for text in TEXT_DATABASE]
    embed_time = time.time() - start_time
    print(f"  ✓ Completed in {embed_time:.3f}s ({embed_time/num_paragraphs*1000:.1f}ms per paragraph)")
    
    # Step 2: Initialize CKKS search engine
    print(f"\n[2/5] Initializing CKKS encryption context...")
    start_time = time.time()
    search = CKKSDotProductSearch(vector_size=dimension)
    init_time = time.time() - start_time
    print(f"  ✓ Completed in {init_time:.3f}s")
    
    # Step 3: Encrypt all paragraph embeddings (simulating private DB insertion)
    print(f"\n[3/5] Encrypting {num_paragraphs} paragraph embeddings (private DB insertion)...")
    start_time = time.time()
    encrypted_vectors = []
    for idx, emb in enumerate(embeddings):
        encrypted_vec = search.encrypt_vector(emb, identifier=f"para_{idx}")
        encrypted_vectors.append(encrypted_vec)
        if (idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"    Progress: {idx+1}/{num_paragraphs} ({elapsed:.2f}s)")
    encrypt_time = time.time() - start_time
    print(f"  ✓ Completed in {encrypt_time:.3f}s")
    print(f"  ✓ Average encryption time: {encrypt_time/num_paragraphs*1000:.1f}ms per vector")
    
    # Step 4: Create query embedding
    query_text = "Deep learning and neural networks transform artificial intelligence applications."
    print(f"\n[4/5] Creating query embedding...")
    print(f'  Query: "{query_text}"')
    query_embedding = text_to_vector(query_text, dimension)
    
    # Step 5: Perform homomorphic similarity search
    print(f"\n[5/5] Performing homomorphic similarity search over encrypted database...")
    start_time = time.time()
    top_matches = search.top_k(query_embedding, encrypted_vectors, k=top_k)
    search_time = time.time() - start_time
    print(f"  ✓ Completed in {search_time:.3f}s")
    print(f"  ✓ Average search time: {search_time/num_paragraphs*1000:.1f}ms per encrypted vector")
    
    # Display results
    print(f"\n" + "-"*80)
    print(f"TOP {top_k} MOST SIMILAR PARAGRAPHS (Homomorphic Encrypted Search):")
    print("-"*80)
    for rank, (identifier, score) in enumerate(top_matches, 1):
        para_idx = int(identifier.split('_')[1])
        print(f"\n#{rank} (Similarity Score: {score:.6f})")
        print(f"  ID: {identifier}")
        print(f"  Category: {_get_category(para_idx)}")
        print(f"  Text: {TEXT_DATABASE[para_idx]}")
    
    # Verify against plaintext computation
    print(f"\n" + "-"*80)
    print("VERIFICATION (Plaintext computation for testing):")
    print("-"*80)
    plain_scores = [
        (f"para_{idx}", float(np.dot(query_embedding, emb)))
        for idx, emb in enumerate(embeddings)
    ]
    plain_scores.sort(key=lambda x: x[1], reverse=True)
    expected_top_k = plain_scores[:top_k]
    
    print("Expected top-3 matches:")
    for rank, (identifier, score) in enumerate(expected_top_k, 1):
        para_idx = int(identifier.split('_')[1])
        print(f"  #{rank} {identifier}: {score:.6f} - {_get_category(para_idx)}")
    
    # Timing summary
    total_time = embed_time + init_time + encrypt_time + search_time
    print(f"\n" + "="*80)
    print("⏱️  TIMING SUMMARY:")
    print("="*80)
    print(f"  1. Embedding generation:      {embed_time:8.3f}s  ({embed_time/total_time*100:5.1f}%)")
    print(f"  2. CKKS context init:         {init_time:8.3f}s  ({init_time/total_time*100:5.1f}%)")
    print(f"  3. Encryption (20 vectors):   {encrypt_time:8.3f}s  ({encrypt_time/total_time*100:5.1f}%)")
    print(f"     - Per vector:              {encrypt_time/num_paragraphs*1000:8.1f}ms")
    print(f"  4. Homomorphic search:        {search_time:8.3f}s  ({search_time/total_time*100:5.1f}%)")
    print(f"     - Per vector comparison:   {search_time/num_paragraphs*1000:8.1f}ms")
    print(f"  " + "-"*76)
    print(f"  TOTAL TIME:                   {total_time:8.3f}s")
    print("="*80)
    
    # Assertions
    assert len(top_matches) == top_k, f"Expected {top_k} results, got {len(top_matches)}"
    
    # Check that the order matches (allowing for small numerical differences in CKKS)
    for i, ((enc_id, enc_score), (plain_id, plain_score)) in enumerate(zip(top_matches, expected_top_k)):
        assert enc_id == plain_id, f"Rank {i+1}: Expected {plain_id}, got {enc_id}"
        # CKKS is approximate, so we allow some tolerance
        assert enc_score == pytest.approx(plain_score, rel=1e-3, abs=1e-3), \
            f"Rank {i+1}: Score mismatch for {enc_id}: {enc_score:.6f} vs {plain_score:.6f}"
    
    print("\n✅ All assertions passed! Homomorphic search matches plaintext results.")
    print("🔒 Privacy preserved: Database remained encrypted during search.\n")


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


if __name__ == "__main__":
    test_encrypted_paragraph_similarity_search()
