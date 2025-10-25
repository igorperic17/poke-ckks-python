"""Test CKKS similarity search with text paragraphs."""

from __future__ import annotations

import time
import numpy as np
import pytest

from poke_ckks import CKKSDotProductSearch

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


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


def get_embedding_model():
    """Load the sentence transformer model (cached after first call)."""
    if not HAS_SENTENCE_TRANSFORMERS:
        pytest.skip("sentence-transformers not installed")
    
    # Use a high-quality multilingual model
    # 'all-MiniLM-L6-v2' is fast and produces 384-dim embeddings
    # Alternative: 'all-mpnet-base-v2' for higher quality (768-dim)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


def text_to_vector(text: str, model: SentenceTransformer, target_dimension: int = 128) -> np.ndarray:
    """
    Convert text to a fixed-dimension vector using SOTA sentence embeddings.
    
    Uses sentence-transformers to create semantic embeddings, then optionally
    reduces dimensionality to fit CKKS constraints.
    """
    # Get semantic embedding from the model
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    
    # If target dimension is different, we need to reduce/expand
    if target_dimension != embedding.shape[0]:
        # Simple dimensionality reduction: take first N dimensions
        # In production, consider PCA or other proper reduction methods
        if target_dimension < embedding.shape[0]:
            vector = embedding[:target_dimension]
        else:
            # If we need more dimensions, pad with zeros
            vector = np.zeros(target_dimension, dtype=np.float64)
            vector[:embedding.shape[0]] = embedding
    else:
        vector = embedding.astype(np.float64)
    
    # Normalize to unit length
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector


def test_encrypted_paragraph_similarity_search() -> None:
    """
    Test homomorphic similarity search with 20 text paragraphs using SOTA embeddings.
    
    Uses sentence-transformers (all-MiniLM-L6-v2) for semantic embeddings.
    
    Simulates:
    1. Private insertion of paragraph embeddings into encrypted database
    2. Query for most similar paragraphs using homomorphic computation
    3. Retrieval of top-3 results without decrypting the database
    """
    
    print("\n" + "="*80)
    print("HOMOMORPHIC TEXT SIMILARITY SEARCH TEST - 20 PARAGRAPHS")
    print("Using SOTA Sentence Embeddings (all-MiniLM-L6-v2)")
    print("="*80)
    
    # Configuration
    dimension = 128  # Reduced from 384 (native model size) for faster CKKS operations
    num_paragraphs = len(TEXT_DATABASE)
    top_k = 3
    
    print(f"\nConfiguration:")
    print(f"  Number of paragraphs: {num_paragraphs}")
    print(f"  Embedding model: all-MiniLM-L6-v2 (384-dim)")
    print(f"  CKKS dimension: {dimension} (reduced for performance)")
    print(f"  Top-K results: {top_k}")
    
    # Step 0: Load embedding model
    print(f"\n[0/5] Loading sentence transformer model...")
    start_time = time.time()
    model = get_embedding_model()
    model_load_time = time.time() - start_time
    print(f"  ✓ Completed in {model_load_time:.3f}s")
    
    # Step 1: Create embeddings
    print(f"\n[1/5] Creating semantic embeddings for {num_paragraphs} paragraphs...")
    start_time = time.time()
    embeddings = [text_to_vector(text, model, dimension) for text in TEXT_DATABASE]
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
    query_embedding = text_to_vector(query_text, model, dimension)
    
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
    total_time = model_load_time + embed_time + init_time + encrypt_time + search_time
    print(f"\n" + "="*80)
    print("⏱️  TIMING SUMMARY:")
    print("="*80)
    print(f"  0. Model loading:             {model_load_time:8.3f}s  ({model_load_time/total_time*100:5.1f}%)")
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
