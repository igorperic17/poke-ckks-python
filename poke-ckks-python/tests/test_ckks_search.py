from __future__ import annotations

import numpy as np
import pytest

from poke_ckks import CKKSDotProductSearch


def test_top_k_matches_plain_computation() -> None:
    dimension = 4
    search = CKKSDotProductSearch(vector_size=dimension)
    catalog = [
        [0.25, 0.1, 0.9, 0.3],
        [0.8, 0.05, 0.1, 0.2],
        [0.4, 0.4, 0.4, 0.4],
        [0.0, 0.7, 0.2, 0.1],
    ]
    encrypted_vectors = [
        search.encrypt_vector(vector, identifier=f"item_{idx}")
        for idx, vector in enumerate(catalog)
    ]
    query = [0.5, 0.2, 0.7, 0.1]
    top_matches = search.top_k(query, encrypted_vectors, k=3)
    plain_scores = [
        (f"item_{idx}", float(np.dot(query, vector)))
        for idx, vector in enumerate(catalog)
    ]
    expected = sorted(plain_scores, key=lambda pair: pair[1], reverse=True)[:3]
    assert len(top_matches) == len(expected)
    for (encrypted_id, encrypted_score), (plain_id, plain_score) in zip(
        top_matches, expected
    ):
        assert encrypted_id == plain_id
        assert encrypted_score == pytest.approx(plain_score, rel=1e-3, abs=1e-3)
