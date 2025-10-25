"""Run a small CKKS dot product demo."""

from __future__ import annotations

from poke_ckks import CKKSDotProductSearch


def main() -> None:
    dimension = 4
    search = CKKSDotProductSearch(vector_size=dimension)
    catalog = [
        [0.25, 0.1, 0.9, 0.3],
        [0.8, 0.05, 0.1, 0.2],
        [0.4, 0.4, 0.4, 0.4],
    ]
    encrypted_vectors = [
        search.encrypt_vector(vector, identifier=f"item_{idx}")
        for idx, vector in enumerate(catalog)
    ]
    query = [0.5, 0.2, 0.7, 0.1]
    top_matches = search.top_k(query, encrypted_vectors, k=2)
    print("Top matches (identifier, dot product):")
    for identifier, score in top_matches:
        print(identifier, round(score, 6))


if __name__ == "__main__":
    main()
