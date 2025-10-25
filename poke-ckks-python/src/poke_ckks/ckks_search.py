"""Homomorphic dot product search using the CKKS scheme."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from Pyfhel import PyCtxt, Pyfhel


@dataclass
class EncryptedVector:
    """Container carrying an encrypted vector and optional metadata."""

    ciphertext: PyCtxt
    length: int
    identifier: str | None = None


class CKKSDotProductSearch:
    """Compute homomorphic dot products and retrieve top-k similarities."""

    def __init__(
        self,
        vector_size: int,
        *,
        scale: float = 2**40,
        polynomial_degree: int = 2**13,
        qi_sizes: Sequence[int] | None = None,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        if not self._is_power_of_two(polynomial_degree):
            raise ValueError("polynomial_degree must be a power of two for CKKS")
        if vector_size > polynomial_degree // 2:
            raise ValueError(
                "vector_size must not exceed half the polynomial degree (number of CKKS slots)"
            )
        self.vector_size = vector_size
        self.polynomial_degree = polynomial_degree
        self.scale = scale
        self.he = Pyfhel()
        modulus_chain = list(qi_sizes) if qi_sizes is not None else [60, 40, 40, 60]
        if len(modulus_chain) < 2:
            raise ValueError("qi_sizes must contain at least two primes")
        self.he.contextGen(
            scheme="CKKS",
            n=polynomial_degree,
            scale=int(scale),
            qi_sizes=modulus_chain,
        )
        self.he.keyGen()
        self.he.relinKeyGen()

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0

    def encrypt_vector(self, vector: Sequence[float], identifier: str | None = None) -> EncryptedVector:
        arr = self._prepare_vector(vector)
        ciphertext = self.he.encryptFrac(arr)
        return EncryptedVector(ciphertext=ciphertext, length=arr.size, identifier=identifier)

    def encrypt_vectors(self, vectors: Iterable[Sequence[float]]) -> List[EncryptedVector]:
        encrypted: List[EncryptedVector] = []
        for idx, vector in enumerate(vectors):
            identifier = f"vec_{idx}"
            encrypted.append(self.encrypt_vector(vector, identifier=identifier))
        return encrypted

    def homomorphic_dot_product(self, plain_vector: Sequence[float], encrypted_vector: EncryptedVector) -> PyCtxt:
        arr = self._prepare_vector(plain_vector)
        if encrypted_vector.length != arr.size:
            raise ValueError("vector length mismatch between query and encrypted vector")
        plain_ptxt = self.he.encodeFrac(arr)
        product = encrypted_vector.ciphertext.copy()
        product *= plain_ptxt
        self.he.rescale_to_next(product)
        # Return the element-wise product ciphertext
        # The sum will be computed after decryption
        return product

    def compute_similarity_ciphertexts(
        self,
        query_vector: Sequence[float],
        encrypted_vectors: Sequence[EncryptedVector],
    ) -> List[Tuple[str | None, PyCtxt]]:
        results: List[Tuple[str | None, PyCtxt]] = []
        for encrypted in encrypted_vectors:
            score_ctxt = self.homomorphic_dot_product(query_vector, encrypted)
            results.append((encrypted.identifier, score_ctxt))
        return results

    def decrypt_scores(self, ciphertexts: Sequence[Tuple[str | None, PyCtxt]]) -> List[Tuple[str | None, float]]:
        scores: List[Tuple[str | None, float]] = []
        for identifier, ctxt in ciphertexts:
            decoded = self.he.decryptFrac(ctxt)
            # Sum the first vector_size slots to get the dot product
            score = float(np.sum(np.real(decoded[:self.vector_size])))
            scores.append((identifier, score))
        return scores

    def top_k(
        self,
        query_vector: Sequence[float],
        encrypted_vectors: Sequence[EncryptedVector],
        k: int,
    ) -> List[Tuple[str | None, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        scored = self.decrypt_scores(
            self.compute_similarity_ciphertexts(query_vector, encrypted_vectors)
        )
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def _prepare_vector(self, vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("vectors must be one-dimensional")
        if arr.size != self.vector_size:
            raise ValueError(
                f"expected vectors of length {self.vector_size}, received {arr.size}"
            )
        return arr
