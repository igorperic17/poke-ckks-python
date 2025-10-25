"""Expose the CKKS-powered similarity search interface."""

from .ckks_search import CKKSDotProductSearch, EncryptedVector

__all__ = [
    "CKKSDotProductSearch",
    "EncryptedVector",
]
