"""Expose the CKKS-powered similarity search interface."""

from .ckks_search import CKKSDotProductSearch, EncryptedVector
from .text_embeddings import TextEmbedder, create_embedder
from .ann_search import ANNSearchIndex
from .encrypted_ann_search import (
    EncryptedANNSearch,
    SAPEncryptionKey,
    generate_sap_key,
    compute_distance_preservation_error,
)
from .oram_search import PathORAM, ORAMBlock, ORAMEncryptedSearch

__all__ = [
    "CKKSDotProductSearch",
    "EncryptedVector",
    "TextEmbedder",
    "create_embedder",
    "ANNSearchIndex",
    "EncryptedANNSearch",
    "SAPEncryptionKey",
    "generate_sap_key",
    "compute_distance_preservation_error",
    "PathORAM",
    "ORAMBlock",
    "ORAMEncryptedSearch",
]
