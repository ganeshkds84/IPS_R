"""Configurable chunk preparation (no embeddings): documents → :class:`Chunk` objects."""

from ips_r.data.chunking.base import CHUNKERS, AbstractChunker, Span, make_chunk_id
from ips_r.data.chunking.semantic import AbstractSemanticChunker, SemanticChunkerPlaceholder
from ips_r.data.chunking.strategies import (
    DEFAULT_SEPARATORS,
    RecursiveCharacterChunker,
    SentenceChunker,
    SlidingWindowChunker,
    TokenChunker,
    merge_pieces,
)

__all__ = [
    "CHUNKERS",
    "DEFAULT_SEPARATORS",
    "AbstractChunker",
    "AbstractSemanticChunker",
    "RecursiveCharacterChunker",
    "SemanticChunkerPlaceholder",
    "SentenceChunker",
    "SlidingWindowChunker",
    "Span",
    "TokenChunker",
    "make_chunk_id",
    "merge_pieces",
]
