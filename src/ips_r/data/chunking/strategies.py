"""Concrete chunking strategies (character-, sentence- and token-based).

All strategies are pure functions of the text and their parameters, so the same
document and config always produce the same chunks.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, ClassVar, Final

from ips_r.core.exceptions import ConfigurationError
from ips_r.data.chunking.base import CHUNKERS, AbstractChunker, Span
from ips_r.data.text import (
    DEFAULT_ABBREVIATIONS,
    DEFAULT_TOKEN_PATTERN,
    compile_pattern,
    sentence_spans,
    token_spans,
)

DEFAULT_SEPARATORS: Final[tuple[str, ...]] = ("\n\n", "\n", ". ", " ", "")
"""Separators tried in order by the recursive character splitter (``""`` = hard cut)."""


def _require_overlap(chunker: AbstractChunker, overlap: int, size: int, unit: str) -> None:
    """Validate ``0 <= overlap < size``."""
    if overlap >= size:
        raise ConfigurationError(
            f"chunker '{chunker.name}': chunk_overlap ({overlap}) must be smaller than "
            f"chunk_size ({size}) {unit}"
        )


def merge_pieces(pieces: Sequence[Span], size: int, overlap: int) -> list[Span]:
    """Greedily merge contiguous pieces into windows of at most ``size`` characters.

    Consecutive windows share trailing pieces totalling at most ``overlap`` characters,
    provided the next piece still fits in the window (so no window is ever contained in
    its predecessor). A single piece longer than ``size`` becomes its own window.
    """
    windows: list[Span] = []
    first = 0
    while first < len(pieces):
        last = first
        while last + 1 < len(pieces) and pieces[last + 1][1] - pieces[first][0] <= size:
            last += 1
        windows.append((pieces[first][0], pieces[last][1]))
        if last + 1 >= len(pieces):
            break
        nxt = last + 1
        while (
            nxt - 1 > first
            and pieces[last][1] - pieces[nxt - 1][0] <= overlap
            and pieces[last + 1][1] - pieces[nxt - 1][0] <= size
        ):
            nxt -= 1
        first = nxt
    return windows


@CHUNKERS.register("recursive_character")
class RecursiveCharacterChunker(AbstractChunker):
    """Recursively split on a hierarchy of separators, then merge up to ``chunk_size``.

    Parameters:
        chunk_size: Maximum chunk length in characters.
        chunk_overlap: Maximum overlap between consecutive chunks in characters.
        separators: Separators tried in order; ``""`` performs hard cuts.
    """

    strategy: ClassVar[str] = "recursive_character"

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self._size = self.int_param("chunk_size", 1000, minimum=1)
        self._overlap = self.int_param("chunk_overlap", 200, minimum=0)
        self._separators = self.str_list_param("separators", list(DEFAULT_SEPARATORS))
        _require_overlap(self, self._overlap, self._size, "characters")
        if not self._separators:
            raise ConfigurationError(f"chunker '{self.name}': separators must not be empty")

    def _split(self, text: str) -> list[Span]:
        pieces = self._pieces(text, 0, len(text), self._separators)
        return merge_pieces(pieces, self._size, self._overlap)

    def _pieces(self, text: str, start: int, end: int, separators: Sequence[str]) -> list[Span]:
        """Split ``text[start:end]`` into contiguous pieces no longer than ``chunk_size``."""
        if end - start <= self._size:
            return [(start, end)]
        for position, separator in enumerate(separators):
            if separator == "":
                return [(i, min(i + self._size, end)) for i in range(start, end, self._size)]
            if text.find(separator, start, end) == -1:
                continue
            pieces: list[Span] = []
            cursor = start
            while (hit := text.find(separator, cursor, end)) != -1:
                pieces.append((cursor, hit + len(separator)))
                cursor = hit + len(separator)
            if cursor < end:
                pieces.append((cursor, end))
            result: list[Span] = []
            for piece in pieces:
                if piece[1] - piece[0] <= self._size:
                    result.append(piece)
                else:
                    result.extend(self._pieces(text, *piece, separators[position + 1 :]))
            return result
        return [(start, end)]


@CHUNKERS.register("sentence")
class SentenceChunker(AbstractChunker):
    """Group consecutive sentences into chunks.

    Parameters:
        sentences_per_chunk: Maximum number of sentences per chunk.
        sentence_overlap: Sentences shared by consecutive chunks.
        max_chunk_chars: Start a new chunk early once this length would be exceeded
            (``null`` = no limit). A single longer sentence becomes its own chunk.
        abbreviations: Tokens whose trailing period does not end a sentence.
    """

    strategy: ClassVar[str] = "sentence"

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self._per_chunk = self.int_param("sentences_per_chunk", 5, minimum=1)
        self._overlap = self.int_param("sentence_overlap", 1, minimum=0)
        self._max_chars = self.optional_int_param("max_chunk_chars", minimum=1)
        self._abbreviations = self.str_list_param("abbreviations", list(DEFAULT_ABBREVIATIONS))
        _require_overlap(self, self._overlap, self._per_chunk, "sentences")

    def _split(self, text: str) -> list[Span]:
        sentences = sentence_spans(text, self._abbreviations)
        chunks: list[Span] = []
        first = 0
        while first < len(sentences):
            last = first
            while (
                last + 1 < len(sentences)
                and last + 1 - first < self._per_chunk
                and (
                    self._max_chars is None
                    or sentences[last + 1][1] - sentences[first][0] <= self._max_chars
                )
            ):
                last += 1
            chunks.append((sentences[first][0], sentences[last][1]))
            if last + 1 >= len(sentences):
                break
            first = max(first + 1, last + 1 - self._overlap)
        return chunks


@CHUNKERS.register("token")
class TokenChunker(AbstractChunker):
    """Fixed-size windows of tokens (regex tokenizer; no model vocabulary).

    Parameters:
        chunk_size: Tokens per chunk.
        chunk_overlap: Tokens shared by consecutive chunks.
        token_pattern: Regular expression matching one token.
    """

    strategy: ClassVar[str] = "token"

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self._size = self.int_param("chunk_size", 256, minimum=1)
        self._overlap = self.int_param("chunk_overlap", 32, minimum=0)
        self._pattern = self.str_param("token_pattern", DEFAULT_TOKEN_PATTERN)
        _require_overlap(self, self._overlap, self._size, "tokens")
        try:
            compile_pattern(self._pattern)
        except re.error as exc:
            raise ConfigurationError(
                f"chunker '{self.name}': invalid token_pattern: {exc}"
            ) from exc

    def _split(self, text: str) -> list[Span]:
        tokens = token_spans(text, self._pattern)
        step = self._size - self._overlap
        chunks: list[Span] = []
        for first in range(0, len(tokens), step):
            last = min(first + self._size, len(tokens)) - 1
            chunks.append((tokens[first][0], tokens[last][1]))
            if last == len(tokens) - 1:
                break
        return chunks


@CHUNKERS.register("sliding_window")
class SlidingWindowChunker(AbstractChunker):
    """Fixed-width character windows advanced by a constant stride.

    Parameters:
        window_size: Window width in characters.
        stride: Characters between window starts (``stride < window_size`` → overlap).
        respect_word_boundaries: Shrink windows so they neither start nor end mid-word.
    """

    strategy: ClassVar[str] = "sliding_window"

    def __init__(self, config: Any = None, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        self._window = self.int_param("window_size", 1000, minimum=1)
        self._stride = self.int_param("stride", 500, minimum=1)
        self._word_boundaries = self.bool_param("respect_word_boundaries", True)
        if self._stride > self._window:
            raise ConfigurationError(
                f"chunker '{self.name}': stride ({self._stride}) must not exceed "
                f"window_size ({self._window})"
            )

    def _split(self, text: str) -> list[Span]:
        chunks: list[Span] = []
        for start in range(0, len(text), self._stride):
            end = min(start + self._window, len(text))
            chunks.append(self._snap(text, start, end) if self._word_boundaries else (start, end))
            if end == len(text):
                break
        return chunks

    @staticmethod
    def _snap(text: str, start: int, end: int) -> Span:
        """Move ``start`` forward and ``end`` backward to whitespace, when possible."""
        if 0 < start < end and not text[start - 1].isspace() and not text[start].isspace():
            boundary = next((i for i in range(start, end) if text[i].isspace()), None)
            if boundary is not None:
                start = boundary
        if start < end < len(text) and not text[end].isspace() and not text[end - 1].isspace():
            boundary = next((i for i in range(end - 1, start, -1) if text[i].isspace()), None)
            if boundary is not None:
                end = boundary
        return start, end
