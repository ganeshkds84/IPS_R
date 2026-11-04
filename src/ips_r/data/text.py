"""Pure text utilities: tokenisation, sentence segmentation, encoding checks, hashing.

Everything here is deterministic and independent of I/O, so it can be shared by
adapters, validation, chunking and statistics.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Final

DEFAULT_TOKEN_PATTERN: Final = r"\w+|[^\w\s]"
"""Default tokenizer: words and individual punctuation marks."""

DEFAULT_WORD_PATTERN: Final = r"\w+"
"""Default word pattern used for vocabulary statistics."""

DEFAULT_ABBREVIATIONS: Final[tuple[str, ...]] = (
    "e.g.",
    "i.e.",
    "et al.",
    "etc.",
    "vs.",
    "cf.",
    "fig.",
    "figs.",
    "eq.",
    "eqs.",
    "sec.",
    "no.",
    "vol.",
    "pp.",
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "inc.",
    "st.",
)
"""Abbreviations after which a period does not end a sentence (lower case)."""

_SENTENCE_END: Final = re.compile(r"[.!?]+[\"'\u201d\u2019)\]]*(?=\s+|$)")
_PARAGRAPH_BREAK: Final = re.compile(r"\n[ \t]*\n")
_INITIAL: Final = re.compile(r"(?:^|\s)[A-Z]\.$")
_MOJIBAKE: Final = re.compile("Ã[\u0080-¿]|â€|Â[\u00a0-¿]")
_ALLOWED_CONTROL: Final = frozenset("\n\r\t")
_SURROGATE_FIRST: Final = 0xD800
_SURROGATE_LAST: Final = 0xDFFF
_WHITESPACE: Final = re.compile(r"\s+")


@lru_cache(maxsize=32)
def compile_pattern(pattern: str) -> re.Pattern[str]:
    """Compile and cache a regular expression.

    Raises:
        re.error: If the pattern is invalid.
    """
    return re.compile(pattern)


def token_spans(text: str, pattern: str = DEFAULT_TOKEN_PATTERN) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans of the tokens matched by ``pattern``."""
    return [match.span() for match in compile_pattern(pattern).finditer(text)]


def tokenize(
    text: str, pattern: str = DEFAULT_TOKEN_PATTERN, *, lowercase: bool = False
) -> list[str]:
    """Split ``text`` into tokens matched by ``pattern``."""
    tokens = compile_pattern(pattern).findall(text)
    return [token.lower() for token in tokens] if lowercase else tokens


def count_tokens(text: str, pattern: str = DEFAULT_TOKEN_PATTERN) -> int:
    """Return the number of tokens matched by ``pattern``."""
    return sum(1 for _ in compile_pattern(pattern).finditer(text))


def _ends_with_abbreviation(prefix: str, abbreviations: frozenset[str]) -> bool:
    """Return whether ``prefix`` (text up to and including a period) ends in an abbreviation."""
    lowered = prefix.lower()
    if any(lowered.endswith(abbr) and _word_start(lowered, abbr) for abbr in abbreviations):
        return True
    return bool(_INITIAL.search(prefix))


def _word_start(lowered: str, abbr: str) -> bool:
    """Return whether ``abbr`` at the end of ``lowered`` starts at a word boundary."""
    index = len(lowered) - len(abbr)
    return index == 0 or not lowered[index - 1].isalnum()


def sentence_spans(
    text: str, abbreviations: Iterable[str] = DEFAULT_ABBREVIATIONS
) -> list[tuple[int, int]]:
    """Segment ``text`` into sentence spans with a rule-based, abbreviation-aware splitter.

    A sentence ends at ``.``, ``!`` or ``?`` (plus closing quotes/brackets) followed by
    whitespace, unless the period belongs to a known abbreviation or a single-letter
    initial. Blank lines always end a sentence. Spans exclude surrounding whitespace.
    """
    abbrs = frozenset(a.lower() for a in abbreviations)
    boundaries: set[int] = {match.start() for match in _PARAGRAPH_BREAK.finditer(text)}
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        if text[match.start()] == "." and _ends_with_abbreviation(text[: match.start() + 1], abbrs):
            continue
        boundaries.add(end)
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in [*sorted(boundaries), len(text)]:
        span = strip_span(text, start, boundary)
        if span is not None:
            spans.append(span)
        start = boundary
    return spans


def strip_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink ``[start, end)`` to exclude surrounding whitespace; ``None`` if nothing remains."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip the ends."""
    return _WHITESPACE.sub(" ", text).strip()


def encoding_problems(text: str) -> list[str]:
    """Describe encoding defects in ``text``.

    Detects Unicode replacement characters (failed decoding), lone surrogates,
    disallowed control characters and typical UTF-8-as-Latin-1 mojibake.
    """
    problems: list[str] = []
    if "�" in text:
        problems.append("contains U+FFFD replacement characters")
    if any(_SURROGATE_FIRST <= ord(char) <= _SURROGATE_LAST for char in text):
        problems.append("contains unpaired surrogate code points")
    if any(unicodedata.category(char) == "Cc" and char not in _ALLOWED_CONTROL for char in text):
        problems.append("contains control characters")
    if _MOJIBAKE.search(text):
        problems.append("contains mojibake (UTF-8 decoded as Latin-1)")
    return problems


def stable_hash(parts: Sequence[str], *, digest_size: int = 16) -> str:
    """Return a hex BLAKE2b digest of ``parts``, independent of ``PYTHONHASHSEED``."""
    hasher = hashlib.blake2b(digest_size=digest_size)
    for part in parts:
        hasher.update(part.encode("utf-8", errors="surrogatepass"))
        hasher.update(b"\x1f")
    return hasher.hexdigest()
