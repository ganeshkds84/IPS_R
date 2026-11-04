"""Dependency-free, deterministic language identification from stop-word profiles.

This is intentionally simple: it tells apart the major languages of academic text well
enough to populate :attr:`ips_r.data.models.DocumentMetadata.language`. Text that is too
short or ambiguous yields :data:`UNDETERMINED`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from ips_r.data.text import tokenize

UNDETERMINED: Final = "und"
"""ISO 639-2 code for an undetermined language."""

STOPWORDS: Final[Mapping[str, frozenset[str]]] = {
    "en": frozenset(
        [
            "the",
            "of",
            "and",
            "to",
            "in",
            "is",
            "that",
            "for",
            "are",
            "with",
            "as",
            "on",
            "this",
            "by",
            "be",
            "we",
            "an",
            "it",
            "from",
            "or",
            "which",
            "not",
            "have",
            "has",
            "were",
            "was",
            "can",
            "our",
            "these",
            "their",
            "at",
        ]
    ),
    "de": frozenset(
        [
            "der",
            "die",
            "und",
            "in",
            "den",
            "von",
            "zu",
            "das",
            "mit",
            "sich",
            "des",
            "auf",
            "für",
            "ist",
            "im",
            "dem",
            "nicht",
            "ein",
            "eine",
            "als",
            "auch",
            "es",
            "an",
            "werden",
            "aus",
            "er",
            "hat",
            "dass",
            "sie",
            "nach",
            "wird",
            "bei",
        ]
    ),
    "fr": frozenset(
        [
            "le",
            "la",
            "les",
            "de",
            "des",
            "et",
            "en",
            "du",
            "un",
            "une",
            "est",
            "que",
            "pour",
            "dans",
            "qui",
            "sur",
            "au",
            "par",
            "pas",
            "plus",
            "ce",
            "sont",
            "avec",
            "ne",
            "se",
            "aux",
            "nous",
            "ou",
        ]
    ),
    "es": frozenset(
        [
            "el",
            "la",
            "de",
            "que",
            "y",
            "en",
            "los",
            "las",
            "del",
            "se",
            "un",
            "una",
            "por",
            "con",
            "para",
            "es",
            "al",
            "lo",
            "como",
            "más",
            "pero",
            "sus",
            "le",
            "ya",
            "o",
            "este",
            "sí",
            "porque",
            "esta",
            "entre",
        ]
    ),
    "it": frozenset(
        [
            "il",
            "di",
            "che",
            "e",
            "la",
            "per",
            "un",
            "una",
            "in",
            "è",
            "del",
            "della",
            "sono",
            "con",
            "non",
            "si",
            "da",
            "come",
            "anche",
            "gli",
            "le",
            "dei",
            "nel",
            "alla",
            "più",
            "questo",
        ]
    ),
    "pt": frozenset(
        [
            "o",
            "a",
            "de",
            "que",
            "e",
            "do",
            "da",
            "em",
            "um",
            "uma",
            "para",
            "é",
            "com",
            "não",
            "os",
            "no",
            "se",
            "na",
            "por",
            "mais",
            "as",
            "dos",
            "como",
            "mas",
            "ao",
            "ele",
            "das",
            "à",
            "seu",
            "sua",
            "ou",
        ]
    ),
    "nl": frozenset(
        [
            "de",
            "het",
            "een",
            "en",
            "van",
            "in",
            "is",
            "dat",
            "op",
            "te",
            "zijn",
            "voor",
            "met",
            "die",
            "niet",
            "aan",
            "er",
            "om",
            "ook",
            "als",
            "bij",
            "of",
            "wordt",
            "door",
            "maar",
            "naar",
            "dan",
        ]
    ),
}
"""Stop-word profile per ISO 639-1 language code."""


def detect_language(
    text: str,
    *,
    min_tokens: int = 20,
    min_ratio: float = 0.05,
    profiles: Mapping[str, frozenset[str]] = STOPWORDS,
) -> str:
    """Return the ISO 639-1 code of the most likely language of ``text``.

    Args:
        text: Text to classify.
        min_tokens: Minimum number of word tokens required for a decision.
        min_ratio: Minimum fraction of tokens that must be stop words of the winner.
        profiles: Stop-word sets per language.

    Returns:
        A language code, or :data:`UNDETERMINED` if the evidence is insufficient or tied.
    """
    tokens = tokenize(text, r"[^\W\d_]+", lowercase=True)
    if len(tokens) < min_tokens:
        return UNDETERMINED
    scores = {
        code: sum(1 for token in tokens if token in words) for code, words in profiles.items()
    }
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_code, best_score = ranked[0]
    if best_score / len(tokens) < min_ratio:
        return UNDETERMINED
    if len(ranked) > 1 and ranked[1][1] == best_score:
        return UNDETERMINED
    return best_code
