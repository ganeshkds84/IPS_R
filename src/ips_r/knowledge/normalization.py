"""Configurable, offset-preserving text cleaning and normalisation.

Every step is a regex substitution over a :class:`TrackedText`, which records for each
output character the span of original characters it came from. Normalised text can
therefore always be traced back to exact offsets in the source document, which is
what lets chunks built on cleaned text carry original-document spans.

Step order (each switchable in ``configs/normalization/default.yaml``):

1. Unicode normalisation (NFC / NFKC / NFD / NFKD / none)
2. Control and zero-width character removal
3. Ligature expansion (``ﬁ`` → ``fi`` …)
4. Hyphen normalisation (soft hyphens, line-break hyphenation, dash variants)
5. Quote normalisation (curly / angle quotes → ASCII)
6. Punctuation removal (only when ``preserve_punctuation`` is false)
7. Lower-casing (optional)
8. Whitespace normalisation (Unicode spaces, runs, blank lines, trimming)
"""

from __future__ import annotations

import re
import unicodedata
from array import array
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Final, Self

from ips_r.config.schema import NormalizationConfig
from ips_r.core.exceptions import ConfigurationError
from ips_r.knowledge.hashing import content_hash

UNICODE_FORMS: Final = ("NFC", "NFKC", "NFD", "NFKD", "none")

LIGATURES: Final[Mapping[str, str]] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
    "\u0132": "IJ",
    "\u0133": "ij",
    "\ua732": "AA",
    "\ua733": "aa",
}
"""Built-in ligature table; ``normalization.ligatures`` adds or overrides entries."""

_COMBINING: Final = "\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f"
# A base character followed by combining marks (so "e" + U+0301 composes under NFC),
# or any single non-ASCII character. Pure ASCII is invariant under every form.
_NON_ASCII_CLUSTER: Final = re.compile(f"[^{_COMBINING}][{_COMBINING}]+|[^\\x00-\\x7f]", re.DOTALL)
_CONTROL: Final = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_ZERO_WIDTH: Final = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_SOFT_HYPHEN: Final = re.compile("\u00ad")
_LINE_BREAK_HYPHEN: Final = re.compile(r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=[a-z])")
_DASHES: Final = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]")
_SINGLE_QUOTES: Final = re.compile("[\u2018\u2019\u201a\u201b\u2032\u00b4`]")
_DOUBLE_QUOTES: Final = re.compile("[\u201c\u201d\u201e\u201f\u2033\u00ab\u00bb]")
_PUNCTUATION: Final = re.compile(r"[^\w\s]")
_UNICODE_SPACE: Final = re.compile("[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000\t\f\v]")
_CRLF: Final = re.compile(r"\r\n?")
_SPACE_RUN: Final = re.compile(r" {2,}")
_SPACE_AROUND_NEWLINE: Final = re.compile(r" *\n *")
_BLANK_LINES: Final = re.compile(r"\n{3,}")
_ANY_WHITESPACE_RUN: Final = re.compile(r"\s+")
_EDGES: Final = re.compile(r"^\s+|\s+$")


class TrackedText:
    """Text plus, for every character, the ``[start, end)`` span it covers in the source.

    Args:
        text: Current text.
        starts: Source start offset of each character.
        ends: Source end offset of each character.
    """

    __slots__ = ("_ends", "_starts", "text")

    def __init__(self, text: str, starts: array[int], ends: array[int]) -> None:
        if not len(text) == len(starts) == len(ends):
            raise ValueError("TrackedText offsets must have one entry per character")
        self.text = text
        self._starts = starts
        self._ends = ends

    @classmethod
    def original(cls, text: str, offset: int = 0) -> Self:
        """Wrap untouched source text whose first character sits at ``offset``."""
        starts = array("q", range(offset, offset + len(text)))
        ends = array("q", range(offset + 1, offset + len(text) + 1))
        return cls(text, starts, ends)

    @classmethod
    def concat(cls, parts: list[TrackedText], separator: str = "") -> TrackedText:
        """Join texts; separator characters map to the end of the preceding part."""
        text_parts: list[str] = []
        starts: array[int] = array("q")
        ends: array[int] = array("q")
        for i, part in enumerate(parts):
            if i and separator:
                anchor = ends[-1] if ends else (part._starts[0] if part.text else 0)
                text_parts.append(separator)
                starts.extend([anchor] * len(separator))
                ends.extend([anchor] * len(separator))
            text_parts.append(part.text)
            starts.extend(part._starts)
            ends.extend(part._ends)
        return cls("".join(text_parts), starts, ends)

    def __len__(self) -> int:
        return len(self.text)

    def source_span(self, start: int, end: int) -> tuple[int, int]:
        """Return the source span covered by ``text[start:end]`` (``end > start``).

        Raises:
            ValueError: If the span is empty or out of range.
        """
        if not 0 <= start < end <= len(self.text):
            raise ValueError(f"invalid span [{start}, {end}) for text of length {len(self.text)}")
        return self._starts[start], max(self._ends[start:end])

    def slice(self, start: int, end: int) -> TrackedText:
        """Return ``text[start:end]`` with its offsets."""
        return TrackedText(self.text[start:end], self._starts[start:end], self._ends[start:end])

    def sub(
        self, pattern: re.Pattern[str], repl: Callable[[re.Match[str]], str] | str
    ) -> TrackedText:
        """Regex substitution that keeps the offset map consistent.

        A replacement of the same length as its match maps character-to-character;
        otherwise every replacement character maps to the whole matched source span.
        """
        text = self.text
        out: list[str] = []
        starts: array[int] = array("q")
        ends: array[int] = array("q")
        position = 0
        changed = False
        for match in pattern.finditer(text):
            begin, finish = match.span()
            replacement = repl if isinstance(repl, str) else repl(match)
            if replacement == match.group(0):
                continue
            changed = True
            out.append(text[position:begin])
            starts.extend(self._starts[position:begin])
            ends.extend(self._ends[position:begin])
            if replacement:
                if len(replacement) == finish - begin:
                    starts.extend(self._starts[begin:finish])
                    ends.extend(self._ends[begin:finish])
                else:
                    source = self._anchor(begin, finish)
                    starts.extend([source[0]] * len(replacement))
                    ends.extend([source[1]] * len(replacement))
                out.append(replacement)
            position = finish
        if not changed:
            return self
        out.append(text[position:])
        starts.extend(self._starts[position:])
        ends.extend(self._ends[position:])
        return TrackedText("".join(out), starts, ends)

    def _anchor(self, begin: int, finish: int) -> tuple[int, int]:
        """Source span for a replacement of ``text[begin:finish]`` (possibly empty)."""
        if finish > begin:
            return self._starts[begin], max(self._ends[begin:finish])
        if begin < len(self.text):
            return self._starts[begin], self._starts[begin]
        last = self._ends[-1] if self.text else 0
        return last, last


class TextNormalizer:
    """Applies the configured cleaning steps to text.

    Args:
        settings: Normalisation options; defaults match ``configs/normalization/default.yaml``.

    Raises:
        ConfigurationError: If ``unicode_form`` is unknown or the ligature table is invalid.
    """

    def __init__(self, settings: NormalizationConfig | None = None) -> None:
        self._s = settings or NormalizationConfig()
        if self._s.unicode_form not in UNICODE_FORMS:
            raise ConfigurationError(
                f"normalization.unicode_form must be one of {UNICODE_FORMS}, "
                f"got {self._s.unicode_form!r}"
            )
        table = {**LIGATURES, **dict(self._s.ligatures)}
        if any(not key for key in table):
            raise ConfigurationError("normalization.ligatures keys must be non-empty")
        self._ligatures = table
        keys = sorted(table, key=lambda k: (-len(k), k))
        self._ligature_pattern = re.compile("|".join(re.escape(k) for k in keys))

    @property
    def settings(self) -> NormalizationConfig:
        """The options in effect."""
        return self._s

    @property
    def fingerprint(self) -> str:
        """Content hash of the settings (part of every downstream artifact hash)."""
        return content_hash({"normalization": asdict(self._s)})

    def normalize(self, text: str, offset: int = 0) -> TrackedText:
        """Normalise ``text`` whose first character sits at source ``offset``."""
        tracked = TrackedText.original(text, offset)
        return self.normalize_tracked(tracked)

    def normalize_tracked(self, tracked: TrackedText) -> TrackedText:
        """Normalise already-tracked text."""
        s = self._s
        if s.unicode_form != "none" and not unicodedata.is_normalized(s.unicode_form, tracked.text):  # type: ignore[arg-type]
            form = s.unicode_form
            tracked = tracked.sub(_NON_ASCII_CLUSTER, lambda m: unicodedata.normalize(form, m.group(0)))  # type: ignore[arg-type]
        if s.remove_control_chars:
            tracked = tracked.sub(_CONTROL, "")
        if s.remove_zero_width:
            tracked = tracked.sub(_ZERO_WIDTH, "")
        if s.expand_ligatures:
            tracked = tracked.sub(self._ligature_pattern, lambda m: self._ligatures[m.group(0)])
        if s.normalize_hyphens:
            tracked = tracked.sub(_SOFT_HYPHEN, "")
            tracked = tracked.sub(_DASHES, s.dash_replacement)
        if s.dehyphenate:
            tracked = tracked.sub(_LINE_BREAK_HYPHEN, "")
        if s.normalize_quotes:
            tracked = tracked.sub(_SINGLE_QUOTES, "'").sub(_DOUBLE_QUOTES, '"')
        if not s.preserve_punctuation:
            tracked = tracked.sub(_PUNCTUATION, " ")
        if s.lowercase:
            tracked = self._lowercase(tracked)
        if s.normalize_whitespace:
            tracked = self._whitespace(tracked)
        return tracked

    def normalize_text(self, text: str) -> str:
        """Convenience wrapper returning only the normalised string."""
        return self.normalize(text).text

    @staticmethod
    def _lowercase(tracked: TrackedText) -> TrackedText:
        """Lower-case, keeping the offset map aligned even for length-changing characters."""
        lowered = tracked.text.lower()
        if len(lowered) == len(tracked.text):
            return TrackedText(lowered, tracked._starts, tracked._ends)
        return tracked.sub(re.compile(r"[^\x00-\x7f]|[A-Z]+"), lambda m: m.group(0).lower())

    def _whitespace(self, tracked: TrackedText) -> TrackedText:
        """Normalise spaces and line breaks, then trim."""
        tracked = tracked.sub(_UNICODE_SPACE, " ").sub(_CRLF, "\n")
        if self._s.preserve_newlines:
            tracked = tracked.sub(_SPACE_RUN, " ")
            tracked = tracked.sub(_SPACE_AROUND_NEWLINE, "\n").sub(_BLANK_LINES, "\n\n")
        else:
            tracked = tracked.sub(_ANY_WHITESPACE_RUN, " ")
        return tracked.sub(_EDGES, "")
