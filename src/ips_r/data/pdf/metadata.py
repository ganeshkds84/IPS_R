"""Bibliographic metadata extraction for academic PDFs.

Combines the PDF info dictionary with the layout analysis (title, author line,
abstract blocks) and pattern matching over the front matter (first page) to fill
:class:`~ips_r.data.models.DocumentMetadata`: title, authors, year, venue, DOI,
keywords, abstract and language.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from ips_r.core.exceptions import ConfigurationError
from ips_r.data.language import UNDETERMINED, detect_language
from ips_r.data.models import DocumentMetadata, ElementType
from ips_r.data.pdf.layout import LayoutResult, settings_from_mapping

_DOI: Final = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>,;]+)", re.IGNORECASE)
_COPYRIGHT_YEAR: Final = re.compile(r"(?:©|\(c\)|copyright)\s*(\d{4})", re.IGNORECASE)
_ARXIV: Final = re.compile(r"\barXiv:\s?(\d{2})(\d{2})\.\d{4,5}(?:v\d+)?", re.IGNORECASE)
_PDF_DATE_YEAR: Final = re.compile(r"^(?:D:)?(\d{4})")
_YEAR: Final = re.compile(r"\b(\d{4})\b")
_MIN_AUTHOR_CHARS: Final = 2
_SHORT_YEAR_DIGITS: Final = 2  # "EMNLP '23"
_AUTHOR_MARKERS: Final = re.compile(r"[\d*\u2020\u2021\u00a7\u00b6\u2217]+")


@dataclass(frozen=True, slots=True)
class MetadataSettings:
    """Options of :class:`MetadataExtractor` (``params.parser.metadata`` in config)."""

    prefer_info_title: bool = False
    use_creation_date_year: bool = True
    min_title_chars: int = 4
    junk_title_markers: tuple[str, ...] = ("untitled", "microsoft word", ".doc", ".tex", ".pdf")
    junk_author_markers: tuple[str, ...] = ("anonymous", "unknown", "author")
    venue_patterns: tuple[str, ...] = (
        r"(?:In\s+)?Proceedings of (?:the\s+)?[^\n.;]+",
        r"(?:Published in|Appeared in|Accepted (?:at|to))\s+[^\n.;]+",
    )
    venue_acronyms: tuple[str, ...] = (
        "ACL",
        "EMNLP",
        "NAACL",
        "EACL",
        "COLING",
        "TACL",
        "SIGIR",
        "CIKM",
        "WSDM",
        "ECIR",
        "KDD",
        "WWW",
        "NeurIPS",
        "ICML",
        "ICLR",
        "AAAI",
        "IJCAI",
        "LREC",
        "CoNLL",
    )
    keyword_pattern: str = r"(?:Keywords|Key words|Index Terms)\s*[:.\u2014\u2013-]\s*(.+)"
    abstract_prefix_pattern: str = r"^abstract\s*[:.\u2014\u2013-]?\s*"
    min_year: int = 1900
    max_year: int = 2100
    language_min_tokens: int = 20
    language_min_ratio: float = 0.05

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> MetadataSettings:
        """Build settings from a config mapping, rejecting unknown keys.

        Raises:
            ConfigurationError: On unknown keys or values of the wrong type.
        """
        return settings_from_mapping(cls, mapping, "parser.metadata")


def split_authors(text: str) -> tuple[str, ...]:
    """Split an author line into names, dropping footnote markers and empty parts."""
    normalized = re.sub(r"\s+(?:and|&)\s+", ",", text)
    names: list[str] = []
    for part in re.split(r"[,;]", normalized):
        name = " ".join(_AUTHOR_MARKERS.sub(" ", part).split())
        if (
            len(name) >= _MIN_AUTHOR_CHARS
            and any(ch.isalpha() for ch in name)
            and name not in names
        ):
            names.append(name)
    return tuple(names)


def _split_keywords(text: str) -> tuple[str, ...]:
    """Split a keyword list on commas, semicolons and bullets."""
    parts = re.split(r"[,;·•]", text)
    keywords: list[str] = []
    for part in parts:
        keyword = " ".join(part.split()).strip(" .")
        if keyword and keyword not in keywords:
            keywords.append(keyword)
    return tuple(keywords)


class MetadataExtractor:
    """Extracts :class:`DocumentMetadata` from layout output and the PDF info dictionary.

    Args:
        settings: Extraction options.

    Raises:
        ConfigurationError: If a configured pattern is not a valid regular expression.
    """

    def __init__(self, settings: MetadataSettings | None = None) -> None:
        self._s = settings or MetadataSettings()
        try:
            self._venues = [re.compile(p, re.IGNORECASE) for p in self._s.venue_patterns]
            self._keywords = re.compile(self._s.keyword_pattern, re.IGNORECASE)
            self._abstract_prefix = re.compile(self._s.abstract_prefix_pattern, re.IGNORECASE)
            acronyms = "|".join(re.escape(a) for a in self._s.venue_acronyms)
            self._acronym = re.compile(rf"\b({acronyms})\b(?:\s*['\u2019]?\s*(\d{{4}}|\d{{2}})\b)?")
        except re.error as exc:
            raise ConfigurationError(f"Invalid parser.metadata pattern: {exc}") from exc

    def extract(
        self, info: Mapping[str, str], layout: LayoutResult, full_text: str
    ) -> DocumentMetadata:
        """Return the metadata of one document."""
        info_lc = {key.lower(): value for key, value in info.items()}
        front = layout.front_text
        venue = self._venue(front)
        return DocumentMetadata(
            title=self._title(info_lc.get("title"), layout.title),
            authors=self._authors(info_lc.get("author"), layout.authors),
            year=self._year(front, venue, info_lc.get("creationdate")),
            venue=venue,
            doi=self._doi(info_lc, front),
            keywords=self._keywords_of(info_lc.get("keywords"), front),
            abstract=self._abstract(layout),
            language=self._language(full_text),
        )

    def _title(self, info_title: str | None, layout_title: str | None) -> str | None:
        """Choose between the info-dictionary title and the typeset title."""
        usable_info = info_title if info_title and self._plausible_title(info_title) else None
        if self._s.prefer_info_title:
            return usable_info or layout_title
        return layout_title or usable_info

    def _plausible_title(self, title: str) -> bool:
        """Reject placeholder titles written by authoring tools."""
        lowered = title.lower()
        return len(title.strip()) >= self._s.min_title_chars and not any(
            marker in lowered for marker in self._s.junk_title_markers
        )

    def _authors(self, info_author: str | None, layout_authors: str | None) -> tuple[str, ...]:
        """Prefer the info-dictionary authors; fall back to the typeset author line."""
        for source in (info_author, layout_authors):
            names = tuple(
                name
                for name in split_authors(source or "")
                if not any(m in name.lower() for m in self._s.junk_author_markers)
            )
            if names:
                return names
        return ()

    def _valid_year(self, year: int) -> int | None:
        """Return ``year`` if plausible, else ``None``."""
        return year if self._s.min_year <= year <= self._s.max_year else None

    def _year(self, front: str, venue: str | None, creation_date: str | None) -> int | None:
        """Copyright notice → venue string → arXiv id → PDF creation date."""
        candidates: list[int] = []
        if match := _COPYRIGHT_YEAR.search(front):
            candidates.append(int(match.group(1)))
        if venue and (match := _YEAR.search(venue)):
            candidates.append(int(match.group(1)))
        if match := _ARXIV.search(front):
            candidates.append(2000 + int(match.group(1)))
        if (
            self._s.use_creation_date_year
            and creation_date
            and (match := _PDF_DATE_YEAR.match(creation_date))
        ):
            candidates.append(int(match.group(1)))
        for candidate in candidates:
            if (year := self._valid_year(candidate)) is not None:
                return year
        return None

    def _venue(self, front: str) -> str | None:
        """Find a venue statement, a known venue acronym (with year) or arXiv."""
        for pattern in self._venues:
            if match := pattern.search(front):
                return re.sub(r"^in\s+", "", " ".join(match.group(0).split()), flags=re.IGNORECASE)
        if match := self._acronym.search(front):
            acronym, year = match.group(1), match.group(2)
            if year and len(year) == _SHORT_YEAR_DIGITS:
                year = f"20{year}"
            return f"{acronym} {year}" if year else acronym
        if _ARXIV.search(front):
            return "arXiv"
        return None

    @staticmethod
    def _doi(info_lc: Mapping[str, str], front: str) -> str | None:
        """Find a DOI in the info dictionary or the front matter."""
        sources = [info_lc.get("doi", ""), info_lc.get("subject", ""), front]
        for source in sources:
            if match := _DOI.search(source):
                return match.group(1).rstrip(".)]")
        return None

    def _keywords_of(self, info_keywords: str | None, front: str) -> tuple[str, ...]:
        """Keywords from the info dictionary, else from a ``Keywords:`` line."""
        if info_keywords:
            keywords = _split_keywords(info_keywords)
            if keywords:
                return keywords
        if match := self._keywords.search(front):
            return _split_keywords(match.group(1))
        return ()

    def _abstract(self, layout: LayoutResult) -> str | None:
        """Join abstract blocks, removing an inline ``Abstract:`` prefix."""
        parts = [
            self._abstract_prefix.sub("", text).strip()
            for text in layout.text_of(ElementType.ABSTRACT)
        ]
        abstract = " ".join(part for part in parts if part)
        return abstract or None

    def _language(self, text: str) -> str | None:
        """Detect the language of the body text."""
        code = detect_language(
            text,
            min_tokens=self._s.language_min_tokens,
            min_ratio=self._s.language_min_ratio,
        )
        return None if code == UNDETERMINED else code
