"""Fixed patterns of structural analysis (labels, markers, bibliographic fields)."""

from __future__ import annotations

import re
from typing import Final

RANGE_BOUNDS: Final = 2  # numeric citation range "3-5"
MAX_CITATION_RANGE: Final = 50  # longer ranges are treated as noise
FIGURE_LABEL: Final = re.compile(r"^(fig(?:ure)?\.?)\s*(\d+)", re.IGNORECASE)
TABLE_LABEL: Final = re.compile(r"^(tab(?:le)?\.?)\s*(\d+)", re.IGNORECASE)
EQUATION_NUMBER: Final = re.compile(r"\((\d+(?:\.\d+)?[a-z]?)\)\s*$")
REFERENCE_MARKER: Final = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)\.\s)")
FOOTNOTE_MARKER: Final = re.compile(r"^\s*(\d{1,2}|[*\u2020\u2021\u00a7\u00b6])\s*(?=\S)")
YEAR: Final = re.compile(r"\b((?:19|20)\d{2})[a-z]?\b")
DOI: Final = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>,;]+)", re.IGNORECASE)
SURNAME: Final = re.compile(r"\b([A-Z][A-Za-z'\-]+)\b(?!\.)")
APPENDIX_LETTER: Final = re.compile(r"^appendix\s+([A-Z])\b", re.IGNORECASE)
_AUTHOR_SEGMENT_END: Final = re.compile(r",|\s(?:and|&)\s|(?<=[A-Za-z]{2})\.(?:\s|$)")
_CAPITALISED: Final = re.compile(r"[A-Z][A-Za-z'\-]+")
CROSS_KIND: Final = (
    ("fig", "figure"),
    ("tab", "table"),
    ("sec", "section"),
    ("eq", "equation"),
    ("app", "appendix"),
)
"""Cross-reference word prefix → target kind."""


def first_author(reference_body: str) -> str | None:
    """Surname of the first author of a reference entry (marker already removed).

    The first author segment ends at a comma, ``and``/``&`` or a period closing a word
    of two or more letters; its last capitalised word of two or more letters is the
    surname (``"B. Author. Title"`` \u2192 ``Author``, ``"Smith, J."`` \u2192 ``Smith``).
    """
    end = _AUTHOR_SEGMENT_END.search(reference_body)
    segment = reference_body[: end.start() if end else len(reference_body)]
    names = _CAPITALISED.findall(segment)
    return names[-1] if names else None
