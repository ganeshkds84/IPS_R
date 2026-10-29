"""PDF extraction backends: turn a PDF file into positioned words, tables and info.

The rest of the PDF pipeline only sees the backend-neutral :class:`RawPDF` structure,
so the extraction engine can be swapped by registering another
:class:`AbstractPDFBackend` in :data:`PDF_BACKENDS`.
"""

from __future__ import annotations

import importlib
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ips_r.core.exceptions import PDFParseError
from ips_r.core.log import get_logger
from ips_r.core.registry import Registry

logger = get_logger(__name__)

BBox = tuple[float, float, float, float]
"""``(x0, top, x1, bottom)`` in PDF points, origin at the top-left of the page."""


@dataclass(frozen=True, slots=True)
class RawWord:
    """A word with its position and font properties."""

    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    fontname: str = ""

    @property
    def bold(self) -> bool:
        """Heuristic: the font name declares a bold or black weight."""
        name = self.fontname.lower()
        return "bold" in name or "black" in name or name.endswith((",b", "-b"))


@dataclass(frozen=True, slots=True)
class RawTable:
    """A detected table: bounding box and row-major cell text."""

    bbox: BBox
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class RawPage:
    """One page of extracted content (``number`` is 1-based)."""

    number: int
    width: float
    height: float
    words: tuple[RawWord, ...] = ()
    tables: tuple[RawTable, ...] = ()


@dataclass(frozen=True, slots=True)
class RawPDF:
    """Backend-neutral extraction result."""

    pages: tuple[RawPage, ...]
    info: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "info", MappingProxyType(dict(self.info)))


class AbstractPDFBackend(ABC):
    """Extracts a :class:`RawPDF` from a file.

    Args:
        options: Backend-specific options (from ``params.parser.backend_options``).
    """

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self._options = dict(options or {})

    @abstractmethod
    def read(self, path: Path) -> RawPDF:
        """Extract ``path``.

        Raises:
            PDFParseError: If the file is corrupted, encrypted or has no pages.
        """
        raise NotImplementedError


PDF_BACKENDS: Registry[AbstractPDFBackend] = Registry("pdf_backend", AbstractPDFBackend)
"""Registry of PDF extraction backends."""


def _info_value(value: Any) -> str | None:
    """Normalise a PDF info-dictionary value (bytes, PSLiteral, ...) to text."""
    if value is None:
        return None
    if isinstance(value, bytes):
        if value.startswith((b"\xfe\xff", b"\xff\xfe")):
            text = value.decode("utf-16", errors="replace")
        else:
            text = value.decode("latin-1")
    else:
        text = str(getattr(value, "name", value))
    text = text.strip()
    return text or None


@PDF_BACKENDS.register("pdfplumber")
class PdfPlumberBackend(AbstractPDFBackend):
    """Backend built on ``pdfplumber`` (pdfminer.six), imported lazily.

    Options:
        ``x_tolerance`` / ``y_tolerance``: word-grouping tolerances in points (default 3).
        ``extract_tables``: detect ruled tables (default ``True``).
    """

    def read(self, path: Path) -> RawPDF:
        """Extract words (with font size/name), tables and the info dictionary.

        Raises:
            PDFParseError: If the file cannot be opened or parsed, or has no pages.
        """
        pdfplumber = importlib.import_module("pdfplumber")
        x_tol = float(self._options.get("x_tolerance", 3))
        y_tol = float(self._options.get("y_tolerance", 3))
        extract_tables = bool(self._options.get("extract_tables", True))
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with pdfplumber.open(path) as pdf:
                    info = {
                        str(k): v
                        for k, raw in (pdf.metadata or {}).items()
                        if (v := _info_value(raw)) is not None
                    }
                    pages = tuple(
                        self._read_page(page, x_tol, y_tol, extract_tables) for page in pdf.pages
                    )
        except PDFParseError:
            raise
        except Exception as exc:  # pdfminer raises many unrelated exception types
            raise PDFParseError(f"Cannot parse PDF {path}: {type(exc).__name__}: {exc}") from exc
        for warning in caught:
            logger.debug("pdf.backend.warning", path=str(path), warning=str(warning.message))
        if not pages:
            raise PDFParseError(f"PDF {path} has no pages")
        return RawPDF(pages=pages, info=info)

    @staticmethod
    def _read_page(page: Any, x_tol: float, y_tol: float, extract_tables: bool) -> RawPage:
        """Convert one pdfplumber page."""
        words = tuple(
            RawWord(
                text=str(w["text"]),
                x0=float(w["x0"]),
                x1=float(w["x1"]),
                top=float(w["top"]),
                bottom=float(w["bottom"]),
                size=float(w.get("size", 0.0)),
                fontname=str(w.get("fontname", "")),
            )
            for w in page.extract_words(
                x_tolerance=x_tol, y_tolerance=y_tol, extra_attrs=["size", "fontname"]
            )
        )
        tables: list[RawTable] = []
        if extract_tables:
            for table in page.find_tables():
                rows = tuple(
                    tuple(" ".join(str(cell or "").split()) for cell in row)
                    for row in table.extract()
                )
                if any(any(cell for cell in row) for row in rows):
                    x0, top, x1, bottom = (float(v) for v in table.bbox)
                    tables.append(RawTable(bbox=(x0, top, x1, bottom), rows=rows))
        return RawPage(
            number=int(page.page_number),
            width=float(page.width),
            height=float(page.height),
            words=words,
            tables=tuple(tables),
        )
