"""PDF → canonical :class:`~ips_r.data.models.Document` parser.

Composes three replaceable stages:

``AbstractPDFBackend`` (extraction) → ``LayoutAnalyzer`` (structure) → ``MetadataExtractor``

Configuration (``params.parser`` of ``configs/dataset/academic_pdf.yaml``)::

    parser:
      backend: pdfplumber          # name in PDF_BACKENDS
      backend_options: {}          # passed to the backend
      layout: {...}                # LayoutSettings fields
      metadata: {...}              # MetadataSettings fields
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from ips_r.core.exceptions import ConfigurationError, PDFParseError, RegistryError
from ips_r.core.log import get_logger
from ips_r.data.models import Document, ElementDraft
from ips_r.data.pdf.backend import PDF_BACKENDS, AbstractPDFBackend
from ips_r.data.pdf.layout import LayoutAnalyzer, LayoutSettings
from ips_r.data.pdf.metadata import MetadataExtractor, MetadataSettings

logger = get_logger(__name__)

DEFAULT_BACKEND: Final = "pdfplumber"
_PARSER_KEYS: Final = frozenset({"backend", "backend_options", "layout", "metadata"})


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    """Validate that a config value is a mapping (``None`` → empty)."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


class PDFParser:
    """Parses academic PDFs into structured documents.

    Args:
        backend: Extraction backend.
        layout: Layout analysis settings.
        metadata: Metadata extraction settings.
    """

    def __init__(
        self,
        backend: AbstractPDFBackend,
        layout: LayoutSettings | None = None,
        metadata: MetadataSettings | None = None,
    ) -> None:
        self._backend = backend
        self._analyzer = LayoutAnalyzer(layout)
        self._extractor = MetadataExtractor(metadata)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> PDFParser:
        """Build a parser from the ``parser`` config mapping.

        Raises:
            ConfigurationError: On unknown keys, an unknown backend or invalid settings.
        """
        params = _mapping(config, "parser")
        unknown = sorted(set(params) - _PARSER_KEYS)
        if unknown:
            raise ConfigurationError(f"Unknown parser settings: {unknown}")
        backend_name = params.get("backend") or DEFAULT_BACKEND
        try:
            backend = PDF_BACKENDS.create(
                str(backend_name), _mapping(params.get("backend_options"), "parser.backend_options")
            )
        except RegistryError as exc:
            raise ConfigurationError(str(exc)) from exc
        return cls(
            backend,
            LayoutSettings.from_mapping(_mapping(params.get("layout"), "parser.layout")),
            MetadataSettings.from_mapping(_mapping(params.get("metadata"), "parser.metadata")),
        )

    def parse(self, path: Path, document_id: str) -> Document:
        """Parse ``path`` into a document with structural elements and metadata.

        Raises:
            PDFParseError: If the PDF is unreadable or contains no extractable text
                (e.g. a scanned image without an OCR layer).
        """
        raw = self._backend.read(path)
        layout = self._analyzer.analyze(raw)
        if not layout.blocks:
            raise PDFParseError(f"PDF {path} contains no extractable text (scanned image?)")
        drafts = [
            ElementDraft(
                type=block.type,
                text=block.text,
                page=block.page,
                section=block.section,
                level=block.level,
                rows=block.rows,
            )
            for block in layout.blocks
        ]
        body = "\n".join(block.text for block in layout.blocks)
        document = Document.from_elements(
            document_id,
            drafts,
            metadata=self._extractor.extract(raw.info, layout, body),
            source=str(path),
            num_pages=layout.num_pages,
            extra={"pdf_info": dict(raw.info), "sections": list(layout.sections)},
        )
        logger.debug(
            "pdf.parsed",
            path=str(path),
            pages=layout.num_pages,
            elements=len(document.elements),
            title=document.title,
        )
        return document
