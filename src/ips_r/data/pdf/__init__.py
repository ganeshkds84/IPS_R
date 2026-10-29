"""Modular PDF ingestion: extraction backend → layout analysis → metadata extraction."""

from ips_r.data.pdf.backend import (
    PDF_BACKENDS,
    AbstractPDFBackend,
    PdfPlumberBackend,
    RawPage,
    RawPDF,
    RawTable,
    RawWord,
)
from ips_r.data.pdf.layout import LayoutAnalyzer, LayoutBlock, LayoutResult, LayoutSettings
from ips_r.data.pdf.metadata import MetadataExtractor, MetadataSettings, split_authors
from ips_r.data.pdf.parser import PDFParser

__all__ = [
    "PDF_BACKENDS",
    "AbstractPDFBackend",
    "LayoutAnalyzer",
    "LayoutBlock",
    "LayoutResult",
    "LayoutSettings",
    "MetadataExtractor",
    "MetadataSettings",
    "PDFParser",
    "PdfPlumberBackend",
    "RawPDF",
    "RawPage",
    "RawTable",
    "RawWord",
    "split_authors",
]
