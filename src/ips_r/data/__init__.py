"""Dataset layer: canonical models, adapters, validation, PDF ingestion, chunking, statistics.

Importing this package registers every dataset adapter (``DATASETS``) and chunking
strategy (``CHUNKERS``).
"""

from ips_r.data import adapters as adapters
from ips_r.data.base import DATASETS, AbstractDataset
from ips_r.data.chunking import CHUNKERS, AbstractChunker
from ips_r.data.issues import IssueCode, Severity, ValidationIssue, ValidationReport
from ips_r.data.models import (
    Chunk,
    DatasetMetadata,
    Document,
    DocumentElement,
    DocumentMetadata,
    ElementDraft,
    ElementType,
    Question,
)
from ips_r.data.preparation import PreparationResult, prepare_dataset
from ips_r.data.statistics import DatasetStatistics, StatisticsAccumulator
from ips_r.data.validation import DatasetValidator

__all__ = [
    "CHUNKERS",
    "DATASETS",
    "AbstractChunker",
    "AbstractDataset",
    "Chunk",
    "DatasetMetadata",
    "DatasetStatistics",
    "DatasetValidator",
    "Document",
    "DocumentElement",
    "DocumentMetadata",
    "ElementDraft",
    "ElementType",
    "IssueCode",
    "PreparationResult",
    "Question",
    "Severity",
    "StatisticsAccumulator",
    "ValidationIssue",
    "ValidationReport",
    "prepare_dataset",
]
