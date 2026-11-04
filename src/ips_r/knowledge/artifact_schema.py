"""Fixed schemas of the knowledge artifact tables.

Tables: ``documents``, ``questions`` and one per knowledge object kind. Column kinds
(``string``, ``int``, ``float``, ``bool``, ``strings``, ``json``) are derived from the
record dataclasses' type hints; ``json`` columns are stored as canonical JSON strings
in Arrow / Parquet so the schema never depends on the data.
"""

from __future__ import annotations

import importlib
import typing
from collections.abc import Mapping
from typing import Any, Final

from ips_r.knowledge.hashing import canonical_json
from ips_r.knowledge.models import KNOWLEDGE_TYPES, field_types
from ips_r.knowledge.records import strip_optional

DOCUMENT_TABLE: Final = "documents"
QUESTION_TABLE: Final = "questions"
TABLES: Final = (DOCUMENT_TABLE, QUESTION_TABLE, *KNOWLEDGE_TYPES)


def pyarrow_module() -> Any:
    """Import pyarrow lazily (heavy; this module is the only place that imports it)."""
    return importlib.import_module("pyarrow")


def parquet_module() -> Any:
    """Import ``pyarrow.parquet`` lazily."""
    return importlib.import_module("pyarrow.parquet")


def _column_kind(hint: Any) -> str:
    """Map a field type hint to a column kind: string, int, float, bool, strings or json."""
    hint = strip_optional(hint)
    if hint is str or (isinstance(hint, type) and issubclass(hint, str)):
        return "string"
    if hint is bool:
        return "bool"
    if hint is int:
        return "int"
    if hint is float:
        return "float"
    if typing.get_origin(hint) is tuple and typing.get_args(hint) in ((str, Ellipsis),):
        return "strings"
    return "json"


_DOCUMENT_COLUMNS: Final[dict[str, str]] = {
    "id": "string",
    "source_hash": "string",
    "knowledge_hash": "string",
    "title": "string",
    "metadata": "string",
    "num_pages": "int",
    "source": "string",
    **{f"num_{name}": "int" for name in KNOWLEDGE_TYPES},
}
_QUESTION_COLUMNS: Final[dict[str, str]] = {
    "id": "string",
    "text": "string",
    "answers": "strings",
    "supporting_document_ids": "strings",
    "question_type": "string",
    "metadata": "json",
}


def table_columns(table: str) -> dict[str, str]:
    """Column name → column kind for an artifact table.

    Raises:
        KeyError: If ``table`` is unknown.
    """
    if table == DOCUMENT_TABLE:
        return dict(_DOCUMENT_COLUMNS)
    if table == QUESTION_TABLE:
        return dict(_QUESTION_COLUMNS)
    return {name: _column_kind(hint) for name, hint in field_types(KNOWLEDGE_TYPES[table]).items()}


def arrow_schema(table: str) -> Any:
    """Return the fixed ``pyarrow.Schema`` of an artifact table."""
    pa = pyarrow_module()
    types = {
        "string": pa.string(),
        "int": pa.int64(),
        "float": pa.float64(),
        "bool": pa.bool_(),
        "strings": pa.list_(pa.string()),
        "json": pa.string(),
    }
    return pa.schema([pa.field(name, types[kind]) for name, kind in table_columns(table).items()])


def arrow_row(record: Mapping[str, Any], columns: Mapping[str, str]) -> dict[str, Any]:
    """Encode JSON-kind columns of ``record`` as canonical JSON strings."""
    return {
        name: (
            canonical_json(record.get(name))
            if kind == "json" and record.get(name) is not None
            else record.get(name)
        )
        for name, kind in columns.items()
    }
