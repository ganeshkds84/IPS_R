"""Integration tests: every adapter reads real files through the uniform interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ips_r.core.exceptions import ComponentError, ConfigurationError, RegistryError
from ips_r.data import DATASETS, AbstractDataset, IssueCode
from ips_r.data.adapters import AcademicPDFDataset
from tests import data_factories as factories

ADAPTERS = ("squad", "hotpotqa", "natural_questions", "ms_marco", "academic_pdf")


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    root = tmp_path_factory.mktemp("sources")
    nq_simplified, nq_original = factories.nq_files(root / "nq")
    pdf_dir = factories.pdf_collection(root / "pdfs")
    return {
        "squad": {"path": str(factories.squad_file(root))},
        "hotpotqa": {"path": str(factories.hotpot_file(root))},
        "natural_questions": {"path": str(nq_simplified)},
        "nq_original": {"path": str(nq_original)},
        "ms_marco": {k: str(v) for k, v in factories.msmarco_files(root / "msm").items()},
        "academic_pdf": {
            "directory": str(pdf_dir),
            "questions_path": str(pdf_dir / "questions.jsonl"),
        },
    }


def _codes(dataset: AbstractDataset) -> set[IssueCode]:
    return {issue.code for issue in dataset.ingestion_issues()}


@pytest.mark.parametrize("name", ADAPTERS)
def test_uniform_interface(name: str, sources: dict[str, dict[str, Any]]) -> None:
    dataset = DATASETS.create(name, sources[name], name=name)
    with dataset:
        documents = dataset.load_documents()
        questions = dataset.load_questions()
        assert documents and questions
        assert dataset.load_documents() is documents  # cached
        assert len(dataset.documents()) == len(documents)
        assert [q.id for q in dataset.queries()] == [r.query_id for r in dataset.references()]
        meta = dataset.describe()
        assert meta.name == name
        assert meta.sources
        assert meta.num_documents is None
        assert dataset.ingestion_issues()
    assert not dataset.is_ready


def test_squad(sources: dict[str, dict[str, Any]]) -> None:
    dataset = DATASETS.create("squad", {**sources["squad"], "split": "dev", "version": "2.0"})
    docs = {d.id: d for d in dataset.iter_documents()}
    assert set(docs) == {"Super_Bowl_50#0", "Super_Bowl_50#1", "article-1#0"}
    assert docs["Super_Bowl_50#0"].metadata.title == "Super Bowl 50"
    questions = {q.id: q for q in dataset.iter_questions()}
    assert set(questions) == {"q1", "q2", "q4", "q5"}
    assert questions["q1"].answers == ("Denver Broncos",)
    assert questions["q1"].metadata["answer_starts"] == [49, 49]
    assert questions["q2"].question_type == "unanswerable"
    assert questions["q4"].answers == ()
    assert _codes(dataset) >= {
        IssueCode.MISSING_FIELD,
        IssueCode.MALFORMED_RECORD,
        IssueCode.INCONSISTENT_SCHEMA,
    }
    assert dataset.describe().split == "dev"


def test_squad_rejects_wrong_top_level(tmp_path: Path) -> None:
    path = factories.write_json(tmp_path / "bad.json", [1, 2])
    dataset = DATASETS.create("squad", {"path": str(path)})
    assert dataset.load_documents() == []
    assert _codes(dataset) == {IssueCode.MALFORMED_RECORD}


def test_hotpotqa(sources: dict[str, dict[str, Any]]) -> None:
    dataset = DATASETS.create("hotpotqa", sources["hotpotqa"])
    docs = {d.id: d for d in dataset.iter_documents()}
    assert set(docs) == {"Arthur's Magazine", "First for Women"}
    magazine = docs["Arthur's Magazine"]
    spans = magazine.extra["sentence_spans"]
    assert [magazine.text[s:e] for s, e in spans] == [
        "Arthur's Magazine was a magazine.",
        " It began in 1844.",
    ]
    questions = {q.id: q for q in dataset.iter_questions()}
    assert set(questions) == {"h1", "h2"}
    assert questions["h1"].supporting_document_ids == ("Arthur's Magazine", "First for Women")
    assert questions["h1"].question_type == "comparison"
    assert questions["h2"].metadata["supporting_facts"] == [["Arthur's Magazine", 1]]
    assert _codes(dataset) >= {
        IssueCode.DUPLICATE_ID,
        IssueCode.MISSING_FIELD,
        IssueCode.MALFORMED_RECORD,
    }


def test_hotpotqa_rejects_wrong_top_level(tmp_path: Path) -> None:
    dataset = DATASETS.create(
        "hotpotqa", {"path": str(factories.write_json(tmp_path / "x.json", {"a": 1}))}
    )
    assert dataset.load_questions() == []
    broken = tmp_path / "broken.json"
    broken.write_text("{nope", encoding="utf-8")
    assert DATASETS.create("hotpotqa", {"path": str(broken)}).load_documents() == []


def test_natural_questions_simplified(sources: dict[str, dict[str, Any]]) -> None:
    dataset = DATASETS.create("natural_questions", sources["natural_questions"])
    docs = {d.id: d for d in dataset.iter_documents()}
    assert docs["101"].text == "Acme was founded by Jane Doe in 1990 ."
    assert docs["101"].metadata.url == "https://en.wikipedia.org/wiki/Acme"
    questions = {q.id: q for q in dataset.iter_questions()}
    assert questions["101"].answers == ("Jane Doe",)
    assert questions["102"].answers == ("yes",)
    assert (
        questions["101"].question_type,
        questions["102"].question_type,
        questions["104"].question_type,
    ) == (
        "short",
        "yes_no",
        "long",
    )
    assert _codes(dataset) >= {
        IssueCode.MISSING_FIELD,
        IssueCode.MALFORMED_RECORD,
        IssueCode.INCONSISTENT_SCHEMA,
    }


def test_natural_questions_original_layout_and_html(sources: dict[str, dict[str, Any]]) -> None:
    dataset = DATASETS.create("natural_questions", {**sources["nq_original"], "strip_html": False})
    (doc,) = dataset.iter_documents()
    assert doc.text == "<P> The tower is in Paris </P>"
    (question,) = dataset.iter_questions()
    assert question.answers == ("Paris",)
    assert IssueCode.MALFORMED_RECORD in _codes(dataset)


def test_natural_questions_unanswered_type(tmp_path: Path) -> None:
    path = factories.write_lines(
        tmp_path / "nq.jsonl",
        ['{"example_id": 1, "question_text": "q", "document_text": "a b", "annotations": []}'],
    )
    (question,) = DATASETS.create("natural_questions", {"path": str(path)}).iter_questions()
    assert question.question_type == "none"


def test_ms_marco(sources: dict[str, dict[str, Any]]) -> None:
    dataset = DATASETS.create("ms_marco", sources["ms_marco"])
    docs = {d.id: d for d in dataset.iter_documents()}
    assert set(docs) == {"0", "1", "3", "4"}
    assert "�" in docs["3"].text
    questions = {q.id: q for q in dataset.iter_questions()}
    assert set(questions) == {"10", "11"}
    assert questions["11"].supporting_document_ids == ("1", "9")
    assert _codes(dataset) >= {
        IssueCode.MALFORMED_RECORD,
        IssueCode.MISSING_FIELD,
        IssueCode.INVALID_ENCODING,
        IssueCode.INCONSISTENT_SCHEMA,
    }


def test_ms_marco_optional_files(sources: dict[str, dict[str, Any]]) -> None:
    params = sources["ms_marco"]
    collection_only = DATASETS.create("ms_marco", {"collection_path": params["collection_path"]})
    assert collection_only.load_questions() == []
    unjudged = DATASETS.create("ms_marco", {**params, "qrels_path": None})
    assert {q.id for q in unjudged.iter_questions()} == {"10", "11", "12"}
    keep_all = DATASETS.create("ms_marco", {**params, "only_judged_queries": False})
    assert {q.id for q in keep_all.iter_questions()} == {"10", "11", "12"}


def test_academic_pdf(sources: dict[str, dict[str, Any]]) -> None:
    dataset = DATASETS.create("academic_pdf", sources["academic_pdf"])
    assert isinstance(dataset, AcademicPDFDataset)
    assert [p.name for p in dataset.files()] == [
        "blank.pdf",
        "broken.pdf",
        "paper-b.pdf",
        "paper-a.pdf",
    ]
    docs = {d.id: d for d in dataset.iter_documents()}
    assert set(docs) == {"paper-a", "paper-b"}
    assert docs["paper-a"].num_pages == 2
    questions = {q.id: q for q in dataset.iter_questions()}
    assert set(questions) == {"p1", "2", "p4"}
    assert questions["p1"].supporting_document_ids == ("paper-a",)
    assert _codes(dataset) >= {
        IssueCode.CORRUPTED_PDF,
        IssueCode.MALFORMED_RECORD,
        IssueCode.MISSING_FIELD,
    }
    nested = DATASETS.create(
        "academic_pdf", {**sources["academic_pdf"], "id_strategy": "relative_path"}
    )
    assert isinstance(nested, AcademicPDFDataset)
    assert (
        nested.document_id(Path(sources["academic_pdf"]["directory"]) / "nested" / "paper-b.pdf")
        == "nested/paper-b"
    )


def test_limits_and_strict_mode(sources: dict[str, dict[str, Any]]) -> None:
    limited = DATASETS.create("squad", {**sources["squad"], "max_documents": 1, "max_questions": 2})
    assert len(limited.load_documents()) == 1
    assert len(limited.load_questions()) == 2
    strict = DATASETS.create("squad", {**sources["squad"], "strict": True})
    with pytest.raises(ComponentError, match="strict mode"):
        strict.load_documents()


@pytest.mark.parametrize(
    ("name", "params", "error"),
    [
        ("squad", {}, RegistryError),
        ("squad", {"path": "x.json", "max_documents": 0}, RegistryError),
        ("academic_pdf", {"directory": ".", "id_strategy": "hash"}, RegistryError),
        ("academic_pdf", {"directory": ".", "parser": {"layout": {"nope": 1}}}, RegistryError),
    ],
)
def test_invalid_configuration_fails_at_construction(
    name: str, params: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        DATASETS.create(name, params)


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("squad", {"path": "does/not/exist.json"}),
        ("academic_pdf", {"directory": "does/not/exist"}),
        ("ms_marco", {"collection_path": "missing.tsv"}),
    ],
)
def test_missing_sources_fail_at_setup(name: str, params: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        DATASETS.create(name, params).setup()


def test_missing_optional_ms_marco_files(
    sources: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
    for key in ("queries_path", "qrels_path"):
        params = {**sources["ms_marco"], key: str(tmp_path / "missing.tsv")}
        with pytest.raises(ConfigurationError, match=key):
            DATASETS.create("ms_marco", params).setup()
    params = {**sources["academic_pdf"], "questions_path": str(tmp_path / "missing.jsonl")}
    with pytest.raises(ConfigurationError, match="questions_path"):
        DATASETS.create("academic_pdf", params).setup()
