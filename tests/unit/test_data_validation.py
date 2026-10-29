"""Tests for validation issues, reports, raw-record schemas and the dataset validator."""

from __future__ import annotations

import pytest

from ips_r.config.schema import ValidationConfig
from ips_r.data.issues import IssueCode, IssueCollector, Severity, ValidationIssue, ValidationReport
from ips_r.data.models import Document, DocumentMetadata, Question
from ips_r.data.validation import DatasetValidator, RecordSchema, SchemaTracker, has_errors

STR = frozenset({"string"})


def _issue(
    code: IssueCode = IssueCode.EMPTY_TEXT, severity: Severity = Severity.ERROR, rid: str = "r"
) -> ValidationIssue:
    return ValidationIssue(code, severity, f"{code.value} problem", rid)


def test_issue_validation_and_dict() -> None:
    issue = ValidationIssue("empty_text", "warning", "msg", "r", "f:1")  # type: ignore[arg-type]
    assert issue.code is IssueCode.EMPTY_TEXT
    assert issue.to_dict()["severity"] == "warning"
    with pytest.raises(ValueError):
        ValidationIssue(IssueCode.EMPTY_TEXT, Severity.ERROR, "")
    with pytest.raises(ValueError):
        ValidationIssue("nope", Severity.ERROR, "m")  # type: ignore[arg-type]


def test_collector_dedupes_counts_and_caps() -> None:
    collector = IssueCollector(max_examples_per_code=2)
    assert collector.add(_issue(rid="a"))
    assert not collector.add(_issue(rid="a"))
    collector.extend(
        [_issue(rid="b"), _issue(rid="c"), _issue(IssueCode.DUPLICATE_ID, Severity.WARNING)]
    )
    assert collector.total == len(collector) == 4
    assert collector.counts_by_code == {"duplicate_id": 1, "empty_text": 3}
    assert collector.counts_by_severity == {"error": 3, "warning": 1, "info": 0}
    assert len(collector.issues) == 3
    collector.clear()
    assert collector.total == 0
    with pytest.raises(ValueError):
        IssueCollector(-1)


def test_report_rendering() -> None:
    collector = IssueCollector(max_examples_per_code=1)
    collector.extend(
        [_issue(rid="a|b"), _issue(rid="c"), _issue(IssueCode.INVALID_METADATA, Severity.WARNING)]
    )
    report = ValidationReport.from_collector(
        "ds", collector, num_documents_checked=5, num_questions_checked=2
    )
    assert not report.is_valid
    assert report.truncated
    assert (report.num_errors, report.num_warnings) == (2, 1)
    data = report.to_dict()
    assert data["counts_by_code"]["empty_text"] == 2
    markdown = report.to_markdown()
    assert "FAILED" in markdown
    assert "a\\|b" in markdown
    clean = ValidationReport("ds", 1, 0).to_markdown()
    assert "PASSED" in clean
    assert "No issues found." in clean


def test_record_schema() -> None:
    schema = RecordSchema(
        "s", {"id": STR, "n": frozenset({"number"})}, {"opt": frozenset({"array"})}
    )
    assert schema.check({"id": "x", "n": 1}) == []
    codes = {i.code for i in schema.check({"n": "1", "opt": "x"})}
    assert codes == {IssueCode.MISSING_FIELD, IssueCode.MALFORMED_RECORD}
    assert schema.check([1])[0].message.endswith("got array")
    assert "got bool" in schema.check({"id": "x", "n": True})[0].message
    assert RecordSchema("s", {"id": STR}, {"o": STR}).check({"id": "x", "o": None}) == []
    assert RecordSchema("s", {"id": STR}).check({"id": None})[0].code is IssueCode.MISSING_FIELD
    assert "got object" in RecordSchema("s", {"id": STR}).check({"id": {}})[0].message
    assert "got set" in RecordSchema("s", {"id": STR}).check({"id": {1}})[0].message


def test_schema_tracker_reports_each_shape_once() -> None:
    tracker = SchemaTracker("t", ignore=("optional",))
    assert tracker.observe({"a": 1, "b": "x"}) == []
    assert tracker.observe({"a": 2, "b": "y", "optional": 1}) == []
    issues = tracker.observe({"a": "1", "c": 1}, location="f:3")
    assert len(issues) == 1
    message = issues[0].message
    assert (
        "missing ['b']" in message
        and "unexpected ['c']" in message
        and "type changed for ['a']" in message
    )
    assert tracker.observe({"a": "1", "c": 1}) == []
    assert tracker.observe_columns(2) != []


def test_validator_detects_content_problems() -> None:
    validator = DatasetValidator(ValidationConfig(max_year=2030))
    assert validator.check_document(Document(id="d1", text="fine")) == []
    duplicate = validator.check_document(Document(id="d1", text=" "))
    assert {i.code for i in duplicate} == {IssueCode.DUPLICATE_ID, IssueCode.EMPTY_TEXT}
    assert has_errors(duplicate)
    bad_meta = DocumentMetadata(
        title=" ",
        authors=(" ",),
        keywords=("",),
        year=1066,
        doi="doi:10/x",
        language="English",
        url="nope",
    )
    issues = validator.check_document(Document(id="d2", text="cafÃ© �", metadata=bad_meta))
    assert sum(i.code is IssueCode.INVALID_METADATA for i in issues) == 7
    assert sum(i.code is IssueCode.INVALID_ENCODING for i in issues) == 2
    assert not has_errors(issues)


def test_validator_questions_and_dangling_references() -> None:
    validator = DatasetValidator()
    validator.check_question(
        Question(id="q1", text="?", answers=(" ",), supporting_document_ids=("late", "ghost"))
    )
    issues = validator.check_question(Question(id="q1", text=" �"))
    assert {i.code for i in issues} == {IssueCode.DUPLICATE_ID, IssueCode.INVALID_ENCODING}
    validator.check_question(Question(id="q2", text=""))
    validator.check_document(Document(id="late", text="arrives after the question"))
    extra = ValidationIssue(IssueCode.CORRUPTED_PDF, Severity.ERROR, "bad pdf", "p")
    report = validator.report("ds", [extra])
    assert report.counts_by_code["dangling_reference"] == 1
    assert report.counts_by_code["corrupted_pdf"] == 1
    assert report.num_questions_checked == 3
    assert report.num_documents_checked == 1


def test_validator_can_disable_checks() -> None:
    validator = DatasetValidator(ValidationConfig(check_encoding=False, check_metadata=False))
    doc = Document(id="d", text="�", metadata=DocumentMetadata(year=1))
    assert validator.check_document(doc) == []
    assert validator.settings.check_encoding is False
    with pytest.raises(ValueError):
        DatasetValidator(ValidationConfig(min_year=2000, max_year=1900))
