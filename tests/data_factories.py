"""Builders for small, hermetic dataset fixtures (no committed binaries, no network).

Every builder writes into a caller-supplied directory and returns the created path.
Each fixture contains a few valid records plus deliberately broken ones, so tests can
check both the happy path and issue reporting.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

INTRO = (
    "Retrieval augmented generation grounds the answers of language models in the documents "
    "that a retriever returns for the query. "
)


def write_json(path: Path, payload: Any) -> Path:
    """Write ``payload`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_lines(path: Path, lines: list[str], *, gz: bool = False) -> Path:
    """Write text lines (optionally gzip-compressed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ("\n".join(lines) + "\n").encode("utf-8")
    path.write_bytes(gzip.compress(data, mtime=0) if gz else data)
    return path


def squad_file(directory: Path) -> Path:
    """SQuAD v2-style file: 2 valid paragraphs, 1 malformed paragraph, mixed question shapes."""
    return write_json(
        directory / "squad.json",
        {
            "version": "v2.0",
            "data": [
                {
                    "title": "Super_Bowl_50",
                    "paragraphs": [
                        {
                            "context": "Super Bowl 50 was an American football game. "
                            "The Denver Broncos won the title.",
                            "qas": [
                                {
                                    "id": "q1",
                                    "question": "Who won Super Bowl 50?",
                                    "answers": [
                                        {"text": "Denver Broncos", "answer_start": 49},
                                        {"text": "Denver Broncos", "answer_start": 49},
                                    ],
                                    "is_impossible": False,
                                },
                                {
                                    "id": "q2",
                                    "question": "Who lost Super Bowl 51?",
                                    "answers": [],
                                    "is_impossible": True,
                                    "plausible_answers": [{"text": "Broncos", "answer_start": 53}],
                                },
                                {"id": "q3", "answers": []},
                                {"id": "q4", "question": "What sport?", "answers": [{"text": 7}]},
                                {"id": "", "question": "Blank id?", "answers": []},
                                {"id": "q5", "question": "What was it?", "answers": []},
                            ],
                        },
                        {"context": "   ", "qas": []},
                        {"qas": []},
                    ],
                },
                {"paragraphs": [{"context": "Untitled article text.", "qas": []}]},
                {"title": "Broken"},
            ],
        },
    )


def hotpot_file(directory: Path) -> Path:
    """HotpotQA-style file with shared, conflicting and malformed context paragraphs."""
    return write_json(
        directory / "hotpot.json",
        [
            {
                "_id": "h1",
                "question": "Which magazine was started first?",
                "answer": "Arthur's Magazine",
                "type": "comparison",
                "level": "medium",
                "supporting_facts": [["Arthur's Magazine", 0], ["First for Women", 0], ["bad"]],
                "context": [
                    [
                        "Arthur's Magazine",
                        ["Arthur's Magazine was a magazine.", " It began in 1844."],
                    ],
                    ["First for Women", ["First for Women is a magazine."]],
                    ["broken entry"],
                ],
            },
            {
                "_id": "h2",
                "question": "When did Arthur's Magazine start?",
                "answer": "1844",
                "type": "bridge",
                "level": "easy",
                "supporting_facts": [["Arthur's Magazine", 1]],
                "context": [
                    [
                        "Arthur's Magazine",
                        ["Arthur's Magazine was a magazine.", " It began in 1844."],
                    ],
                    ["First for Women", ["A different text for the same title."]],
                ],
            },
            {"_id": "h3", "question": "No context?"},
            {"_id": "", "question": "Blank id", "context": []},
            "not an object",
        ],
    )


def nq_files(directory: Path) -> tuple[Path, Path]:
    """Natural Questions: a gzipped simplified file and a plain original-layout file."""
    simplified = [
        {
            "example_id": 101,
            "question_text": "who founded the company",
            "document_title": "Acme",
            "document_url": "https://en.wikipedia.org/wiki/Acme",
            "document_text": "<P> Acme was founded by Jane Doe in 1990 . </P>",
            "annotations": [
                {
                    "long_answer": {"start_token": 0, "end_token": 11},
                    "short_answers": [{"start_token": 5, "end_token": 7}],
                    "yes_no_answer": "NONE",
                },
                {
                    "long_answer": {"start_token": -1},
                    "short_answers": [{"start_token": 50, "end_token": 60}],
                    "yes_no_answer": "NONE",
                },
                "not an annotation",
            ],
            "long_answer_candidates": [],
        },
        {
            "example_id": 102,
            "question_text": "is acme a company",
            "document_text": "<P> Acme is a company . </P>",
            "annotations": [
                {
                    "long_answer": {"start_token": 0, "end_token": 7},
                    "short_answers": [],
                    "yes_no_answer": "YES",
                }
            ],
        },
        {"example_id": 103, "question_text": "no document", "annotations": []},
        {
            "example_id": 104,
            "question_text": "what is it",
            "document_text": "<P> Nothing here . </P>",
            "annotations": [
                {
                    "long_answer": {"start_token": 0, "end_token": 5},
                    "short_answers": [],
                    "yes_no_answer": "NONE",
                }
            ],
        },
    ]
    lines = [json.dumps(r) for r in simplified] + ["{not json"]
    original = [
        {
            "example_id": "201",
            "question_text": "where is the tower",
            "document_title": "Tower",
            "document_tokens": [
                {"token": "<P>", "html_token": True},
                {"token": "The", "html_token": False},
                {"token": "tower", "html_token": False},
                {"token": "is", "html_token": False},
                {"token": "in", "html_token": False},
                {"token": "Paris", "html_token": False},
                {"token": "</P>", "html_token": True},
            ],
            "annotations": [
                {"short_answers": [{"start_token": 5, "end_token": 6}], "yes_no_answer": "NONE"}
            ],
        },
        {
            "example_id": "202",
            "question_text": "bad tokens",
            "document_tokens": [1, 2],
            "annotations": [],
        },
    ]
    return (
        write_lines(directory / "nq.jsonl.gz", lines, gz=True),
        write_lines(directory / "nq_original.jsonl", [json.dumps(r) for r in original]),
    )


def msmarco_files(directory: Path) -> dict[str, Path]:
    """MS MARCO collection/queries/qrels with malformed rows and an invalid byte sequence."""
    collection = directory / "collection.tsv"
    collection.parent.mkdir(parents=True, exist_ok=True)
    collection.write_bytes(
        b"0\tThe Manhattan Project produced the first atomic bombs.\n"
        b"1\tIt was led by the United States.\n"
        b"2\tthree\tcolumns\n"
        b"\tmissing id\n"
        b"3\tCaf\xe9 latin-1 byte.\n"
        b"4\tThe Manhattan Project produced the first atomic bombs.\n"
    )
    return {
        "collection_path": collection,
        "queries_path": write_lines(
            directory / "queries.tsv",
            ["10\twhat was the manhattan project", "11\twho led it", "12\tunjudged query", "13"],
        ),
        "qrels_path": write_lines(
            directory / "qrels.tsv",
            ["10\t0\t0\t1", "11\t0\t1\t1", "11\t0\t4\t0", "11\t9", "10\t0\t1\tx", "12"],
        ),
    }


def _styles() -> dict[str, ParagraphStyle]:
    sheet = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=sheet["Title"], fontSize=20),
        "heading": ParagraphStyle("h", parent=sheet["Heading1"], fontSize=14),
        "subheading": ParagraphStyle("h2", parent=sheet["Heading2"], fontSize=12),
        "body": ParagraphStyle("b", parent=sheet["BodyText"]),
    }


def paper_pdf(path: Path, *, info: bool = True) -> Path:
    """A two-page paper with title, authors, abstract, headings, caption, table and references."""
    path.parent.mkdir(parents=True, exist_ok=True)
    s = _styles()
    kwargs: dict[str, Any] = (
        {
            "title": "Intent Preservation in RAG",
            "author": "Ada Lovelace; Alan Turing",
            "keywords": "rag, retrieval",
        }
        if info
        else {"title": "Microsoft Word - draft.docx"}
    )
    doc = SimpleDocTemplate(str(path), pagesize=A4, **kwargs)
    story = [
        Paragraph("Intent Preservation in RAG", s["title"]),
        Paragraph("Ada Lovelace, Alan Turing", s["body"]),
        Paragraph("University of Examples, Department of Computing", s["body"]),
        Paragraph("Abstract", s["heading"]),
        Paragraph(
            "We study query reformulation for retrieval. The DOI of this work is "
            "10.1234/ipsr.2024.001 and it appears in Proceedings of the 47th SIGIR Conference 2024.",
            s["body"],
        ),
        Paragraph("Keywords: retrieval; query reformulation; intent", s["body"]),
        Paragraph("1 Introduction", s["heading"]),
        Paragraph(INTRO * 6, s["body"]),
        Paragraph(
            INTRO * 2 + "Prior work [1, 2] studied retrieval (Author, 2021). "
            "Figure 1 and Table 1 summarise the setting described in Section 1.1.",
            s["body"],
        ),
        Paragraph("Figure 1: Overview of the system.", s["body"]),
        Paragraph("1.1 Problem Setting", s["subheading"]),
        Paragraph("The problem is defined over queries and documents. " * 4, s["body"]),
        Paragraph("score(q, d) = w * sim(q, d) + b (1)", s["body"]),
        Paragraph("We optimise Eq. (1) on the training data, see Appendix A.", s["body"]),
        Table(
            [["Model", "Score"], ["BM25", "0.41"], ["Dense", "0.52"]],
            style=TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]),
        ),
        Paragraph("Table 1: Results on the dev set.", s["body"]),
        PageBreak(),
        Paragraph("References", s["heading"]),
        Paragraph("[1] A. Author. A paper about retrieval. In Proc. ACL, 2020.", s["body"]),
        Paragraph("[2] B. Author. Another paper on generation. 2021.", s["body"]),
        Paragraph("Appendix A Proofs", s["heading"]),
        Paragraph("A.1 Lemma Details", s["subheading"]),
        Paragraph("The lemma follows from the definitions in Section 1.", s["body"]),
    ]
    doc.build(story, onFirstPage=_footnote)
    return path


def _footnote(c: canvas.Canvas, _doc: Any) -> None:
    """Draw a small-font footnote at the bottom of the first page."""
    c.saveState()
    c.setFont("Helvetica", 7)
    c.drawString(72, 60, "1 Code and data are available from the authors.")
    c.restoreState()


def two_column_pdf(path: Path) -> Path:
    """A one-page, two-column PDF drawn with absolute positions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, 780, "A Two Column Study")
    c.setFont("Helvetica", 10)
    for i in range(6):
        c.drawString(72, 740 - 12 * i, f"left column line {i} continues")
        c.drawString(320, 740 - 12 * i, f"right column line {i} continues")
    c.drawString(290, 40, "1")
    c.showPage()
    c.save()
    return path


def blank_pdf(path: Path) -> Path:
    """A valid PDF page with no text (like a scanned image without OCR)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    c.showPage()
    c.save()
    return path


def corrupted_pdf(path: Path) -> Path:
    """A file with a ``.pdf`` suffix that is not a PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\nthis is not really a pdf\n%%EOF")
    return path


def pdf_collection(directory: Path) -> Path:
    """A directory with one paper, one nested paper, one blank and one corrupted PDF."""
    paper_pdf(directory / "paper-a.pdf")
    paper_pdf(directory / "nested" / "paper-b.pdf", info=False)
    blank_pdf(directory / "blank.pdf")
    corrupted_pdf(directory / "broken.pdf")
    write_lines(
        directory / "questions.jsonl",
        [
            json.dumps(
                {
                    "id": "p1",
                    "question": "What is studied?",
                    "answers": ["query reformulation"],
                    "document_ids": ["paper-a"],
                }
            ),
            json.dumps({"id": 2, "question": "Which paper?", "type": "lookup"}),
            json.dumps({"id": "p3", "question": "Bad answers", "answers": [1]}),
            json.dumps({"question": "No id"}),
            json.dumps({"id": "p4", "question": "Unknown doc", "document_ids": ["missing"]}),
        ],
    )
    return directory
