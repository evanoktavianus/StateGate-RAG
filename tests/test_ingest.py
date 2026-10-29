"""Ingest: parse, chunk, embed text, write. Expected values follow the rules in the module."""

import array
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from docflow.document_service import ingest
from docflow.stores.search_index import DIMS, Corpus


def test_the_canary_line():
    assert ingest.canary("doc-a", 2) == "Reference code CANARY-doc-a-v2."


def test_the_breadcrumb_is_the_opening_lines_up_to_the_limit():
    assert ingest.breadcrumb("Line one\n\n  Line two  \nLine three") == "Line one / Line two / Line three"
    # three lines of 10 characters, limit 25: the second brings the size to 20, the third would make 30, so it stops
    assert ingest.breadcrumb("a" * 10 + "\n" + "b" * 10 + "\n" + "c" * 10, limit=25) == "a" * 10 + " / " + "b" * 10
    assert ingest.breadcrumb("x" * 400) == "x" * 300  # a first line that is too long is cut, not dropped
    assert ingest.breadcrumb("") == ""


def test_a_short_text_is_one_passage_with_its_breadcrumb():
    assert ingest.text_records("d", 2, "Alpha beta.\n\nGamma delta.") == [
        {"doc": "d", "version": 2, "seq": 0, "pages": [0, 0], "breadcrumb": "Alpha beta. / Gamma delta.", "content": "Alpha beta.\n\nGamma delta."}]


def test_a_long_text_is_split_at_paragraphs_and_every_passage_carries_the_breadcrumb():
    # three paragraphs of 700 characters: two do not fit in 1,050 (1,200 less the 150 overlap), so one passage each;
    # a paragraph with no space has nothing to overlap with
    text = "\n\n".join(c * 700 for c in "abc")
    records = ingest.text_records("d", 1, text)
    assert [r["content"] for r in records] == ["a" * 700, "b" * 700, "c" * 700]
    assert [r["seq"] for r in records] == [0, 1, 2]
    assert {r["breadcrumb"] for r in records} == {"a" * 300}  # the first line, cut to 300


def test_what_is_embedded_puts_the_breadcrumb_before_every_passage_after_the_first():
    first, second, third = ingest.text_records("d", 1, "\n\n".join(c * 700 for c in "abc"))
    assert ingest.embedding_text(first) == "a" * 700  # the first passage starts with it already
    assert ingest.embedding_text(second) == "a" * 300 + "\n\n" + "b" * 700
    assert ingest.embedding_text({"seq": 3, "breadcrumb": "", "content": "just this"}) == "just this"  # no breadcrumb, nothing to add


def fake_pdftotext(monkeypatch, stdout):
    monkeypatch.setattr(ingest.subprocess, "run", lambda cmd, **kw: SimpleNamespace(stdout=stdout) if cmd[0] == "pdftotext" else pytest.fail(cmd))


def test_the_pages_of_a_pdf_drop_empty_pages(monkeypatch):
    fake_pdftotext(monkeypatch, " page one \f\npage two\f\f")  # pdftotext ends each page with a form feed
    assert ingest.parse_pdf("x.pdf") == ["page one", "page two"]
    fake_pdftotext(monkeypatch, "")
    assert ingest.parse_pdf("x.pdf") == [""]  # a PDF with no text still has one (empty) page


def test_a_pdf_version_is_chunked_page_by_page_with_the_canary_on_the_last(monkeypatch):
    fake_pdftotext(monkeypatch, "First page.\fSecond page.\f")
    assert ingest.pdf_records("doc-a", 1, "x.pdf") == [
        {"doc": "doc-a", "version": 1, "seq": 0, "pages": [0, 0], "breadcrumb": "", "content": "First page."},
        {"doc": "doc-a", "version": 1, "seq": 1, "pages": [1, 1], "breadcrumb": "", "content": "Second page.\n\nReference code CANARY-doc-a-v1."}]


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext is not installed")
def test_a_real_pdf_text_layer_is_read(tmp_path):
    stream = b"BT /F1 12 Tf 20 50 Td (Hello pdf text) Tj ET"
    pdf = (b"%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
           b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 100]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
           b"4 0 obj<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream endobj\n"
           b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\ntrailer<</Root 1 0 R/Size 6>>\n%%EOF\n")
    path = tmp_path / "t.pdf"
    path.write_bytes(pdf)
    assert ingest.parse_pdf(path) == ["Hello pdf text"]


def test_the_index_is_written_in_the_layout_the_search_index_reads(tmp_path):
    records = ingest.text_records("d", 1, "\n\n".join(c * 700 for c in "abc"))
    vectors = [array.array("f", [float(i + 1)] * DIMS).tobytes() for i in range(3)]
    ingest.write_index(tmp_path, records, vectors)
    lines = [json.loads(l) for l in (tmp_path / "chunks.jsonl").read_text().splitlines()]
    assert [list(l) for l in lines] == [["doc", "version", "seq", "pages", "breadcrumb", "content", "row"]] * 3  # this order, row last
    assert [l["row"] for l in lines] == [0, 1, 2]
    assert (tmp_path / "embeddings.f32").read_bytes() == b"".join(vectors)
    corpus = Corpus.load(tmp_path)
    assert [(p.doc, p.version, p.seq, p.breadcrumb[:3], p.content[:1]) for p in corpus.passages] == [("d", 1, i, "aaa", c) for i, c in enumerate("abc")]
    assert corpus.vector_of(corpus.passages[2]) == [3.0] * DIMS  # the vector of row 2 is the third one written
