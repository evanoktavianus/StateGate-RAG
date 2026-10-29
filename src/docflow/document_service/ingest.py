"""Ingest: a document's text becomes the passages and vectors the search index holds.

    parse    a PDF's text layer (`pdftotext`, no OCR) or plain text, as pages
    chunk    passages of at most 1,200 characters, 150 overlapping, page by page; no heading analysis, no summaries
    embed    the caller supplies the embedding function; a passage after the first can carry the document's
             breadcrumb (its opening lines), which is searched and embedded with it
    write    chunks.jsonl and embeddings.f32, the layout `Corpus.load` reads

Splitting is on the largest boundary that works (blank line, line, sentence, word, then raw characters), packs
neighbouring pieces together up to the limit, and starts each chunk after the first with up to `overlap`
characters from the end of the one before it, cut at a word boundary. The same for every document.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

SEPARATORS = ("\n\n", "\n", ". ", " ")


def split(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    overlap = min(overlap, max_chars // 2)
    pieces = _pieces(text.strip(), max_chars, SEPARATORS)
    return _with_overlap(_pack(pieces, max_chars - overlap), overlap)


def _pieces(text: str, limit: int, seps: tuple[str, ...]) -> list[str]:
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    if not seps:
        return [text[i : i + limit] for i in range(0, len(text), limit)]
    parts = _split_keeping(text, seps[0])
    if len(parts) == 1:
        return _pieces(text, limit, seps[1:])
    return [p for part in parts for p in _pieces(part, limit, seps[1:])]


def _split_keeping(text: str, sep: str) -> list[str]:
    """Pieces with the separator kept on the piece it ends, so joining them loses nothing."""
    parts = text.split(sep)
    return [p + sep for p in parts[:-1] if p + sep] + ([parts[-1]] if parts[-1] else [])


def _pack(pieces: list[str], limit: int) -> list[str]:
    out: list[str] = []
    for piece in pieces:
        if out and len(out[-1]) + len(piece) <= limit:
            out[-1] += piece
        else:
            out.append(piece)
    return [c.strip() for c in out if c.strip()]


def _with_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap == 0 or len(chunks) < 2:
        return chunks
    return [chunks[0]] + [_tail(prev, overlap) + cur for prev, cur in zip(chunks, chunks[1:])]


def _tail(text: str, overlap: int) -> str:
    if len(text) <= overlap:
        return text + " "
    cut = text[-overlap:]
    _, sep, whole = cut.partition(" ")
    return whole + " " if sep else ""


# ── Parsing ────────────────────────────────────────────────────────────


def parse_pdf(path: str | Path) -> list[str]:
    """The pages of a PDF's text layer, empty pages dropped (one empty page if there is no text)."""
    text = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True).stdout
    pages = [p.strip() for p in text.split("\f")]
    return [p for p in pages if p] or [""]


def canary(doc: str, version: int) -> str:
    """The reference line added to the end of every version, to tell its text from any other in an answer."""
    return f"Reference code CANARY-{doc}-v{version}."


def breadcrumb(text: str, limit: int = 300) -> str:
    """What names the document: its opening lines, up to `limit` characters. Every passage carries it, so a passage
    in the middle of a document still says which document it belongs to (searched as words and embedded with the passage)."""
    out, size = [], 0
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        if size + len(line) > limit and out:
            break
        out.append(line[:limit]); size += len(line)
    return " / ".join(out)


# ── Passages ───────────────────────────────────────────────────────────
# A record is {doc, version, seq, pages, breadcrumb, content}; `write_index` adds the row.


def pdf_records(doc: str, version: int, pdf: str | Path) -> list[dict]:
    """A PDF version: its pages, the canary line after the last one, each page split on its own."""
    pages = parse_pdf(pdf)
    pages[-1] = pages[-1] + "\n\n" + canary(doc, version)
    chunks = [(i, c) for i, page in enumerate(pages) for c in split(page)]
    return [{"doc": doc, "version": version, "seq": seq, "pages": [page, page], "breadcrumb": "", "content": text}
            for seq, (page, text) in enumerate(chunks)]


def text_records(doc: str, version: int, text: str) -> list[dict]:
    """A plain-text version: split as a whole, every passage carrying the document's breadcrumb."""
    crumb = breadcrumb(text)
    return [{"doc": doc, "version": version, "seq": seq, "pages": [0, 0], "breadcrumb": crumb, "content": content}
            for seq, content in enumerate(split(text))]


def embedding_text(record: dict) -> str:
    """What is embedded for a passage: its content, behind the breadcrumb for every passage after the first (the first
    already starts with it)."""
    if record["seq"] == 0 or not record["breadcrumb"]:
        return record["content"]
    return record["breadcrumb"] + "\n\n" + record["content"]


def write_index(out: str | Path, records: list[dict], vectors: list[bytes]) -> None:
    """chunks.jsonl (one record per line, with its row) and embeddings.f32 (the vectors, in the same order)."""
    out = Path(out)
    with open(out / "chunks.jsonl", "w") as cf, open(out / "embeddings.f32", "wb") as ef:
        for row, (record, vector) in enumerate(zip(records, vectors)):
            ef.write(vector)
            cf.write(json.dumps({**record, "row": row}, ensure_ascii=False) + "\n")
