# SysMeAIEngine - Private AI Model Engine
# Copyright 2026 Jose Rodriguez
#
# Developer: Jose Rodriguez <jrpcone@gmail.com>
# GitHub:    https://github.com/jorodriguezpr
# Site:      https://microrepair.net
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Document ingestion: extract → chunk → dataset-build."""
from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable


# ── extractors ──────────────────────────────────────────────────────────────
# A blank line between a fragment that doesn't end a sentence and one that
# starts lowercase is pdfminer layout-box splitting, not a real paragraph --
# measured at 59% (1082/1825) of all "\n\n" in the real aprocrifos-v2
# extraction, e.g. "Ellos engendran\n\na otras Personas". Left in place,
# chunk_text_paragraphs() (and RAG indexing, which uses it) can end chunks
# mid-sentence at those fake boundaries. Whitespace classes deliberately exclude
# : pdfminer's page separator must survive, extract_pdf() splits on it.
_FALSE_PARA_BREAK_RE = re.compile(r"(?<=[^\s.!?:;»\"'”)\]])[ \t]*\n[ \t\r]*\n[ \t\r\n]*(?=[a-záéíóúüñàèìòùâêîôûçäëïöÿ])")


def repair_paragraph_breaks(text: str) -> str:
    """Join paragraph breaks that fall mid-sentence (see _FALSE_PARA_BREAK_RE)
    into a single space. Conservative on purpose: requires BOTH no sentence-
    ending punctuation before AND a lowercase letter after, so headings,
    list items and real paragraph starts (capitalized) are never merged."""
    return _FALSE_PARA_BREAK_RE.sub(" ", text)


def extract_pdf(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception as e:
        raise ImportError(
            "pdfminer.six is required for PDF extraction. "
            "`pip install pdfminer.six`"
        ) from e
    text = repair_paragraph_breaks(extract_text(str(path)))

    # Scanned (image-only) pages have no text layer, so pdfminer returns ""
    # (just the \x0c page separator) for them -- OCR exactly those pages
    # and keep the real text layer everywhere else. Skipped entirely when
    # OCR isn't installed, i.e. unchanged pre-OCR behavior.
    from .ocr import ocr_available, ocr_pdf_pages, pages_needing_ocr

    if not ocr_available():
        return text
    pages = text.split("\x0c")
    if text.endswith("\x0c"):
        pages = pages[:-1]  # pdfminer terminates every page, incl. the last
    need = pages_needing_ocr(pages)
    if not need:
        return text
    try:
        ocr_pages = ocr_pdf_pages(path, need)
    except Exception as e:  # never lose the text layer over an OCR failure
        import logging

        logging.getLogger(__name__).warning("OCR failed for %s: %s", path.name, e)
        return text
    for i, ocr_text in ocr_pages.items():
        if sum(c.isalnum() for c in ocr_text) > sum(c.isalnum() for c in pages[i]):
            pages[i] = ocr_text
    # Pages joined by a paragraph break (not \x0c, which clean_text() would
    # collapse to a single space) so a page boundary never glues the last
    # paragraph of one page onto the first of the next.
    return "\n\n".join(pages)


_PARA_BREAK_RE = re.compile(r"\s*\n\s*\n\s*")
_WS_RE = re.compile(r"\s+")
_PARA_MARKER = "\x00"


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Mark real paragraph breaks (any whitespace run containing 2+
    # newlines) BEFORE the generic whitespace collapse below, or they're
    # indistinguishable from a single line-wrap newline -- every extractor
    # downstream (chunk_text_paragraphs()) depends on "\n\n" surviving here
    # as the one signal that a paragraph boundary existed in the source.
    text = _PARA_BREAK_RE.sub(_PARA_MARKER, text)
    # Collapse everything else -- single line-wrap newlines, tabs, runs of
    # spaces -- into one space, same as before.
    text = _WS_RE.sub(" ", text)
    text = text.replace(_PARA_MARKER, "\n\n")
    return text.strip()


def extract_html(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as e:
        raise ImportError(
            "beautifulsoup4 is required for HTML extraction. "
            "`pip install beautifulsoup4 lxml`"
        ) from e
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")


def _rows_to_markdown_table(rows: list[list[str]]) -> str:
    """First row is the header. LLMs parse Markdown tables far better than
    pipe-joined single lines, and it round-trips through chunking cleanly."""
    if not rows:
        return ""
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_docx(path: Path) -> str:
    try:
        import docx  # type: ignore
    except ImportError as e:
        raise ImportError(
            "python-docx is required for .docx extraction. `pip install python-docx`"
        ) from e
    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        rows = [[c.text.strip() for c in row.cells] for row in table.rows]
        rows = [r for r in rows if any(r)]
        if rows:
            parts.append(_rows_to_markdown_table(rows))
    return "\n".join(parts)


def extract_pptx(path: Path) -> str:
    try:
        from pptx import Presentation  # type: ignore
    except ImportError as e:
        raise ImportError(
            "python-pptx is required for .pptx extraction. `pip install python-pptx`"
        ) from e
    prs = Presentation(str(path))
    slides_text = []
    for i, slide in enumerate(prs.slides, 1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs)
                    if text.strip():
                        parts.append(text)
            if shape.has_table:
                rows = [[c.text.strip() for c in row.cells] for row in shape.table.rows]
                rows = [r for r in rows if any(r)]
                if rows:
                    parts.append(_rows_to_markdown_table(rows))
        if parts:
            slides_text.append(f"[Slide {i}]\n" + "\n".join(parts))
    return "\n\n".join(slides_text)


def extract_xlsx(path: Path) -> str:
    try:
        import openpyxl  # type: ignore
    except ImportError as e:
        raise ImportError(
            "openpyxl is required for .xlsx extraction. `pip install openpyxl`"
        ) from e
    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    sheets_text = []
    for sheet in wb.worksheets:
        # Blank (not drop) None cells -- dropping shifts remaining cells
        # into the wrong column, silently misaligning a Markdown table.
        rows = [[str(c) if c is not None else "" for c in row] for row in sheet.iter_rows(values_only=True)]
        rows = [r for r in rows if any(r)]
        if rows:
            sheets_text.append(f"[Sheet: {sheet.title}]\n" + _rows_to_markdown_table(rows))
    return "\n\n".join(sheets_text)


DOC_GLOBS = ("*.pdf", "*.txt", "*.docx", "*.pptx", "*.xlsx")


def supported_globs() -> tuple[str, ...]:
    """Folder-scan patterns for training/preview. Image files are only
    included when OCR is actually available -- otherwise every stray logo or
    screenshot in a docs folder would just count as a failed extraction."""
    from .ocr import IMAGE_EXTS, ocr_available

    if ocr_available():
        return DOC_GLOBS + tuple(f"*{e}" for e in IMAGE_EXTS)
    return DOC_GLOBS


def find_documents(source: Path) -> list[Path]:
    """Sorted, de-duplicated supported documents under `source`. rglob's
    patterns are case-sensitive on some platforms, so match on the
    lowercased suffix instead (a real folder of scans is often *.JPG)."""
    exts = {g[1:] for g in supported_globs()}
    return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def extract_any(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext in (".html", ".htm"):
        return extract_html(path)
    if ext == ".docx":
        return extract_docx(path)
    if ext == ".pptx":
        return extract_pptx(path)
    if ext == ".xlsx":
        return extract_xlsx(path)
    from .ocr import IMAGE_EXTS, extract_image

    if ext in IMAGE_EXTS:
        return extract_image(path)
    # markdown / txt / code — treat as plaintext
    return path.read_text(encoding="utf-8", errors="ignore")


# ── chunking ────────────────────────────────────────────────────────────────
def chunk_text(
    text: str, chunk_size: int = 1200, overlap: int = 150
) -> list[dict[str, Any]]:
    """Sliding-window chunks with deterministic ids for dedupe."""
    chunks: list[dict[str, Any]] = []
    if not text:
        return chunks
    step = max(1, chunk_size - overlap)
    for i, start in enumerate(range(0, len(text), step)):
        piece = text[start : start + chunk_size].strip()
        if not piece:
            continue
        cid = hashlib.sha256(piece.encode("utf-8")).hexdigest()[:16]
        chunks.append({"id": cid, "chunk_index": i, "char_start": start, "text": piece})
    return chunks


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text_structured(
    text: str, chunk_size: int = 1200, overlap_sentences: int = 1
) -> list[dict[str, Any]]:
    """Sentence-aware alternative to chunk_text(): packs whole sentences
    into each chunk up to `chunk_size` characters, instead of cutting mid-
    sentence (or mid-word) at a fixed character offset. A single sentence
    longer than chunk_size on its own falls back to character-slicing just
    that sentence, so this can never infinite-loop or silently drop text.

    Opt-in, not chunk_text()'s default: changing the default would alter
    behavior (and RAG index contents) for every existing caller at once.
    See --chunking in scripts/train_from_folder.py. See also
    chunk_text_paragraphs() for a coarser, paragraph-level alternative.
    """
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        if not current:
            return
        piece = " ".join(current)
        cid = hashlib.sha256(piece.encode("utf-8")).hexdigest()[:16]
        chunks.append({"id": cid, "chunk_index": len(chunks), "text": piece})

    for sent in sentences:
        if len(sent) > chunk_size:
            # Oversized single sentence: flush what's pending, then hard-slice
            # just this one sentence so nothing is silently dropped.
            flush()
            current, current_len = [], 0
            for start in range(0, len(sent), chunk_size):
                piece = sent[start : start + chunk_size]
                cid = hashlib.sha256(piece.encode("utf-8")).hexdigest()[:16]
                chunks.append({"id": cid, "chunk_index": len(chunks), "text": piece})
            continue
        if current and current_len + len(sent) + 1 > chunk_size:
            flush()
            # Sentence-based overlap: repeat the trailing N sentences from
            # the previous chunk so retrieval doesn't lose context right at
            # a chunk boundary (chunk_text()'s character overlap doesn't
            # translate cleanly to a sentence-packed scheme).
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) + 1 for s in current)
        current.append(sent)
        current_len += len(sent) + 1
    flush()
    return chunks


_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def chunk_text_paragraphs(
    text: str, chunk_size: int = 1200, overlap_paragraphs: int = 0
) -> list[dict[str, Any]]:
    """Paragraph-aware alternative to chunk_text_structured(): packs whole
    paragraphs into each chunk up to `chunk_size` characters, instead of
    packing at sentence granularity. Requires clean_text() (or equivalent)
    to have preserved blank-line paragraph breaks as "\\n\\n" first --
    plain single-newline-collapsed text has no paragraph boundaries left
    to split on.

    A paragraph longer than chunk_size on its own falls back to
    chunk_text_structured()'s sentence-level packing for just that
    paragraph, so nothing is silently dropped and nothing is cut mid-
    sentence.

    Opt-in, like chunk_text_structured() -- see --chunking in
    scripts/train_from_folder.py.
    """
    if not text:
        return []
    paragraphs = [p.strip() for p in _PARA_SPLIT_RE.split(text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        if not current:
            return
        piece = "\n\n".join(current)
        cid = hashlib.sha256(piece.encode("utf-8")).hexdigest()[:16]
        chunks.append({"id": cid, "chunk_index": len(chunks), "text": piece})

    for para in paragraphs:
        if len(para) > chunk_size:
            # Oversized single paragraph: flush what's pending, then fall
            # back to sentence-level packing for just this paragraph.
            flush()
            current, current_len = [], 0
            for sub in chunk_text_structured(para, chunk_size=chunk_size, overlap_sentences=0):
                chunks.append({"id": sub["id"], "chunk_index": len(chunks), "text": sub["text"]})
            continue
        if current and current_len + len(para) + 2 > chunk_size:
            flush()
            # Paragraph-based overlap: repeat the trailing N paragraphs
            # from the previous chunk, same rationale as
            # chunk_text_structured()'s sentence overlap.
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current_len = sum(len(p) + 2 for p in current)
        current.append(para)
        current_len += len(para) + 2
    flush()
    return chunks


def chunk_folder(
    folder: Path, chunker: Callable[[str], list[dict[str, Any]]] = chunk_text, tag_source: bool = False
) -> list[dict[str, Any]]:
    """Chunk every *.txt file in `folder` (sorted for determinism) with
    `chunker`. This is the "glob a folder of already-extracted .txt files
    and chunk each one" loop that src/cli.py's `ingest chunk`/`ingest
    build-dataset` and scripts/train_from_folder.py each used to duplicate
    with minor variations -- one shared implementation now.

    tag_source=False (default, matches the CLI's prior behavior): yields
    each chunker's own dict shape unchanged (e.g. chunk_text()'s
    id/chunk_index/char_start/text).
    tag_source=True (matches train_from_folder.py's prior behavior):
    yields the reduced {"id", "source", "text"} shape, where `source` is
    the originating file's stem -- used for dataset/corpus provenance,
    not needed by the CLI's simpler actions.
    """
    docs: list[dict[str, Any]] = []
    for txt_path in sorted(folder.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        for c in chunker(text):
            docs.append({"id": c["id"], "source": txt_path.stem, "text": c["text"]} if tag_source else c)
    return docs


def is_low_signal(text: str, min_alnum_ratio: float = 0.3) -> bool:
    """Catches TOC dot-leaders ("..... 12"), page-number-only lines, and
    other decoration-only chunks that add noise without content. The
    threshold is a starting heuristic, not a precisely tuned constant."""
    if not text.strip():
        return True
    alnum = sum(c.isalnum() for c in text)
    return (alnum / len(text)) < min_alnum_ratio


def filter_chunks(chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop low-signal chunks and exact-duplicate chunks (chunk_text()'s
    sha256-based ids make duplicates trivial to key on -- first occurrence
    wins, since `source` is just metadata, not used in training). Filter
    before dedup so a junk chunk that happens to be the first occurrence of
    a duplicate id doesn't win the slot."""
    by_id: dict[str, dict[str, Any]] = {}
    for c in chunks:
        if is_low_signal(c["text"]):
            continue
        by_id.setdefault(c["id"], c)
    return list(by_id.values())


# ── dataset builders ────────────────────────────────────────────────────────
def build_lm_dataset(chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Plain continued-pretraining rows: {"text": chunk, "id": ...}, no
    instruction framing. This is the honest format for "train on my
    documents" -- see build_sft_dataset's docstring for why that one isn't."""
    return [{"text": c["text"], "id": c["id"]} for c in chunks]


def build_sft_dataset(
    chunks: Iterable[dict[str, Any]],
    instruction: str = "Using the passage, answer:",
) -> list[dict[str, Any]]:
    """Instruction-wrapper rows with a permanently empty `output` -- kept
    only for backward compatibility with existing callers (src/cli.py,
    scripts/build_dataset.py) that depend on this exact shape. Since
    src.core.tokenize_dataset computes unmasked full-sequence loss for this
    shape, training on these rows is equivalent to build_lm_dataset() with
    a wasted constant instruction prefix -- prefer build_lm_dataset() for
    new document-training use. Not the default for scripts/train_from_folder.py
    since 2026-09 (was previously, before this distinction existed)."""
    return [
        {
            "instruction": instruction,
            "input": c["text"],
            "output": "",
            "id": c["id"],
        }
        for c in chunks
    ]


def build_qa_dataset(chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Heuristic: use first sentence of each chunk as the question stem."""
    out = []
    for c in chunks:
        first, _, rest = c["text"].partition(".")
        q = first.strip() or c["text"][:120]
        out.append(
            {
                "question": "What does the document say about: " + q + "?",
                "answer": (rest.strip() or c["text"]),
                "id": c["id"],
            }
        )
    return out
