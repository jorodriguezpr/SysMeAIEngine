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

"""Ingestion pipeline tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.pipeline import (build_lm_dataset, build_qa_dataset,  # noqa: E402
                                    build_sft_dataset, chunk_folder,
                                    chunk_text, chunk_text_paragraphs,
                                    chunk_text_structured, clean_text,
                                    extract_docx, extract_pptx, extract_xlsx,
                                    filter_chunks, is_low_signal)


def test_clean_text_normalizes_whitespace():
    assert clean_text("a\r\n  b   \t c\n") == "a b c"


def test_clean_text_preserves_paragraph_breaks():
    # A real blank-line paragraph break must survive as "\n\n" -- previously
    # this collapsed to a single space, indistinguishable from a line wrap.
    assert clean_text("Para one.\n\nPara two.") == "Para one.\n\nPara two."


def test_clean_text_preserves_paragraph_breaks_with_stray_whitespace():
    # A "blank" line that actually has trailing spaces/tabs on it is still
    # a paragraph break, not a run of generic whitespace.
    assert clean_text("Para one.\n  \t\nPara two.") == "Para one.\n\nPara two."


def test_clean_text_collapses_single_line_wraps_to_space():
    # A lone newline (mid-paragraph line wrap, no blank line) is not a
    # paragraph break -- it should join into a space like any other run.
    assert clean_text("word one\nword two") == "word one word two"


def test_clean_text_collapses_three_or_more_newlines_to_one_break():
    assert clean_text("Para one.\n\n\n\nPara two.") == "Para one.\n\nPara two."


def test_chunk_text_is_deterministic_and_overlapping():
    text = "0123456789" * 100  # 1000 chars
    c1 = chunk_text(text, chunk_size=400, overlap=50)
    c2 = chunk_text(text, chunk_size=400, overlap=50)
    assert [c["id"] for c in c1] == [c["id"] for c in c2]
    assert c1[0]["text"] == text[:400]
    assert c1[1]["char_start"] == 350  # 400 - 50


def test_build_sft_dataset():
    chunks = [{"id": "x", "text": "hello world"}]
    rows = build_sft_dataset(chunks, instruction="inst")
    assert rows == [{"instruction": "inst", "input": "hello world", "output": "", "id": "x"}]


def test_build_qa_dataset_uses_first_sentence():
    chunks = [{"id": "x", "text": "The quick brown fox. Jumps over dogs."}]
    rows = build_qa_dataset(chunks)
    assert rows[0]["question"].endswith("The quick brown fox?")
    assert rows[0]["answer"].strip().startswith("Jumps")


def test_chunk_text_structured_never_cuts_mid_sentence():
    text = "First sentence here. Second sentence follows. Third one too. Fourth."
    chunks = chunk_text_structured(text, chunk_size=40, overlap_sentences=0)
    for c in chunks:
        # every chunk should end at a real sentence boundary, not mid-word
        assert c["text"].rstrip().endswith((".", "!", "?"))
    # reassembling (ignoring overlap, which is 0 here) should reproduce
    # every sentence with nothing silently dropped
    joined = " ".join(c["text"] for c in chunks)
    assert "First sentence here." in joined
    assert "Fourth." in joined


def test_chunk_text_structured_overlap_repeats_trailing_sentences():
    text = "One. Two. Three. Four. Five."
    chunks = chunk_text_structured(text, chunk_size=12, overlap_sentences=1)
    # with overlap, consecutive chunks should share at least the boundary sentence
    assert len(chunks) >= 2
    for i in range(len(chunks) - 1):
        last_sentence_of_prev = chunks[i]["text"].split(".")[-2].strip() + "."
        assert last_sentence_of_prev in chunks[i + 1]["text"]


def test_chunk_text_structured_oversized_sentence_falls_back_to_slicing():
    huge_sentence = "word " * 500 + "."  # a single "sentence" far exceeding chunk_size
    chunks = chunk_text_structured(huge_sentence, chunk_size=100)
    assert len(chunks) > 1
    # nothing lost: every char of the original appears somewhere across chunks
    assert sum(len(c["text"]) for c in chunks) >= len(huge_sentence) - 10


def test_chunk_text_structured_empty_text():
    assert chunk_text_structured("") == []


def test_chunk_text_paragraphs_keeps_whole_paragraphs_together():
    text = "First paragraph, short.\n\nSecond paragraph, also short.\n\nThird one."
    chunks = chunk_text_paragraphs(text, chunk_size=1000)
    # all three paragraphs fit comfortably in one chunk together
    assert len(chunks) == 1
    assert "First paragraph, short." in chunks[0]["text"]
    assert "Third one." in chunks[0]["text"]


def test_chunk_text_paragraphs_splits_when_over_budget():
    text = "Para one is here.\n\nPara two is here.\n\nPara three is here."
    chunks = chunk_text_paragraphs(text, chunk_size=30)
    assert len(chunks) > 1
    # nothing lost: every paragraph appears somewhere across the chunks
    joined = "\n\n".join(c["text"] for c in chunks)
    assert "Para one is here." in joined
    assert "Para two is here." in joined
    assert "Para three is here." in joined


def test_chunk_text_paragraphs_oversized_paragraph_falls_back_to_sentences():
    huge_para = "Sentence one. " * 200  # a single paragraph far exceeding chunk_size
    chunks = chunk_text_paragraphs(huge_para, chunk_size=100)
    assert len(chunks) > 1
    for c in chunks:
        assert c["text"].rstrip().endswith(".")


def test_chunk_text_paragraphs_empty_text():
    assert chunk_text_paragraphs("") == []


def test_chunk_text_paragraphs_ignores_plain_line_wraps():
    # No real paragraph breaks (single newlines only) -- treated as one paragraph.
    text = "line one\nline two\nline three"
    chunks = chunk_text_paragraphs(text, chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_build_lm_dataset():
    chunks = [{"id": "x", "text": "hello world"}]
    assert build_lm_dataset(chunks) == [{"text": "hello world", "id": "x"}]


def test_is_low_signal_flags_toc_dots():
    assert is_low_signal("..... 12 ..... 13 .....") is True


def test_is_low_signal_flags_empty():
    assert is_low_signal("   ") is True


def test_is_low_signal_passes_prose():
    assert is_low_signal("This is a normal sentence with real content.") is False


def test_filter_chunks_drops_low_signal_and_exact_duplicates():
    chunks = [
        {"id": "a", "text": "A real sentence with content."},
        {"id": "a", "text": "A real sentence with content."},  # exact dup id
        {"id": "b", "text": "..... 12 ....."},  # low signal
        {"id": "c", "text": "Another real sentence here."},
    ]
    out = filter_chunks(chunks)
    assert [c["id"] for c in out] == ["a", "c"]


def test_chunk_folder_default_shape_matches_chunker_output(tmp_path):
    (tmp_path / "a.txt").write_text("0123456789" * 100, encoding="utf-8")
    chunks = chunk_folder(tmp_path)
    assert chunks == chunk_text("0123456789" * 100)


def test_chunk_folder_tag_source_reduces_shape_and_tags_stem(tmp_path):
    (tmp_path / "doc-one.txt").write_text("hello world, this is doc one.", encoding="utf-8")
    chunks = chunk_folder(tmp_path, tag_source=True)
    assert len(chunks) == 1
    assert set(chunks[0]) == {"id", "source", "text"}
    assert chunks[0]["source"] == "doc-one"


def test_chunk_folder_processes_multiple_files_in_sorted_order(tmp_path):
    (tmp_path / "b.txt").write_text("second file content here.", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first file content here.", encoding="utf-8")
    chunks = chunk_folder(tmp_path, tag_source=True)
    assert [c["source"] for c in chunks] == ["a", "b"]


def test_chunk_folder_uses_given_chunker(tmp_path):
    (tmp_path / "a.txt").write_text("One. Two. Three. Four. Five.", encoding="utf-8")
    chunks = chunk_folder(tmp_path, chunker=chunk_text_structured)
    assert chunks == chunk_text_structured("One. Two. Three. Four. Five.")


def test_extract_docx_emits_markdown_table(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "t.docx"
    doc = docx.Document()
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Name"
    table.rows[0].cells[1].text = "Value"
    table.rows[1].cells[0].text = "Foo"
    table.rows[1].cells[1].text = "Bar"
    doc.save(str(path))
    text = extract_docx(path)
    assert "| Name | Value |" in text
    assert "|---|---|" in text
    assert "| Foo | Bar |" in text


def test_extract_pptx_emits_markdown_table(tmp_path):
    pptx_mod = pytest.importorskip("pptx")
    path = tmp_path / "t.pptx"
    prs = pptx_mod.Presentation()
    layout = prs.slide_layouts[6]  # blank layout
    slide = prs.slides.add_slide(layout)
    shape = slide.shapes.add_table(2, 2, 0, 0, pptx_mod.util.Inches(4), pptx_mod.util.Inches(2))
    table = shape.table
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Foo"
    table.cell(1, 1).text = "Bar"
    prs.save(str(path))
    text = extract_pptx(path)
    assert "| Name | Value |" in text
    assert "|---|---|" in text
    assert "| Foo | Bar |" in text


def test_extract_xlsx_blanks_none_cells_not_drops(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "t.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["A", "B", "C"])
    ws.append(["1", None, "3"])
    wb.save(str(path))
    text = extract_xlsx(path)
    # column alignment preserved: 3 cells per row, not 2 (None dropped)
    assert "| 1 |  | 3 |" in text


def test_repair_paragraph_breaks_joins_mid_sentence_layout_splits():
    from src.ingestion.pipeline import repair_paragraph_breaks

    assert repair_paragraph_breaks("Ellos engendran\n\na otras Personas.") == "Ellos engendran a otras Personas."
    # real paragraph starts / sentence ends / headings are left alone
    assert repair_paragraph_breaks("End of one.\n\nnext starts lower") == "End of one.\n\nnext starts lower"
    assert repair_paragraph_breaks("Heading\n\nCapitalized body.") == "Heading\n\nCapitalized body."
    assert repair_paragraph_breaks("- item one\n\n- item two") == "- item one\n\n- item two"


def test_repair_paragraph_breaks_never_crosses_pdf_page_separator():
    from src.ingestion.pipeline import repair_paragraph_breaks

    text = "page one ends here\n\n\x0cand page two continues"
    assert repair_paragraph_breaks(text).count("\x0c") == 1
