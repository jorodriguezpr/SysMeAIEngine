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

"""Tests for src.ingestion.ocr and its wiring into extract_pdf()/extract_any().

The pure layout logic (line grouping, paragraph-gap detection, which pages
need OCR) is tested directly. extract_pdf()'s page-selection is tested with
pdfminer/OCR stubbed. One real end-to-end OCR test renders text with PIL
and runs the actual RapidOCR engine -- skipped when the optional OCR
packages aren't installed (requirements-ocr.txt)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import ocr as ocr_mod  # noqa: E402
from src.ingestion import pipeline  # noqa: E402


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_lines_group_boxes_on_same_visual_line_left_to_right():
    boxes = [_box(200, 10, 300, 30), _box(0, 12, 150, 31), _box(0, 50, 100, 70)]
    lines = ocr_mod._lines_from_result(boxes, ["world", "hello", "next"])
    assert [t for _, _, t in lines] == ["hello world", "next"]


def test_paragraph_break_inserted_only_at_large_vertical_gap():
    # 20px lines, 10px normal leading, one 40px gap before "Second".
    lines = [(0, 20, "First para line one"), (30, 50, "line two."), (90, 110, "Second para.")]
    assert ocr_mod._join_lines_with_paragraphs(lines) == "First para line one\nline two.\n\nSecond para."


def test_join_lines_uniform_spacing_has_no_paragraph_breaks():
    lines = [(i * 30, i * 30 + 20, f"line {i}") for i in range(5)]
    assert "\n\n" not in ocr_mod._join_lines_with_paragraphs(lines)


def test_pages_needing_ocr_flags_empty_and_near_empty_pages():
    pages = ["Real body text " * 10, "", "  12  ", "Another real page with plenty of words."]
    assert ocr_mod.pages_needing_ocr(pages) == [1, 2]


def test_ocr_status_respects_env_off(monkeypatch):
    monkeypatch.setenv("ENGINE_OCR", "off")
    st = ocr_mod.ocr_status()
    assert st["available"] is False and "ENGINE_OCR" in st["reason"]


def test_extract_pdf_without_ocr_returns_text_layer_unchanged(monkeypatch, tmp_path):
    import pdfminer.high_level as hl

    monkeypatch.setattr(hl, "extract_text", lambda p: "Only text.\x0c\x0c")
    monkeypatch.setattr(ocr_mod, "ocr_available", lambda: False)
    assert pipeline.extract_pdf(tmp_path / "x.pdf") == "Only text.\x0c\x0c"


def test_extract_pdf_ocrs_only_pages_without_text_layer(monkeypatch, tmp_path):
    import pdfminer.high_level as hl

    body = "A real text layer page with lots of words in it."
    monkeypatch.setattr(hl, "extract_text", lambda p: f"{body}\x0c\x0c{body}\x0c")
    monkeypatch.setattr(ocr_mod, "ocr_available", lambda: True)
    calls = []

    def fake_ocr(path, idx):
        calls.append(idx)
        return {i: "Recovered scanned page text." for i in idx}

    monkeypatch.setattr(ocr_mod, "ocr_pdf_pages", fake_ocr)
    out = pipeline.extract_pdf(tmp_path / "x.pdf")
    assert calls == [[1]]
    assert out == f"{body}\n\nRecovered scanned page text.\n\n{body}"


def test_extract_pdf_keeps_text_layer_when_ocr_raises(monkeypatch, tmp_path):
    import pdfminer.high_level as hl

    monkeypatch.setattr(hl, "extract_text", lambda p: "Real text here, plenty of it.\x0c\x0c")
    monkeypatch.setattr(ocr_mod, "ocr_available", lambda: True)

    def boom(path, idx):
        raise RuntimeError("render failed")

    monkeypatch.setattr(ocr_mod, "ocr_pdf_pages", boom)
    assert pipeline.extract_pdf(tmp_path / "x.pdf") == "Real text here, plenty of it.\x0c\x0c"


def test_find_documents_includes_images_only_when_ocr_available(monkeypatch, tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "scan.JPG").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    monkeypatch.setattr(ocr_mod, "ocr_available", lambda: False)
    assert [p.name for p in pipeline.find_documents(tmp_path)] == ["a.pdf", "notes.txt"]
    monkeypatch.setattr(ocr_mod, "ocr_available", lambda: True)
    assert [p.name for p in pipeline.find_documents(tmp_path)] == ["a.pdf", "notes.txt", "scan.JPG"]


@pytest.mark.skipif(not ocr_mod.ocr_available(), reason="OCR packages not installed (requirements-ocr.txt)")
def test_real_ocr_on_rendered_image(tmp_path):
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except OSError:
        font = ImageFont.load_default(size=36)
    img = Image.new("RGB", (1100, 330), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 20), "The quick brown fox jumps", fill="black", font=font)
    d.text((20, 70), "over the lazy dog.", fill="black", font=font)
    d.text((20, 200), "Second paragraph here.", fill="black", font=font)
    path = tmp_path / "page.png"
    img.save(path)

    text = pipeline.extract_any(path)
    assert "quick brown fox" in text
    assert "Second paragraph" in text
    # the 80px gap before the second paragraph survives as a blank line
    assert text.index("\n\n") > text.index("lazy dog")
