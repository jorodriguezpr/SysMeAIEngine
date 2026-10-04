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

"""OCR for scanned PDFs and image files.

Backend: RapidOCR (`pip install rapidocr pypdfium2`, see requirements-ocr.txt)
-- chosen over Tesseract because it's pip-only (no separate Windows binary
install) and its ONNX models ship inside the wheel (no network on first
use). It runs on whichever `onnxruntime` module is importable, including
the one `onnxruntime-directml` provides -- `rapidocr` 3.x does NOT pull in
a separate plain `onnxruntime` wheel, which would conflict with the
DirectML build (see docs/TRAINING.md).

OCR is optional everywhere: when the packages aren't installed, or
ENGINE_OCR=off, callers fall back to exactly the pre-OCR behavior (text-
layer extraction only; image files are simply not picked up).

Output keeps paragraph structure: text boxes are grouped into visual lines
by vertical position, and a vertical gap noticeably larger than normal line
spacing becomes a "\\n\\n" paragraph break -- the one signal
clean_text()/chunk_text_paragraphs() rely on.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

# A PDF page whose text layer has fewer alphanumeric characters than this is
# treated as scanned (image-only) and OCR'd instead. Real text pages have
# hundreds; scanned pages typically yield "" or just a page-number/header.
MIN_TEXT_LAYER_ALNUM = 25

# Render scale for PDF pages (1.0 = 72 dpi). 2.0 (~144 dpi) measured as
# accurate on real Spanish text incl. accents at ~3s/page on CPU; higher is
# slower for little gain on normal body text.
PDF_RENDER_SCALE = 2.0

_engine: Any = None
_unavailable_reason: str | None = None


def _disabled_by_env() -> bool:
    return os.getenv("ENGINE_OCR", "auto").strip().lower() in ("off", "0", "false", "no")


def ocr_status() -> dict[str, Any]:
    """{"available": bool, "backend": str|None, "reason": str|None} -- cheap
    import check only, does not load the OCR models."""
    if _disabled_by_env():
        return {"available": False, "backend": None, "reason": "disabled via ENGINE_OCR=off"}
    missing = []
    for mod in ("rapidocr", "pypdfium2", "onnxruntime"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        return {
            "available": False,
            "backend": None,
            "reason": f"missing package(s): {', '.join(missing)} -- `pip install -r requirements-ocr.txt`",
        }
    return {"available": True, "backend": "rapidocr", "reason": None}


def ocr_available() -> bool:
    return ocr_status()["available"]


def _get_engine() -> Any:
    global _engine, _unavailable_reason
    if _engine is not None:
        return _engine
    status = ocr_status()
    if not status["available"]:
        raise ImportError(f"OCR unavailable: {status['reason']}")
    from rapidocr import RapidOCR  # type: ignore

    # RapidOCR logs every model-file check at INFO -- a wall of noise in a
    # training job's log for zero information. Its constructor re-applies
    # its own configured level, so setting the logger's level from outside
    # doesn't stick; this config key is the supported way.
    _engine = RapidOCR(params={"Global.log_level": "warning"})
    return _engine


def _lines_from_result(boxes: Any, txts: Any) -> list[tuple[float, float, str]]:
    """Group text boxes into visual lines -> [(top, bottom, text)], top-to-
    bottom. RapidOCR can emit several boxes for one visual line (columns,
    wide word gaps); boxes whose vertical centers fall within half a line
    height of each other are joined left-to-right."""
    items = []
    for box, txt in zip(boxes, txts):
        if not txt or not str(txt).strip():
            continue
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        items.append((min(ys), max(ys), min(xs), str(txt).strip()))
    if not items:
        return []
    items.sort(key=lambda it: ((it[0] + it[1]) / 2, it[2]))

    lines: list[list[tuple[float, float, float, str]]] = []
    for it in items:
        center = (it[0] + it[1]) / 2
        if lines:
            cur = lines[-1]
            cur_top = min(i[0] for i in cur)
            cur_bot = max(i[1] for i in cur)
            cur_center = (cur_top + cur_bot) / 2
            if abs(center - cur_center) <= (cur_bot - cur_top) / 2:
                cur.append(it)
                continue
        lines.append([it])

    out = []
    for ln in lines:
        ln.sort(key=lambda i: i[2])
        out.append((min(i[0] for i in ln), max(i[1] for i in ln), " ".join(i[3] for i in ln)))
    return out


def _join_lines_with_paragraphs(lines: list[tuple[float, float, str]]) -> str:
    """Single newline between lines of the same paragraph, blank line where
    the vertical gap exceeds ~1.5x the typical (median) gap -- i.e. the
    extra spacing a real paragraph break or heading leaves on the page."""
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0][2]
    gaps = [max(0.0, lines[i + 1][0] - lines[i][1]) for i in range(len(lines) - 1)]
    heights = sorted(b - t for t, b, _ in lines)
    median_h = heights[len(heights) // 2] or 1.0
    sorted_gaps = sorted(gaps)
    # Lower median: with few lines (e.g. 2 gaps, one of them the paragraph
    # break) the upper median would BE the paragraph gap and hide it.
    median_gap = sorted_gaps[(len(sorted_gaps) - 1) // 2]
    # Floor the threshold at a fraction of line height so a page with
    # near-zero normal leading doesn't turn every tiny jitter into a break.
    threshold = max(median_gap * 1.5, median_h * 0.6)
    parts = [lines[0][2]]
    for gap, (_, _, text) in zip(gaps, lines[1:]):
        parts.append("\n\n" if gap > threshold else "\n")
        parts.append(text)
    return "".join(parts)


def ocr_image(image: Any) -> str:
    """OCR one image -- a path or a PIL.Image. Returns text with paragraph
    breaks preserved as blank lines."""
    engine = _get_engine()
    if isinstance(image, (str, Path)):
        from PIL import Image  # type: ignore

        with Image.open(image) as im:
            arr = _to_array(im)
    else:
        arr = _to_array(image)
    result = engine(arr)
    if result is None or result.txts is None or result.boxes is None:
        return ""
    return _join_lines_with_paragraphs(_lines_from_result(result.boxes, result.txts))


def _to_array(im: Any) -> Any:
    import numpy as np  # type: ignore

    return np.asarray(im.convert("RGB"))


def extract_image(path: Path) -> str:
    """Image-file extractor for extract_any(). Raises ImportError with the
    install hint when OCR isn't available (same contract as the other
    optional-dependency extractors)."""
    return ocr_image(path)


def _alnum_count(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def pages_needing_ocr(page_texts: list[str]) -> list[int]:
    return [i for i, t in enumerate(page_texts) if _alnum_count(t) < MIN_TEXT_LAYER_ALNUM]


def ocr_pdf_pages(path: Path, page_indices: list[int]) -> dict[int, str]:
    """Render the given 0-based pages of a PDF and OCR each one."""
    import pypdfium2 as pdfium  # type: ignore

    out: dict[int, str] = {}
    pdf = pdfium.PdfDocument(str(path))
    try:
        for i in page_indices:
            if i >= len(pdf):
                continue
            page = pdf[i]
            try:
                img = page.render(scale=PDF_RENDER_SCALE).to_pil()
                out[i] = ocr_image(img)
            finally:
                page.close()
    finally:
        pdf.close()
    return out
