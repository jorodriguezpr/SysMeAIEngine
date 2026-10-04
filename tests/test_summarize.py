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

"""Tests for src.core.summarize -- the map-reduce document summarizer and
its chat-side intent detector. Mocks core_chat/extraction so this stays
fast/offline."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import summarize as summarize_mod  # noqa: E402


# ── detect_summarize_request ──────────────────────────────────────────────
@pytest.mark.parametrize(
    "prompt",
    [
        r"summarize C:\opt\Aprocrifos\book.pdf",
        r'please give me a summary of "C:\opt\Aprocrifos\book.pdf"',
        r"resume el documento C:\opt\Aprocrifos\book.pdf",
        r"/home/user/docs/book.txt please summarize this",
    ],
)
def test_detect_summarize_request_matches_keyword_plus_path(prompt):
    assert summarize_mod.detect_summarize_request(prompt) is not None


def test_detect_summarize_request_ignores_keyword_without_path():
    assert summarize_mod.detect_summarize_request("can you summarize that for me?") is None


def test_detect_summarize_request_ignores_path_without_keyword():
    assert summarize_mod.detect_summarize_request(r"open C:\opt\Aprocrifos\book.pdf") is None


def test_detect_summarize_request_false_positive_guard():
    """"resume" alone (English "resume the download") must not fire without
    an accompanying file path -- this is the whole reason the path is
    required, not just the keyword."""
    assert summarize_mod.detect_summarize_request("please resume the download") is None


def test_detect_summarize_request_extracts_exact_path():
    path = summarize_mod.detect_summarize_request(r"summarize C:\opt\Aprocrifos\book.pdf now")
    assert path == r"C:\opt\Aprocrifos\book.pdf"


# ── summarize_document (map-reduce) ───────────────────────────────────────
def _fake_chat_result(text: str) -> dict:
    return {"model": "m", "prompt_tokens": 1, "completion_tokens": 1, "elapsed_ms": 0.0, "text": text}


def _fake_state():
    return type("FakeState", (), {"config": type("FakeConfig", (), {"default_model": "m"})()})()


def test_summarize_document_single_chunk_skips_reduce(tmp_path):
    doc = tmp_path / "book.txt"
    doc.write_text("A short document with just one paragraph of content.", encoding="utf-8")

    with patch.object(summarize_mod, "core_chat", return_value=_fake_chat_result("A short summary.")):
        result = summarize_mod.summarize_document(state=_fake_state(), source=doc)

    assert result["summary"] == "A short summary."
    assert result["chunk_count"] == 1
    assert result["reduce_rounds"] == 0


def test_summarize_document_many_chunks_triggers_reduce(tmp_path):
    # Enough distinct paragraphs to become multiple chunks, and long enough
    # per-chunk summaries that their combined length forces a reduce round.
    paras = [f"Paragraph {i} with some real content to chunk on its own. " * 10 for i in range(10)]
    doc = tmp_path / "book.txt"
    doc.write_text("\n\n".join(paras), encoding="utf-8")

    long_summary = "This chunk talks about something specific and detailed. " * 20  # ~1100 chars
    calls = {"n": 0}

    def fake_chat(state, prompt, model_name=None, max_new_tokens=256, temperature=0.7):
        calls["n"] += 1
        # reduce-round prompts are built from prior summaries and are much
        # longer than a single chunk-summary reply -- return a short final
        # answer once we're clearly in a reduce/final call to let the loop
        # actually terminate.
        if "Summary:" in prompt and prompt.count("Paragraph") == 0:
            return _fake_chat_result("Final combined summary.")
        return _fake_chat_result(long_summary)

    with patch.object(summarize_mod, "core_chat", side_effect=fake_chat):
        result = summarize_mod.summarize_document(
            state=_fake_state(), source=doc, chunk_size=200  # small chunk_size -> many chunks
        )

    assert result["chunk_count"] > 1
    assert calls["n"] > result["chunk_count"]  # at least one reduce/final call happened
    assert result["summary"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("El evangelio de Nicodemo cuenta que Jesús fue llevado ante Pilato en Jerusalén.", "Spanish"),
        ("The gospel of Nicodemus tells that Jesus was brought before Pilate in Jerusalem.", "English"),
        ("Según el texto, la fe y la esperanza son fundamentales para el creyente.", "Spanish"),
    ],
)
def test_detect_language(text, expected):
    assert summarize_mod._detect_language(text) == expected


def test_summarize_document_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize_mod.summarize_document(state=None, source=tmp_path / "nope.pdf")
