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

"""Tests for src.core.token_preview -- exact token counts for the
Train-from-Folder preview. The gpt2 tokenizer test uses the real tokenizer
(files only, ~1MB, already cached wherever this project has trained gpt2)
and is skipped if it can't be loaded, e.g. offline on a fresh machine."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import token_preview as tp  # noqa: E402
from src.core.tokenize_dataset import qa_prompt  # noqa: E402


def test_row_training_text_matches_what_training_tokenizes():
    assert tp.row_training_text({"text": "plain"}) == "plain"
    assert tp.row_training_text({"question": "Q?", "answer": "A."}) == qa_prompt("Q?") + "A."


def test_tinygpt_counts_utf8_bytes_exactly():
    out = tp.count_row_tokens([{"text": "año"}], "minilm", max_seq=2)
    assert out["sample_tokens"] == 4  # 'ñ' is 2 bytes
    assert out["truncation_applies"] is False
    assert out["method"].startswith("exact")


def test_falls_back_to_labeled_estimate_when_tokenizer_unavailable(monkeypatch):
    monkeypatch.setattr(tp, "load_preview_tokenizer", lambda m: (None, "OSError: offline"))
    out = tp.count_row_tokens([{"text": "x" * 400}], "some/model", max_seq=50)
    assert out["sample_tokens"] == 100
    assert out["method"].startswith("estimate") and "offline" in out["method"]
    assert out["rows_truncated"] == 1
    assert out["sample_tokens_trained"] == 50


def test_real_gpt2_tokenizer_counts_and_truncation():
    tok, err = tp.load_preview_tokenizer("gpt2")
    if tok is None:
        pytest.skip(f"gpt2 tokenizer unavailable: {err}")
    rows = [{"text": "hello world"}, {"text": "word " * 300}]
    out = tp.count_row_tokens(rows, "gpt2", max_seq=128)
    expected = [len(tok(r["text"])["input_ids"]) for r in rows]
    assert out["sample_tokens"] == sum(expected)
    assert out["rows_truncated"] == 1
    assert out["sample_tokens_trained"] == expected[0] + 128
    assert out["method"] == "exact (gpt2 tokenizer)"
