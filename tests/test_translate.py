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

"""Tests for src.core.translate -- the dedicated MarianMT translator and its
chat-side intent detector. Mocks the model/tokenizer so this stays
fast/offline (opus-mt checkpoints are real ~300MB downloads, not something
to fetch in a unit test)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import translate as translate_mod  # noqa: E402


# ── translate() ────────────────────────────────────────────────────────────
def _fake_seq2seq_model_and_tokenizer():
    torch = pytest.importorskip("torch")
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tokenizer.decode.return_value = "translated text"
    model = MagicMock()
    model.generate.return_value = torch.tensor([[9, 9]])
    return model, tokenizer


def test_translate_uses_loaded_model_and_decodes(monkeypatch):
    model, tok = _fake_seq2seq_model_and_tokenizer()
    monkeypatch.setattr(translate_mod, "_load", lambda direction: (model, tok))

    out = translate_mod.translate("hola mundo", "es-en")

    assert out == "translated text"
    model.generate.assert_called_once()


def test_translate_rejects_unsupported_direction():
    with pytest.raises(ValueError):
        translate_mod.translate("bonjour", "fr-en")


# ── detect_translate_request ──────────────────────────────────────────────
@pytest.mark.parametrize(
    "prompt,expected_text,expected_direction",
    [
        ('translate "hola mundo" to english', "hola mundo", "es-en"),
        ("traduce al inglés: hola mundo", "hola mundo", "es-en"),
        ('translate "hello world" to spanish', "hello world", "en-es"),
        ("traducir al español: hello world", "hello world", "en-es"),
    ],
)
def test_detect_translate_request_matches(prompt, expected_text, expected_direction):
    result = translate_mod.detect_translate_request(prompt)
    assert result == (expected_text, expected_direction)


def test_detect_translate_request_ignores_keyword_without_direction():
    assert translate_mod.detect_translate_request('translate "hola mundo"') is None


def test_detect_translate_request_ignores_direction_without_text_marker():
    # keyword + direction present, but no quotes or colon to anchor the text
    assert translate_mod.detect_translate_request("translate hola mundo to english please") is None


def test_detect_translate_request_ignores_unrelated_prompt():
    assert translate_mod.detect_translate_request("what time is it?") is None
