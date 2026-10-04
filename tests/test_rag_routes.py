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

"""Tests for the RAG chat route's second-pass answer polish -- a cheap
follow-up generation that asks the model to fix its own spelling/grammar
without changing meaning. Mocks core_chat so this stays fast/offline; a real
end-to-end comparison (aprocrifos-v2 vs the base model, same corpus) was done
manually against the live server during development to confirm the garbling
this targets is real (e.g. "Nicodmo" for "Nicodemo")."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import rag_routes  # noqa: E402


def _fake_chat_result(text: str) -> dict:
    return {
        "model": "m", "prompt_tokens": 1, "completion_tokens": 1,
        "elapsed_ms": 0.0, "text": text,
    }


def test_polish_answer_returns_corrected_text():
    original = "Nicodemo es un interlocutor biblico mencionado en el Evangelio de Nicodmo."
    corrected = "Nicodemo es un interlocutor biblico mencionado en el Evangelio de Nicodemo."
    with patch.object(rag_routes, "core_chat", return_value=_fake_chat_result(corrected)):
        out = rag_routes._polish_answer(state=None, text=original, model_name="m")
    assert out == corrected


def test_polish_answer_falls_back_when_result_balloons():
    original = "Short answer."
    runaway = "Short answer. " * 50  # far more than 2x the original length
    with patch.object(rag_routes, "core_chat", return_value=_fake_chat_result(runaway)):
        out = rag_routes._polish_answer(state=None, text=original, model_name="m")
    assert out is None


def test_polish_answer_falls_back_when_result_guts_the_answer():
    original = "This is a reasonably detailed answer with real content in it."
    gutted = "No."
    with patch.object(rag_routes, "core_chat", return_value=_fake_chat_result(gutted)):
        out = rag_routes._polish_answer(state=None, text=original, model_name="m")
    assert out is None


def test_polish_answer_falls_back_on_exception():
    with patch.object(rag_routes, "core_chat", side_effect=RuntimeError("boom")):
        out = rag_routes._polish_answer(state=None, text="anything", model_name="m")
    assert out is None


def test_polish_answer_falls_back_on_empty_result():
    with patch.object(rag_routes, "core_chat", return_value=_fake_chat_result("   ")):
        out = rag_routes._polish_answer(state=None, text="anything", model_name="m")
    assert out is None
