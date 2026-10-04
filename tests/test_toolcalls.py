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

"""Tests for src.core.toolcalls -- specifically citation accumulation
through the auto-executed search_local_knowledge tool, which previously had
no way to reach the caller (the model's own prose doesn't reliably mention
sources). Mocks both core_chat and rag_mod.retrieve so this stays fast and
offline, unlike the live verification done manually against a real
registered model + real RAG corpus during development."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import toolcalls  # noqa: E402


_FAKE_HITS = [
    {"corpus": "borland", "source": "doc-a", "id": "id1", "text": "passage one", "score": 0.9},
    {"corpus": "borland", "source": "doc-b", "id": "id2", "text": "passage two", "score": 0.8},
]


def test_citations_populated_from_auto_executed_tool_call():
    responses = [
        {
            "model": "stub",
            "text": '<tool_call>{"name": "search_local_knowledge", "arguments": {"query": "x"}}</tool_call>',
            "prompt_tokens": 1, "completion_tokens": 1,
        },
        {"model": "stub", "text": "Here is the answer.", "prompt_tokens": 1, "completion_tokens": 1},
    ]

    def fake_chat(*a, **kw):
        return responses.pop(0)

    with patch.object(toolcalls, "core_chat", side_effect=fake_chat), \
         patch.object(toolcalls.rag_mod, "retrieve", return_value=_FAKE_HITS):
        result = toolcalls.run_chat_with_tools(
            state=None, messages=[{"role": "user", "content": "hi"}],
            model_name=None, caller_tools=None, max_new_tokens=10, temperature=0,
        )

    assert result["citations"] == [
        {"corpus": "borland", "source": "doc-a", "id": "id1"},
        {"corpus": "borland", "source": "doc-b", "id": "id2"},
    ]
    assert result["text"] == "Here is the answer."


def test_citations_deduped_across_repeated_tool_calls():
    responses = [
        {
            "model": "stub",
            "text": '<tool_call>{"name": "search_local_knowledge", "arguments": {"query": "x"}}</tool_call>',
            "prompt_tokens": 1, "completion_tokens": 1,
        },
        {
            "model": "stub",
            "text": '<tool_call>{"name": "search_local_knowledge", "arguments": {"query": "y"}}</tool_call>',
            "prompt_tokens": 1, "completion_tokens": 1,
        },
        {"model": "stub", "text": "Final answer.", "prompt_tokens": 1, "completion_tokens": 1},
    ]

    def fake_chat(*a, **kw):
        return responses.pop(0)

    with patch.object(toolcalls, "core_chat", side_effect=fake_chat), \
         patch.object(toolcalls.rag_mod, "retrieve", return_value=_FAKE_HITS):
        result = toolcalls.run_chat_with_tools(
            state=None, messages=[{"role": "user", "content": "hi"}],
            model_name=None, caller_tools=None, max_new_tokens=10, temperature=0,
        )

    # Same two hits retrieved twice (across two rounds) -- deduped, not doubled.
    assert len(result["citations"]) == 2


def test_citations_empty_when_no_tool_called():
    with patch.object(
        toolcalls, "core_chat",
        return_value={"model": "stub", "text": "plain answer, no tools", "prompt_tokens": 1, "completion_tokens": 1},
    ):
        result = toolcalls.run_chat_with_tools(
            state=None, messages=[{"role": "user", "content": "hi"}],
            model_name=None, caller_tools=None, max_new_tokens=10, temperature=0,
        )
    assert result["citations"] == []
