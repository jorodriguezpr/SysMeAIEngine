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

"""llama.cpp provider: must use the GGUF's chat template (like Ollama),
Ollama-equivalent sampling, and refuse prompts that exceed the context
window with a clear message. The Llama object is stubbed."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.providers import llama_cpp as lc  # noqa: E402


class FakeLlama:
    def __init__(self, template=True, n_ctx=8192):
        self.metadata = {"tokenizer.chat_template": "{{ messages }}"} if template else {}
        self._n_ctx = n_ctx
        self.calls = []

    def n_ctx(self):
        return self._n_ctx

    def tokenize(self, b, add_bos=False):
        return list(range(len(b.split())))

    def create_chat_completion(self, messages, **kw):
        self.calls.append(("chat", messages, kw))
        return {"choices": [{"message": {"content": "respuesta"}}]}

    def __call__(self, prompt, **kw):
        self.calls.append(("raw", prompt, kw))
        return {"choices": [{"text": "raw respuesta"}]}


def test_uses_chat_template_and_ollama_sampling():
    m = FakeLlama()
    assert lc.LlamaCppProvider().generate(m, m, "Pregunta", max_new_tokens=50, temperature=0.0) == "respuesta"
    kind, msgs, kw = m.calls[0]
    assert kind == "chat" and msgs == [{"role": "user", "content": "Pregunta"}]
    assert kw["repeat_penalty"] == 1.1 and kw["top_k"] == 40 and kw["top_p"] == 0.9 and kw["max_tokens"] == 50


def test_raw_completion_without_template():
    m = FakeLlama(template=False)
    assert lc.LlamaCppProvider().generate(m, m, "p") == "raw respuesta"


def test_context_overflow_is_a_clear_error():
    m = FakeLlama(n_ctx=10)
    with pytest.raises(ValueError, match="context window"):
        lc.LlamaCppProvider().generate(m, m, "una dos tres cuatro cinco seis", max_new_tokens=8)


def test_defaults():
    assert lc.DEFAULT_N_CTX >= 8192
    assert lc._default_threads() >= 1
