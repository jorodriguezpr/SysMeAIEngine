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

"""Local Ollama provider: the registry `path` (Ollama tag) must be what's
sent, and num_ctx must be set -- Ollama's default window silently cuts
long grounded prompts."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.providers import ollama as ollama_mod  # noqa: E402


def test_load_uses_registry_path_as_model_tag():
    model, _ = ollama_mod.OllamaProvider().load("qwen7b-local", Path("x"), path="qwen2.5:7b-instruct")
    assert model["model"] == "qwen2.5:7b-instruct"


def test_load_falls_back_to_name_without_path():
    model, _ = ollama_mod.OllamaProvider().load("phi3", Path("x"))
    assert model["model"] == "phi3"


def test_generate_sends_num_ctx_and_tag():
    resp = MagicMock(); resp.json.return_value = {"response": "hola"}
    with patch.object(ollama_mod.requests, "post", return_value=resp) as post:
        out = ollama_mod.OllamaProvider().generate(
            {"base_url": "http://x", "model": "qwen2.5:7b-instruct"}, None, "p", max_new_tokens=5, temperature=0.0)
    assert out == "hola"
    body = post.call_args.kwargs["json"]
    assert body["model"] == "qwen2.5:7b-instruct"
    assert body["options"]["num_ctx"] >= 8192
    assert post.call_args.kwargs["timeout"] >= 600


def test_generate_disables_thinking_by_default():
    # granite4.2 writes its English reasoning (ending in </think>) into the
    # answer unless "think": false is sent.
    resp = MagicMock(); resp.json.return_value = {"response": "La capital es París."}
    with patch.object(ollama_mod.requests, "post", return_value=resp) as post:
        ollama_mod.OllamaProvider().generate({"base_url": "http://x", "model": "granite4.2:3b"}, None, "p")
    assert post.call_args.kwargs["json"]["think"] is False
