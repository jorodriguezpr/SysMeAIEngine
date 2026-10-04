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

"""Provider registry — name → class, with lazy imports to keep startup fast."""
from __future__ import annotations

from typing import Type

from .base import Provider

_REGISTRY: dict[str, str] = {
    # name      → dotted path (lazy import)
    "minilm": "src.core.providers.minilm:MiniLMProvider",
    "hf": "src.core.providers.hf:HFProvider",
    "llama_cpp": "src.core.providers.llama_cpp:LlamaCppProvider",
    "onnx_directml": "src.core.providers.onnx_directml:ONNXDirectMLProvider",
    "onnx_npu": "src.core.providers.onnx_npu:ONNXXNPUProvider",
    "ollama": "src.core.providers.ollama:OllamaProvider",
    "sentence_embed": "src.core.providers.sentence_embed:SentenceEmbedProvider",
}


def available_providers() -> list[str]:
    return sorted(_REGISTRY)


def load_provider(name: str) -> Type[Provider]:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown provider '{name}'. Available: {available_providers()}")
    mod_path, cls_name = _REGISTRY[name].rsplit(":", 1)
    import importlib

    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)
