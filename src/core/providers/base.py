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

"""Provider interface — one implementation per model backend.

A provider is responsible for:
  • loading a model + tokenizer pair (or a self-contained model)
  • exposing `.generate()` and `.embed()` callables
  • exposing `info()` metadata used by the dashboard
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Provider(Protocol):
    name: str  # unique provider key, e.g. "hf"

    def load(self, name: str, registry_path: Path, **kwargs: Any) -> tuple[Any, Any]:
        """Load (model, tokenizer). Tokenizer may be None for self-contained models."""
        ...

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        ...

    def embed(
        self, model: Any, tokenizer: Any, text: str | list[str]
    ) -> list[list[float]]:
        ...

    def info(self, model: Any) -> dict[str, Any]:
        ...
