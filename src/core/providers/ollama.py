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

"""Ollama provider -- answer with a model served by a LOCAL Ollama daemon
(fully offline once the model is pulled), or point it at another Ollama-
compatible backend. Register with the Ollama model tag as the path:

    python -m src.cli model register --name qwen7b-local --provider ollama --path qwen2.5:7b-instruct

Note: Ollama tags ending in ":cloud" run on Ollama's servers, not locally.
"""
from __future__ import annotations

import os
from typing import Any

import requests

# Ollama's own default context window (2048-4096 tokens depending on
# version) silently truncates long prompts from the front -- a grounded
# prompt with retrieved passages is ~1.2K tokens compact and ~5K widened,
# so the instructions/passages would be cut without any error.
DEFAULT_NUM_CTX = int(os.getenv("ENGINE_OLLAMA_NUM_CTX", "8192"))
# CPU inference of a 7B model on a multi-thousand-token prompt can take
# minutes; 300s was too short to be safe.
TIMEOUT_S = int(os.getenv("ENGINE_OLLAMA_TIMEOUT_S", "900"))
# Reasoning models (e.g. granite4.2) otherwise write their English "thinking"
# into `response`, ending in </think>, ahead of the answer. Non-thinking
# models (qwen2.5) accept the flag and ignore it.
THINK = os.getenv("ENGINE_OLLAMA_THINK", "false").lower() in ("1", "true", "yes")


class OllamaProvider:
    name = "ollama"

    def __init__(self) -> None:
        self.base_url: str = os.getenv("ENGINE_OLLAMA_URL", "http://127.0.0.1:11434")

    def load(self, name: str, registry_path, **kwargs: Any) -> tuple[Any, Any]:
        self.base_url = kwargs.get("base_url", self.base_url)
        # The registry's `path` holds the Ollama model tag; the registered
        # name is only the engine-side label. (Previously the label was sent
        # to Ollama, so any name differing from the tag could never load.)
        return {"base_url": self.base_url, "model": kwargs.get("path") or name}, None

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs: Any,
    ) -> str:
        r = requests.post(
            f"{model['base_url']}/api/generate",
            json={
                "model": model["model"],
                "prompt": prompt,
                "stream": False,
                "think": THINK,
                "options": {
                    "num_predict": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "num_ctx": DEFAULT_NUM_CTX,
                },
            },
            timeout=TIMEOUT_S,
        )
        r.raise_for_status()
        return r.json()["response"]

    def embed(
        self, model: Any, tokenizer: Any, text: str | list[str]
    ) -> list[list[float]]:
        texts = [text] if isinstance(text, str) else text
        out = []
        for t in texts:
            r = requests.post(
                f"{model['base_url']}/api/embeddings",
                json={"model": model["model"], "prompt": t},
                timeout=TIMEOUT_S,
            )
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return out

    def info(self, model: Any) -> dict[str, Any]:
        return {"kind": "ollama", "base_url": model["base_url"], "model": model["model"]}
