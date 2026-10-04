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

"""llama.cpp GGUF provider via the `llama-cpp-python` package (optional) --
runs a quantized model INSIDE the engine process, no Ollama daemon needed.

Register with the .gguf file as the path, e.g. the Qwen2.5-7B Q4_K_M file
Ollama downloaded (hard-linked into models/gguf/, no extra disk):

    python -m src.cli model register --name qwen7b-cpp --provider llama_cpp \
        --path models/gguf/qwen2.5-7b-instruct-q4_k_m.gguf

Defaults mirror what Ollama does with the same file, so answers are
comparable: the model's own chat template (embedded in the GGUF), an
8192-token context, and Ollama's sampling defaults (repeat penalty 1.1,
top-k 40, top-p 0.9).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# The previous default (2048) is too small for a grounded prompt (~1.2K
# tokens of passages + instructions) plus a complete answer (350 tokens):
# llama.cpp would reject the request outright.
DEFAULT_N_CTX = int(os.getenv("ENGINE_LLAMA_N_CTX", "8192"))
# Ollama's defaults for the same model -- keeps the two routes comparable.
REPEAT_PENALTY = 1.1
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.9


def _default_threads() -> int:
    # llama.cpp is fastest with one thread per PHYSICAL core; os.cpu_count()
    # counts SMT threads (16 on an 8-core Ryzen AI 7 350)
    env = os.getenv("ENGINE_LLAMA_THREADS")
    if env:
        return int(env)
    return max(1, (os.cpu_count() or 2) // 2)


class LlamaCppProvider:
    name = "llama_cpp"

    def load(self, name: str, registry_path: Path, **kwargs: Any) -> tuple[Any, Any]:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "llama-cpp-python is not installed. `pip install llama-cpp-python "
                "--only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`"
            ) from e
        from ..device_select import resolve_hf_device_map  # lazy

        model_path = kwargs.get("path")
        if not model_path:
            raise ValueError("llama_cpp provider requires a .gguf path")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"GGUF model file not found: {model_path}")
        # n_gpu_layers only actually offloads if the installed llama-cpp-python
        # wheel was itself built with CUDA (cuBLAS) support — a separate,
        # platform-specific build step (see requirements-cuda.txt), unlike the
        # HF provider where CUDA support is already inside the standard torch
        # wheel. On a CPU-only build this is silently ignored, not an error.
        device_map = resolve_hf_device_map(kwargs.get("device_map", "auto"))
        n_gpu_layers = kwargs.get("n_gpu_layers")
        if n_gpu_layers is None:
            n_gpu_layers = -1 if device_map == "cuda" else 0
        llm = Llama(
            model_path=str(model_path),
            n_ctx=kwargs.get("n_ctx") or DEFAULT_N_CTX,
            n_threads=kwargs.get("n_threads") or _default_threads(),
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        return llm, llm  # self-contained: .tokenize() / .detokenize()

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        n_ctx = model.n_ctx()
        prompt_tokens = len(model.tokenize(prompt.encode("utf-8"), add_bos=False))
        if prompt_tokens + max_new_tokens > n_ctx:
            raise ValueError(
                f"prompt ({prompt_tokens} tokens) + answer budget ({max_new_tokens}) exceeds the "
                f"llama.cpp context window ({n_ctx}); raise ENGINE_LLAMA_N_CTX"
            )
        sampling = dict(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                        top_k=top_k, repeat_penalty=REPEAT_PENALTY)
        if model.metadata.get("tokenizer.chat_template"):
            # Same as Ollama: wrap the prompt in the model's own chat template
            # (instruct models answer much worse from a raw completion prompt).
            msgs = messages or [{"role": "user", "content": prompt}]
            out = model.create_chat_completion(messages=msgs, **sampling)
            return out["choices"][0]["message"]["content"] or ""
        out = model(prompt, **sampling)
        return out["choices"][0]["text"]

    def embed(
        self, model: Any, tokenizer: Any, text: str | list[str]
    ) -> list[list[float]]:
        texts = [text] if isinstance(text, str) else text
        return [model.embed(t) for t in texts]

    def info(self, model: Any) -> dict[str, Any]:
        return {"kind": "llama.cpp", "n_ctx": model.n_ctx()}
