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

"""Model registry + load/unload/get + chat plumbing."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from .metrics import record_chat
from .providers.registry import available_providers, load_provider
from .state import EngineState, LoadedModel


def _registry_file(cfg) -> Path:
    return cfg.models_dir / "registry.json"


def _read_registry(cfg) -> list[dict]:
    f = _registry_file(cfg)
    if not f.exists():
        return []
    return json.loads(f.read_text(encoding="utf-8"))


def _write_registry(cfg, entries: list[dict]) -> None:
    _registry_file(cfg).write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def register_model(
    state: EngineState,
    name: str,
    provider: str,
    path: str,
    rag_corpus: Optional[str] = None,
    device: Optional[str] = None,
) -> dict:
    if provider not in available_providers():
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {available_providers()}"
        )
    entries = _read_registry(state.config)
    if any(e["name"] == name for e in entries):
        raise ValueError(f"Model '{name}' is already registered")
    entry = {
        "name": name,
        "provider": provider,
        "path": path,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "size_bytes": None,
        "params": None,
        # 'auto' | 'cpu' | 'cuda' — resolved per-provider at load time (see
        # device_select.resolve_hf_device_map). Defaults to the engine-wide
        # setting so most models never need to think about this.
        "device": device or state.config.default_device,
        # which RAG corpus `/api/rag/chat` should use for this model when the
        # caller doesn't specify one explicitly — defaults to the model's own
        # name (the convention train_from_folder.py follows), but can point
        # at a different model's corpus (e.g. a RAG-only model with no
        # fine-tuning of its own, grounded in an existing corpus instead)
        "rag_corpus": rag_corpus,
    }
    entries.append(entry)
    _write_registry(state.config, entries)
    return entry


def get_model_entry(state: EngineState, name: str) -> Optional[dict]:
    return next((e for e in _read_registry(state.config) if e["name"] == name), None)


def set_rag_corpus(state: EngineState, name: str, rag_corpus: str) -> dict:
    entries = _read_registry(state.config)
    entry = next((e for e in entries if e["name"] == name), None)
    if not entry:
        raise ValueError(f"Model '{name}' not registered")
    entry["rag_corpus"] = rag_corpus
    _write_registry(state.config, entries)
    return entry


def unregister_model(state: EngineState, name: str) -> None:
    entries = _read_registry(state.config)
    new = [e for e in entries if e["name"] != name]
    if len(new) == len(entries):
        raise ValueError(f"Model '{name}' not found")
    _write_registry(state.config, new)


def list_models(state: EngineState) -> dict[str, Any]:
    return {
        "models": _read_registry(state.config),
        "loaded": state.loaded.name if state.loaded else None,
        "providers": available_providers(),
    }


def load_model(state: EngineState, name: str) -> LoadedModel:
    if state.loaded and state.loaded.name == name:
        return state.loaded  # idempotent
    entries = _read_registry(state.config)
    entry = next((e for e in entries if e["name"] == name), None)
    if not entry:
        raise ValueError(f"Model '{name}' not registered. Use `model register`.")
    provider_cls = load_provider(entry["provider"])
    provider = provider_cls()
    storage = state.config.weights_dir / name
    device = entry.get("device") or state.config.default_device
    model, tokenizer = provider.load(name, storage, path=entry.get("path"), device_map=device)
    loaded = LoadedModel(name=name, model=model, tokenizer=tokenizer, registry_entry=entry)
    state.loaded = loaded
    return loaded


def unload_model(state: EngineState) -> None:
    state.loaded = None


def get_loaded(state: EngineState) -> Optional[LoadedModel]:
    return state.loaded


def chat(
    state: EngineState,
    prompt: str,
    model_name: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    messages: Optional[list[dict[str, Any]]] = None,
    tools: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    name = model_name or state.config.default_model
    if not state.loaded or state.loaded.name != name:
        load_model(state, name)
    loaded = state.loaded
    assert loaded is not None
    provider_cls = load_provider(loaded.registry_entry["provider"])
    provider = provider_cls()
    t0 = time.perf_counter()
    text = provider.generate(
        loaded.model,
        loaded.tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        messages=messages,
        tools=tools,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # strip the prompt prefix when the provider echoes it
    completion = text[len(prompt):] if text.startswith(prompt) else text
    tokens_in = len(prompt) // 4 + 1
    tokens_out = len(completion) // 4 + 1
    record_chat(state, tokens_in, tokens_out, elapsed_ms)
    return {
        "model": name,
        "prompt_tokens": tokens_in,
        "completion_tokens": tokens_out,
        "elapsed_ms": round(elapsed_ms, 1),
        "text": completion,
    }


def embeddings(
    state: EngineState, text: str | list[str], model_name: Optional[str] = None
) -> dict[str, Any]:
    name = model_name or state.config.default_model
    if not state.loaded or state.loaded.name != name:
        load_model(state, name)
    loaded = state.loaded
    assert loaded is not None
    provider_cls = load_provider(loaded.registry_entry["provider"])
    provider = provider_cls()
    vectors = provider.embed(loaded.model, loaded.tokenizer, text)
    from .metrics import record_embed

    tokens_in = sum(len(t) for t in ([text] if isinstance(text, str) else text)) // 4 + 1
    record_embed(state, tokens_in)
    return {"model": name, "dimension": len(vectors[0]) if vectors else 0, "data": vectors}


def tensor_stats(state: EngineState) -> list[dict]:
    if not state.loaded:
        load_model(state, state.config.default_model)
    loaded = state.loaded
    assert loaded is not None
    model = loaded.model
    if hasattr(model, "tensor_stats"):
        return model.tensor_stats()
    # HF torch model fallback
    import torch  # lazy

    stats = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            data = p.detach().float().cpu()
            stats.append(
                {
                    "name": name,
                    "shape": list(p.shape),
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "sparsity": float((data.abs() < 1e-8).float().mean()),
                }
            )
    return stats
