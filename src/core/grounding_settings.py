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

"""Grounding-layer settings for chat: retrieve-before-generate ("RAG chat
mode") and the knowledge-graph retriever.

Stored like cloud_settings.py — a plain JSON file next to registry.json,
same local-single-user threat model.

chat: "grounded" mode retrieves passages from the local RAG indexes (and,
if built, the knowledge graph) and injects them into the prompt before any
generation happens, so answers come from the ingested documents instead of
the model's weights. This is the P0 fix for "chat can't answer from my
documents" — /api/chat previously generated directly from weights with
zero retrieval.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state import EngineState

DEFAULTS: dict[str, Any] = {
    # /api/chat behavior: "plain" (weights only, the old behavior) or
    # "grounded" (retrieve → augment → generate).
    "chat_mode": "plain",
    # number of passages to inject per chat turn
    "chat_top_k": 6,
    # use multi-query expansion + parent-child context expansion (P1) when
    # retrieving for a chat turn. Costs one extra cloud call per question
    # (query rewrites) and reranker work, but bridges the phrasing gap
    # between user questions and book prose.
    # Off by default: measured with scripts/eval_grounded.py (2 runs x 6
    # configs, aprocrifos-v2): zero context-recall gain, same accuracy as
    # baseline, retrieval 6.5s vs 3.2s -- and it sends every question to
    # the cloud provider. See docs/GROUNDING.md "Measured feature value".
    "chat_multi_query": False,
    # inject the knowledge-graph passages (entity/path context) when a graph
    # has been built for the corpus being searched
    # Off by default: same measurement -- no recall or accuracy gain, and
    # graph rows displace a retrieved passage from the context. The graph
    # stays available; turn this on per corpus if a re-run shows a gain.
    "chat_kgraph": False,
    # cloud model is the answer *writer* in grounded mode when enabled
    # (falls back to the local model when the provider is disabled/misconfigured)
    "grounded_use_cloud": True,
    # refuse instead of answering when retrieval returns nothing usable
    "grounded_refuse_when_empty": True,
    # prompt used when a LOCAL model writes the grounded answer. Measured
    # (scripts/eval_grounded.py, aprocrifos-v2, local qwen2.5-7b, top-3):
    # "complete" 0.67 accuracy / 5% wrong-when-answering vs "native" (short)
    # 0.44 / 10%; English "default" instructions were worse still for
    # Spanish questions with small models. See docs/GROUNDING.md.
    "local_prompt_style": "complete",
    # reserve one retrieval slot for the knowledge-graph walk
    # (src/core/graph_retrieval.py) -- bridges paraphrased / multi-step
    # questions to passages with no shared words. Needs a built graph for
    # the corpus; without one it's a no-op.
    "chat_graph_slot": True,
}

_MODES = ("plain", "grounded")
MAX_TOP_K = 20


def _settings_file(cfg) -> Path:
    return cfg.models_dir / "grounding_settings.json"


def get_grounding_settings(state: EngineState) -> dict[str, Any]:
    f = _settings_file(state.config)
    if not f.exists():
        return dict(DEFAULTS)
    try:
        saved = json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULTS)
    return {**DEFAULTS, **saved}


def save_grounding_settings(state: EngineState, **fields: Any) -> dict[str, Any]:
    current = get_grounding_settings(state)
    for key, value in fields.items():
        if key in DEFAULTS and value is not None:
            if key == "chat_mode" and value not in _MODES:
                raise ValueError(f"chat_mode must be one of {_MODES}")
            if key == "local_prompt_style" and value not in ("default", "native", "complete"):
                raise ValueError("local_prompt_style must be default, native or complete")
            if key == "chat_top_k" and not (1 <= int(value) <= MAX_TOP_K):
                # a negative value used to be saved as-is, and hits[:top_k]
                # then silently meant "all but the last N"
                raise ValueError(f"chat_top_k must be between 1 and {MAX_TOP_K}")
            current[key] = value
    f = _settings_file(state.config)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    return current