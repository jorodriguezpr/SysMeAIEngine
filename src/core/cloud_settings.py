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

"""Persistent settings for optional cloud AI provider use (Anthropic, OpenAI,
Ollama Cloud) -- stored the same way the model registry is (a plain JSON
file next to registry.json), consistent with this project's existing
local-single-user threat model (registry.json and ENGINE_API_KEY are also
plaintext on disk, not encrypted)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state import EngineState

DEFAULT_MAX_TOKENS = 10000

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "provider": "anthropic",  # "anthropic" | "openai" | "ollama_cloud"
    "model": "",
    "api_key": "",
    "base_url": "",  # optional override; empty = provider's own default
    # prepended as a system-role instruction on every cloud call -- this is
    # the "prompt customization" hook, e.g. "Always answer in Spanish" or
    # "You are a theology research assistant."
    "system_prompt": "",
    "temperature": 0.2,
    # 10000, not 512: reasoning models (glm-5.3-flash) spend part of the
    # budget "thinking" before writing the answer, and small budgets
    # measurably came back EMPTY (200 always; 1024 on a real grounded chat
    # turn). A cap only limits usage -- billing is for tokens actually used.
    "max_tokens": DEFAULT_MAX_TOKENS,
}


def _settings_file(cfg) -> Path:
    return cfg.models_dir / "cloud_settings.json"


def get_cloud_settings(state: EngineState) -> dict[str, Any]:
    f = _settings_file(state.config)
    if not f.exists():
        return dict(DEFAULTS)
    try:
        saved = json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULTS)
    return {**DEFAULTS, **saved}


def save_cloud_settings(state: EngineState, **fields: Any) -> dict[str, Any]:
    current = get_cloud_settings(state)
    for key, value in fields.items():
        if key in DEFAULTS and value is not None:
            current[key] = value
    f = _settings_file(state.config)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    return current


def masked(settings: dict[str, Any]) -> dict[str, Any]:
    """Never send the real key back to the browser once saved -- only enough
    of it to confirm one is set."""
    out = dict(settings)
    key = out.get("api_key") or ""
    out["api_key"] = ("*" * max(0, len(key) - 4)) + key[-4:] if key else ""
    out["api_key_set"] = bool(key)
    return out
