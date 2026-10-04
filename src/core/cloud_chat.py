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

"""Calls out to an optional cloud AI provider (Anthropic, OpenAI, or Ollama
Cloud) for generation, using plain `requests` the same way this project's
existing `providers/ollama.py` calls a remote backend -- no new SDK
dependencies.

Ollama Cloud (verified: https://ollama.com/api and https://ollama.com/v1,
Bearer-token auth from an ollama.com API key) exposes the same
OpenAI-compatible chat/completions shape OpenAI itself does, just at a
different base URL -- so both share one implementation here.
"""
from __future__ import annotations

from typing import Any, Optional

import requests

from .cloud_settings import DEFAULT_MAX_TOKENS

ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
OLLAMA_CLOUD_DEFAULT_BASE_URL = "https://ollama.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


def _anthropic_chat(settings: dict[str, Any], prompt: str, max_tokens: Optional[int] = None) -> str:
    base_url = settings.get("base_url") or ANTHROPIC_DEFAULT_BASE_URL
    body: dict[str, Any] = {
        "model": settings["model"],
        "max_tokens": int(max_tokens or settings.get("max_tokens") or DEFAULT_MAX_TOKENS),
        "messages": [{"role": "user", "content": prompt}],
    }
    if settings.get("system_prompt"):
        body["system"] = settings["system_prompt"]
    r = _post(
        f"{base_url.rstrip('/')}/v1/messages",
        headers={
            "x-api-key": settings["api_key"],
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        json=body,
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"Anthropic API error {r.status_code}: {r.text[:500]}")
    data = r.json()
    return "".join(b.get("text", "") for b in data.get("content", []))


def _openai_compatible_chat(
    settings: dict[str, Any], prompt: str, default_base_url: str, max_tokens: Optional[int] = None,
) -> str:
    base_url = settings.get("base_url") or default_base_url
    messages = []
    if settings.get("system_prompt"):
        messages.append({"role": "system", "content": settings["system_prompt"]})
    messages.append({"role": "user", "content": prompt})
    r = _post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings["model"],
            "messages": messages,
            "temperature": float(settings.get("temperature") or 0.2),
            "max_tokens": int(max_tokens or settings.get("max_tokens") or DEFAULT_MAX_TOKENS),
        },
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"{default_base_url} API error {r.status_code}: {r.text[:500]}")
    data = r.json()
    # Reasoning models (e.g. glm-5.3-flash on Ollama Cloud) return
    # content "" -- or null -- when the whole max_tokens budget went to
    # their separate "reasoning" field. null must not leak out as None.
    return data["choices"][0]["message"].get("content") or ""


def _post(url: str, **kwargs: Any) -> requests.Response:
    """requests.post, with transport failures (connection refused, DNS,
    timeout...) re-raised as RuntimeError -- the one error type every
    caller of cloud_generate() already handles as "provider failed".
    Uncaught, requests' own exceptions (OSError subclasses) escaped every
    `except (ValueError, RuntimeError)` fallback as an HTTP 500 instead
    (live-verified: grounded /api/chat and /api/rag/chat with the network
    down, and a knowledge-graph build aborting on one timeout)."""
    try:
        return requests.post(url, **kwargs)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"cloud provider request failed: {type(e).__name__}: {e}") from e


def cloud_generate(settings: dict[str, Any], prompt: str, max_tokens: Optional[int] = None) -> str:
    """`max_tokens` overrides the saved settings value for just this one
    call -- for interactive chat/RAG use, the user's own configured budget
    is right; for batch use cases like generating several QA pairs from one
    passage (see cloud_distill.py), that budget is often too tight and the
    response gets cut off mid-answer, which is worse than just a short
    answer -- it silently produces truncated, broken training examples."""
    if not settings.get("enabled"):
        raise ValueError("cloud provider is not enabled in settings")
    if not settings.get("api_key"):
        raise ValueError(f"no API key configured for provider '{settings.get('provider')}'")
    if not settings.get("model"):
        raise ValueError("no model configured for the cloud provider")

    text = _dispatch(settings, prompt, max_tokens)
    budget = int(max_tokens or settings.get("max_tokens") or DEFAULT_MAX_TOKENS)
    if not text.strip() and budget < EMPTY_RETRY_MAX_TOKENS:
        # Reasoning models (glm-5.3-flash) spend the budget in a separate
        # "reasoning" field before writing the answer; when it runs out
        # first, the answer is "" with HTTP 200. Live-verified on a real
        # grounded chat turn (~16K-char prompt, settings max_tokens=1024):
        # a blank reply. One retry with a larger budget instead of handing
        # an empty answer back as if it were one.
        text = _dispatch(settings, prompt, EMPTY_RETRY_MAX_TOKENS)
    return text


# Retry at the default budget: only fires when a caller (or the saved
# settings) asked for less than that and got nothing back.
EMPTY_RETRY_MAX_TOKENS = DEFAULT_MAX_TOKENS


def _dispatch(settings: dict[str, Any], prompt: str, max_tokens: Optional[int]) -> str:
    provider = settings.get("provider")
    if provider == "anthropic":
        return _anthropic_chat(settings, prompt, max_tokens)
    if provider == "openai":
        return _openai_compatible_chat(settings, prompt, OPENAI_DEFAULT_BASE_URL, max_tokens)
    if provider == "ollama_cloud":
        return _openai_compatible_chat(settings, prompt, OLLAMA_CLOUD_DEFAULT_BASE_URL, max_tokens)
    raise ValueError(f"unknown cloud provider '{provider}' (expected anthropic, openai, or ollama_cloud)")
