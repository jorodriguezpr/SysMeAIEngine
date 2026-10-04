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

"""Tool-calling support for the OpenAI/Anthropic-compatible chat endpoints.

Local instruction-tuned models (Qwen2.5-Instruct and similar) don't have a
separate structured tool-call channel the way hosted OpenAI/Anthropic APIs do —
their chat template renders a `<tool_call>{"name": ..., "arguments": {...}}
</tool_call>` block (the Hermes format) straight into the generated text.
This module parses that format, and transparently executes the engine's own
built-in tools (currently just local RAG search) so callers get back a
normal grounded answer instead of having to implement tool execution
themselves. Any tool call the caller's own `tools` list declares (that isn't
one of ours) is handed back to them unexecuted, exactly like a real
OpenAI/Anthropic API would.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from . import rag as rag_mod
from .model_loader import chat as core_chat
from .state import EngineState

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

MAX_TOOL_ROUNDS = 3

BUILTIN_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_local_knowledge",
            "description": (
                "Search the engine's local RAG corpora (documents trained "
                "into local models via Train-from-Folder or the "
                "self-learning agent) for passages relevant to a query. "
                "Call this before answering questions about locally "
                "trained content instead of guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "corpus": {
                        "type": "string",
                        "description": "Corpus name, or omit to search all corpora.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]
_BUILTIN_NAMES = {t["function"]["name"] for t in BUILTIN_TOOLS}


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    """Plain-text fallback for models with no chat template."""
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if m.get("tool_calls"):
            calls = "; ".join(
                f"{c['function']['name']}({c['function']['arguments']})"
                for c in m["tool_calls"]
            )
            content = f"{content} [tool_calls: {calls}]".strip()
        lines.append(f"<{role}>: {content}")
    lines.append("<assistant>:")
    return "\n".join(lines)


def extract_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append({"name": obj["name"], "arguments": obj.get("arguments") or {}})
    clean = _TOOL_CALL_RE.sub("", text).strip()
    return clean, calls


def run_builtin_tool(state: EngineState, name: str, arguments: dict[str, Any]) -> Any:
    if name == "search_local_knowledge":
        hits = rag_mod.retrieve(
            state, arguments.get("query", ""), 5, corpus=arguments.get("corpus") or "all"
        )
        return {"hits": hits}
    raise KeyError(f"'{name}' is not a built-in tool")


def run_chat_with_tools(
    state: EngineState,
    messages: list[dict[str, Any]],
    model_name: Optional[str],
    caller_tools: Optional[list[dict[str, Any]]],
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Runs the chat/tool-call loop, auto-executing built-in tools and only
    handing pending tool calls back to the caller when a model asks for a
    tool the engine itself doesn't own.

    Returns {"model", "text", "tool_calls", "prompt_tokens", "completion_tokens",
    "citations"}. `tool_calls` is non-empty only when the caller must execute
    one itself and send the result back in a follow-up request. `citations`
    collects {corpus, source, id} for every passage any auto-executed
    search_local_knowledge call retrieved -- the model sees the full hit
    (text, score) to answer from, but the caller gets back which documents
    were actually used, independent of whether the model's own prose
    mentions them (it often doesn't, reliably).
    """
    caller_names = {t["function"]["name"] for t in (caller_tools or [])}
    all_tools = list(caller_tools or []) + [
        t for t in BUILTIN_TOOLS if t["function"]["name"] not in caller_names
    ]

    convo = list(messages)
    prompt_tokens = completion_tokens = 0
    model_used = model_name
    clean_text = ""
    citations: list[dict[str, Any]] = []
    seen_citation_keys: set[tuple[str, str]] = set()

    for _round in range(MAX_TOOL_ROUNDS):
        out = core_chat(
            state,
            flatten_messages(convo),
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            messages=convo,
            tools=all_tools,
        )
        model_used = out["model"]
        prompt_tokens += out["prompt_tokens"]
        completion_tokens += out["completion_tokens"]
        clean_text, calls = extract_tool_calls(out["text"])

        if not calls:
            break

        pending = [c for c in calls if c["name"] not in _BUILTIN_NAMES]
        if pending:
            return {
                "model": model_used,
                "text": clean_text or None,
                "tool_calls": calls,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "citations": citations,
            }

        call_ids = [f"call_{i}" for i in range(len(calls))]
        convo.append(
            {
                "role": "assistant",
                "content": clean_text,
                "tool_calls": [
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
                    }
                    for cid, c in zip(call_ids, calls)
                ],
            }
        )
        for cid, c in zip(call_ids, calls):
            try:
                result = run_builtin_tool(state, c["name"], c["arguments"])
            except Exception as e:
                result = {"error": str(e)}
            for hit in result.get("hits", []):
                key = (hit["corpus"], hit["id"])
                if key not in seen_citation_keys:
                    seen_citation_keys.add(key)
                    citations.append({"corpus": hit["corpus"], "source": hit["source"], "id": hit["id"]})
            convo.append(
                {"role": "tool", "tool_call_id": cid, "content": json.dumps(result, ensure_ascii=False)}
            )

    return {
        "model": model_used,
        "text": clean_text,
        "citations": citations,
        "tool_calls": [],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
