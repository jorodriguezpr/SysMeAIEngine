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

"""OpenAI-compatible + Anthropic Messages API compatible endpoints.

Shapes implemented:
  • POST /v1/chat/completions      (OpenAI, non-streaming + SSE optional)
  • POST /v1/completions
  • POST /v1/embeddings
  • GET  /v1/models
  • POST /anthropic/v1/messages    (Anthropic Messages shape, minimal)
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..core import metrics as M
from ..core import toolcalls
from ..core.model_loader import chat as core_chat
from ..core.model_loader import embeddings as core_embed
from ..core.model_loader import list_models
from ..core.state import EngineState
from .chat_routes import get_state, require_auth

router = APIRouter()


# ── helpers ─────────────────────────────────────────────────────────────────
def _openai_tool_call_payload(result: dict[str, Any]) -> dict:
    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
        }
        for i, c in enumerate(result["tool_calls"])
    ]
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result["model"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"],
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }
    # A built-in RAG call could have already run in an earlier round before
    # this caller-owned tool call was requested -- see _openai_chat_payload.
    if result.get("citations"):
        payload["citations"] = result["citations"]
    return payload


def _openai_chat_payload(
    model: str, completion: str, in_t: int, out_t: int, citations: Optional[list[dict]] = None,
) -> dict:
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": in_t,
            "completion_tokens": out_t,
            "total_tokens": in_t + out_t,
        },
    }
    # Non-standard extra field (OpenAI clients ignore unknown top-level keys)
    # -- which documents search_local_knowledge actually retrieved, since
    # the model's own prose doesn't reliably mention them. Omitted entirely
    # when empty rather than sent as [] on every response.
    if citations:
        payload["citations"] = citations
    return payload


# ── /v1/chat/completions ────────────────────────────────────────────────────
class ChatCompletionIn(BaseModel):
    model: Optional[str] = None
    messages: list[dict[str, Any]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Any] = None


@router.post("/v1/chat/completions")
def chat_completions(
    body: ChatCompletionIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    from ..core.calculator import try_calculate

    last_user = next(
        (m.get("content", "") for m in reversed(body.messages) if m.get("role") == "user"),
        "",
    )
    calc = try_calculate(last_user if isinstance(last_user, str) else "")
    if calc is not None:
        return JSONResponse(_openai_chat_payload("calculator", calc, 0, 0))

    result = toolcalls.run_chat_with_tools(
        state,
        body.messages,
        body.model,
        body.tools,
        max_new_tokens=body.max_tokens or 256,
        temperature=body.temperature if body.temperature is not None else 0.7,
    )

    if result["tool_calls"]:
        # a tool the caller declared (not one of ours) needs to run on their
        # side — hand it back exactly like a real OpenAI API would, no SSE.
        return JSONResponse(_openai_tool_call_payload(result))

    if body.stream:
        # very small SSE stream — delivers the whole completion in one delta event.
        # True token-by-token streaming requires provider support; tracked in roadmap.
        def _gen():
            yield "data: " + json.dumps(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": result["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": result["text"]},
                            "finish_reason": None,
                        }
                    ],
                }
            ) + "\n\n"
            yield "data: " + json.dumps(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": result["model"],
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            ) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    return JSONResponse(
        _openai_chat_payload(
            result["model"], result["text"], result["prompt_tokens"], result["completion_tokens"],
            citations=result.get("citations"),
        )
    )


# ── /v1/completions ─────────────────────────────────────────────────────────
class CompletionIn(BaseModel):
    model: Optional[str] = None
    prompt: str | list[str]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


@router.post("/v1/completions")
def completions(
    body: CompletionIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    prompts = [body.prompt] if isinstance(body.prompt, str) else body.prompt
    choices = []
    total_in = total_out = 0
    for i, p in enumerate(prompts):
        out = core_chat(
            state,
            p,
            model_name=body.model,
            max_new_tokens=body.max_tokens or 256,
            temperature=body.temperature if body.temperature is not None else 0.7,
        )
        choices.append({"index": i, "text": out["text"], "finish_reason": "stop"})
        total_in += out["prompt_tokens"]
        total_out += out["completion_tokens"]
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": body.model or state.config.default_model,
        "choices": choices,
        "usage": {
            "prompt_tokens": total_in,
            "completion_tokens": total_out,
            "total_tokens": total_in + total_out,
        },
    }


# ── /v1/embeddings ──────────────────────────────────────────────────────────
class EmbeddingsIn(BaseModel):
    model: Optional[str] = None
    input: str | list[str]


@router.post("/v1/embeddings")
def embeddings(
    body: EmbeddingsIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    result = core_embed(state, body.input, model_name=body.model)
    data = [
        {"object": "embedding", "index": i, "embedding": vec}
        for i, vec in enumerate(result["data"])
    ]
    return {
        "object": "list",
        "model": result["model"],
        "data": data,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


# ── /v1/models ──────────────────────────────────────────────────────────────
@router.get("/v1/models")
def list_openai_models(state: EngineState = Depends(get_state)):
    reg = list_models(state)
    return {
        "object": "list",
        "data": [
            {
                "id": m["name"],
                "object": "model",
                "created": 0,
                "owned_by": m["provider"],
            }
            for m in reg["models"]
        ],
    }


# ── Anthropic Messages style ────────────────────────────────────────────────────────────
class AnthropicMessageIn(BaseModel):
    model: Optional[str] = None
    max_tokens: int = 256
    messages: list[dict[str, Any]]
    system: Optional[str] = None
    temperature: Optional[float] = None
    tools: Optional[list[dict[str, Any]]] = None


def _anthropic_tools_to_internal(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        }
        for t in tools
    ]


def _anthropic_messages_to_internal(
    system: Optional[str], messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Anthropic content blocks (text / tool_use / tool_result) -> the
    OpenAI-ish role/content(+tool_calls) shape run_chat_with_tools expects."""
    internal: list[dict[str, Any]] = []
    if system:
        internal.append({"role": "system", "content": system})
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not isinstance(content, list):
            internal.append({"role": role, "content": content})
            continue

        tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
        if tool_results:
            for b in tool_results:
                tr_content = b.get("content", "")
                if isinstance(tr_content, list):
                    tr_content = " ".join(
                        p.get("text", "") for p in tr_content if isinstance(p, dict)
                    )
                internal.append(
                    {"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": tr_content}
                )
            continue

        tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
        text = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        if tool_uses:
            internal.append(
                {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": [
                        {
                            "id": b.get("id"),
                            "type": "function",
                            "function": {"name": b.get("name"), "arguments": json.dumps(b.get("input", {}))},
                        }
                        for b in tool_uses
                    ],
                }
            )
        else:
            internal.append({"role": role, "content": text})
    return internal


@router.post("/anthropic/v1/messages")
def anthropic_messages(
    body: AnthropicMessageIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    messages = _anthropic_messages_to_internal(body.system, body.messages)
    caller_tools = _anthropic_tools_to_internal(body.tools) if body.tools else None

    result = toolcalls.run_chat_with_tools(
        state,
        messages,
        body.model,
        caller_tools,
        max_new_tokens=body.max_tokens,
        temperature=body.temperature if body.temperature is not None else 0.7,
    )

    if result["tool_calls"]:
        content: list[dict[str, Any]] = []
        if result["text"]:
            content.append({"type": "text", "text": result["text"]})
        content.extend(
            {"type": "tool_use", "id": f"toolu_{i}", "name": c["name"], "input": c["arguments"]}
            for i, c in enumerate(result["tool_calls"])
        )
        response = {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": result["model"],
            "content": content,
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": result["prompt_tokens"], "output_tokens": result["completion_tokens"]},
        }
        if result.get("citations"):
            response["citations"] = result["citations"]
        return response

    response = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": result["model"],
        "content": [{"type": "text", "text": result["text"]}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": result["prompt_tokens"], "output_tokens": result["completion_tokens"]},
    }
    # Non-standard extra field, same rationale as the OpenAI-shaped payloads.
    if result.get("citations"):
        response["citations"] = result["citations"]
    return response
