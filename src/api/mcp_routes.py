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

"""MCP (Model Context Protocol) server — JSON-RPC over HTTP.

Exposes engine capabilities as tools/resources/prompts so desktop AI
assistants, MCP-compatible IDEs, and other MCP clients can drive SysMeAIEngine.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..core import metrics as M
from ..core import rag as rag_mod
from ..core.model_loader import chat as core_chat
from ..core.model_loader import embeddings as core_embed
from ..core.model_loader import list_models, load_model
from ..core.state import EngineState
from .chat_routes import get_state, require_auth

router = APIRouter()

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "SysMeAIEngine", "version": "0.1.0"}
CAPABILITIES = {"tools": {}, "resources": {}, "prompts": {}}

# ── JSON-RPC plumbing ───────────────────────────────────────────────────────
class RpcIn(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Any] = None
    method: str
    params: Optional[dict[str, Any]] = None


def _ok(id_: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _err(id_: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


# ── Tools ───────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "engine.chat",
        "description": "Run a chat completion against a loaded model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "model": {"type": ["string", "null"]},
                "max_tokens": {"type": "integer", "default": 256},
                "temperature": {"type": "number", "default": 0.7},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "engine.embeddings",
        "description": "Compute embeddings for one or more texts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input": {"type": ["string", "array"]},
                "model": {"type": ["string", "null"]},
            },
            "required": ["input"],
        },
    },
    {
        "name": "engine.models",
        "description": "List registered and loaded models.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "engine.metrics",
        "description": "Return runtime metrics (CPU/GPU/NPU/tokens/etc).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "engine.load",
        "description": "Load a registered model by name.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "rag.search",
        "description": (
            "Search the engine's local RAG corpora (documents/text trained "
            "into local models via Train-from-Folder or the self-learning "
            "agent) for passages relevant to a query. Use this to ground "
            "answers in the engine's own local knowledge instead of guessing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "corpus": {
                    "type": ["string", "array", "null"],
                    "description": (
                        "A corpus name, a list of names, or 'all' to search "
                        "every corpus with a built index. Defaults to 'all'."
                    ),
                },
                "top_k": {"type": "integer", "default": 5},
                "method": {
                    "type": "string",
                    "enum": ["hybrid", "semantic", "tfidf"],
                    "default": "hybrid",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "rag.list_corpora",
        "description": "List local RAG corpora that have a built search index.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _first_registered_model(state: EngineState) -> Optional[str]:
    models = list_models(state)["models"]
    return models[0]["name"] if models else None


def _resolve_model(state: EngineState, requested: Optional[str]) -> Optional[str]:
    if requested:
        return requested
    if state.config.default_model and any(
        m["name"] == state.config.default_model
        for m in list_models(state)["models"]
    ):
        return state.config.default_model
    return _first_registered_model(state)


def _call_tool(state: EngineState, name: str, args: dict[str, Any]) -> Any:
    if name == "engine.chat":
        model = _resolve_model(state, args.get("model"))
        if model is None:
            raise ValueError("no model registered — register one first")
        return core_chat(
            state,
            args["prompt"],
            model_name=model,
            max_new_tokens=args.get("max_tokens", 256),
            temperature=args.get("temperature", 0.7),
        )
    if name == "engine.embeddings":
        model = _resolve_model(state, args.get("model"))
        if model is None:
            raise ValueError("no model registered — register one first")
        return core_embed(state, args["input"], model_name=model)
    if name == "engine.models":
        return list_models(state)
    if name == "engine.metrics":
        return M.snapshot(state)
    if name == "engine.load":
        load_model(state, args["name"])
        return {"loaded": args["name"]}
    if name == "rag.search":
        hits = rag_mod.retrieve(
            state,
            args["query"],
            args.get("top_k", 5),
            method=args.get("method", rag_mod.DEFAULT_METHOD),
            corpus=args.get("corpus") or "all",
        )
        return {"hits": hits}
    if name == "rag.list_corpora":
        # "hybrid" isn't a stored index kind — corpora are indexed under
        # semantic/ or tfidf/, and build_index always builds semantic when
        # asked for hybrid, so that's the enumerable marker to check.
        return {"corpora": rag_mod.list_corpora(state, method="semantic")}
    raise ValueError(f"unknown tool '{name}'")


# ── Resources ───────────────────────────────────────────────────────────────
def _resources(state: EngineState) -> list[dict[str, Any]]:
    return [
        {"uri": "engine://models", "mimeType": "application/json", "name": "model registry"},
        {"uri": "engine://loaded", "mimeType": "application/json", "name": "loaded model"},
        {"uri": "engine://metrics", "mimeType": "application/json", "name": "metrics snapshot"},
        {"uri": "engine://config", "mimeType": "application/json", "name": "engine config"},
        {"uri": "engine://agent/state", "mimeType": "application/json", "name": "self-learning agent state"},
        {
            "uri": "engine://datasets",
            "mimeType": "application/json",
            "name": "available datasets",
        },
    ]


def _read_resource(state: EngineState, uri: str) -> dict[str, Any]:
    cfg = state.config
    if uri == "engine://models":
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": __import__("json").dumps(list_models(state))}]}
    if uri == "engine://loaded":
        loaded = (
            {"name": state.loaded.name, "provider": state.loaded.registry_entry["provider"]}
            if state.loaded
            else None
        )
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": __import__("json").dumps(loaded)}]}
    if uri == "engine://metrics":
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": __import__("json").dumps(M.snapshot(state))}]}
    if uri == "engine://config":
        c = {
            "host": cfg.host,
            "port": cfg.port,
            "default_model": cfg.default_model,
            "default_provider": cfg.default_provider,
            "datasets_dir": str(cfg.datasets_dir),
            "checkpoints_dir": str(cfg.checkpoints_dir),
            "agent_allowlist": cfg.agent_allowlist,
        }
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": __import__("json").dumps(c)}]}
    if uri == "engine://agent/state":
        a = {
            "last_topic": state.agent.last_topic,
            "last_run_at": state.agent.last_run_at,
            "last_results": state.agent.last_results,
            "history": state.agent.history[-20:],
        }
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": __import__("json").dumps(a)}]}
    if uri == "engine://datasets":
        files = [
            {"name": p.name, "size": p.stat().st_size}
            for p in sorted(cfg.datasets_dir.glob("*.jsonl"))
        ]
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": __import__("json").dumps(files)}]}
    raise ValueError(f"unknown resource '{uri}'")


# ── Prompts ─────────────────────────────────────────────────────────────────
PROMPTS = [
    {
        "name": "engine.chat",
        "description": "Simple one-shot chat prompt.",
        "arguments": [{"name": "prompt", "required": True}],
    },
    {
        "name": "engine.summarize",
        "description": "Summarize the supplied text.",
        "arguments": [{"name": "text", "required": True}],
    },
    {
        "name": "engine.embed",
        "description": "Embed the supplied text.",
        "arguments": [{"name": "text", "required": True}],
    },
]


def _get_prompt(state: EngineState, name: str, args: dict[str, Any]) -> Any:
    if name == "engine.chat":
        prompt = args["prompt"]
        out = core_chat(state, prompt)
        return {
            "description": "chat result",
            "messages": [{"role": "user", "content": {"type": "text", "text": out["text"]}}],
        }
    if name == "engine.summarize":
        text = args["text"]
        out = core_chat(state, f"Summarize:\n{text}\n\nSummary:", max_new_tokens=128)
        return {
            "description": "summary",
            "messages": [{"role": "user", "content": {"type": "text", "text": out["text"]}}],
        }
    if name == "engine.embed":
        vec = core_embed(state, args["text"])
        return {
            "description": "embedding",
            "messages": [
                {"role": "user", "content": {"type": "text", "text": f"dimension={vec['dimension']}"}}
            ],
        }
    raise ValueError(f"unknown prompt '{name}'")


# ── endpoint ────────────────────────────────────────────────────────────────
@router.post("/mcp")
def mcp(
    body: RpcIn,
    request: Request,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    start = time.perf_counter()
    try:
        if body.method == "initialize":
            return _ok(
                body.id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": CAPABILITIES,
                    "serverInfo": SERVER_INFO,
                },
            )
        if body.method == "notifications/initialized":
            return _ok(body.id, {})
        if body.method == "ping":
            return _ok(body.id, {})
        if body.method == "tools/list":
            return _ok(body.id, {"tools": TOOLS})
        if body.method == "tools/call":
            params = body.params or {}
            result = _call_tool(state, params.get("name", ""), params.get("arguments", {}))
            return _ok(
                body.id,
                {
                    "content": [{"type": "text", "text": __import__("json").dumps(result)}],
                    "isError": False,
                },
            )
        if body.method == "resources/list":
            return _ok(body.id, {"resources": _resources(state)})
        if body.method == "resources/read":
            uri = (body.params or {}).get("uri", "")
            return _ok(body.id, _read_resource(state, uri))
        if body.method == "prompts/list":
            return _ok(body.id, {"prompts": PROMPTS})
        if body.method == "prompts/get":
            params = body.params or {}
            return _ok(
                body.id,
                _get_prompt(state, params.get("name", ""), params.get("arguments", {})),
            )
        return _err(body.id, -32601, f"method not found: {body.method}")
    except HTTPException:
        raise
    except Exception as e:
        return _err(body.id, -32603, str(e))
    finally:
        _ = time.perf_counter() - start  # keep hook for future telemetry
