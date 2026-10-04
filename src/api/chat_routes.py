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

"""Simple chat & generation endpoints (not OpenAI-shaped — direct and small)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from ..core import metrics as M
from ..core.model_loader import chat as core_chat
from ..core.model_loader import embeddings as core_embed
from ..core.model_loader import list_models
from ..core.model_loader import load_model, unload_model
from ..core.state import EngineState

router = APIRouter()


def get_state(request: Request) -> EngineState:
    return request.app.state.engine


def require_auth(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> None:
    cfg = request.app.state.engine.config
    if not cfg.api_key:
        return
    if x_api_key == cfg.api_key:
        return
    if authorization and authorization.startswith("Bearer "):
        if authorization.removeprefix("Bearer ").strip() == cfg.api_key:
            return
    raise HTTPException(status_code=401, detail="missing or invalid API key")


class ChatIn(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    # None -> fall back to the saved Settings-page default; explicit
    # true/false overrides it for this one request (the Chat page's
    # checkbox always sends an explicit value)
    use_cloud: Optional[bool] = None
    # None -> fall back to the saved Settings-page chat_mode ("plain" =
    # weights-only generation, the old behavior; "grounded" = retrieve from
    # the local RAG indexes + knowledge graph first, then generate from the
    # retrieved passages). Explicit value overrides for this one request.
    chat_mode: Optional[str] = None
    # None -> Settings default (chat_top_k); explicit value caps passages
    # retrieved for this one grounded request
    top_k: Optional[int] = None


class ChatOut(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: float
    text: str
    # grounded mode only (None otherwise): the passages the answer was
    # generated from, and which retrieval steps actually ran
    sources: Optional[list[dict]] = None
    grounding: Optional[dict] = None


@router.post("/api/chat", response_model=ChatOut)
def chat(
    body: ChatIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
) -> ChatOut:
    import time

    from ..core.calculator import try_calculate
    from ..core.summarize import detect_summarize_request, summarize_document
    from ..core.translate import detect_translate_request
    from ..core.translate import translate as run_translate

    calc = try_calculate(body.prompt)
    if calc is not None:
        return ChatOut(
            model="calculator", prompt_tokens=0, completion_tokens=0,
            elapsed_ms=0.0, text=calc,
        )
    doc_path = detect_summarize_request(body.prompt)
    if doc_path is not None:
        t0 = time.perf_counter()
        try:
            result = summarize_document(state, doc_path, model_name=body.model)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return ChatOut(
            model=result["model"], prompt_tokens=0, completion_tokens=0,
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            text=result["summary"],
        )
    translate_req = detect_translate_request(body.prompt)
    if translate_req is not None:
        text, direction = translate_req
        t0 = time.perf_counter()
        try:
            translation = run_translate(text, direction)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return ChatOut(
            model=f"opus-mt-{direction}", prompt_tokens=0, completion_tokens=0,
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            text=translation,
        )

    # ── grounded chat mode (P0): retrieve before generate ──────────────────
    # The old path generated directly from model weights with zero
    # retrieval, so the ingested documents were never consulted — the root
    # cause of "chat can't answer from my books". In grounded mode every
    # prompt is first answered with the local RAG indexes (+ knowledge
    # graph) and only the retrieved passages reach the generator.
    from ..core.grounding_settings import get_grounding_settings

    gs = get_grounding_settings(state)
    chat_mode = body.chat_mode if body.chat_mode is not None else gs["chat_mode"]
    if chat_mode == "grounded":
        return _grounded_chat(body, state, gs)

    from ..core.cloud_chat import cloud_generate
    from ..core.cloud_settings import get_cloud_settings

    cloud_settings = get_cloud_settings(state)
    use_cloud = body.use_cloud if body.use_cloud is not None else cloud_settings["enabled"]
    if use_cloud:
        t0 = time.perf_counter()
        try:
            text = cloud_generate(cloud_settings, body.prompt)
        except (ValueError, RuntimeError) as e:
            M.record_error(state)
            raise HTTPException(status_code=400, detail=str(e))
        return ChatOut(
            model=f"cloud:{cloud_settings['provider']}:{cloud_settings['model']}",
            prompt_tokens=len(body.prompt) // 4 + 1, completion_tokens=len(text) // 4 + 1,
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            text=text,
        )
    try:
        out = core_chat(
            state,
            body.prompt,
            model_name=body.model,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
        )
        return ChatOut(**out)
    except Exception as e:
        M.record_error(state)
        raise HTTPException(status_code=400, detail=str(e))


def _grounded_chat(body: ChatIn, state: EngineState, gs: dict) -> ChatOut:
    """Grounded mode: retrieve passages (RAG + optional knowledge graph),
    then generate the answer strictly from them. Multi-query + parent-child
    expansion + MMR live in rag.retrieve_grounded(); here we only wire the
    result into whichever generator is configured (cloud preferred, local
    fallback)."""
    import time

    from ..core.cloud_chat import cloud_generate
    from ..core.cloud_settings import get_cloud_settings
    from ..core import rag as rag_mod
    from ..core.grounding_settings import MAX_TOP_K
    from ..core.model_loader import get_model_entry

    model_name = body.model or state.config.default_model
    corpus = None
    entry = get_model_entry(state, model_name)
    corpus = (entry.get("rag_corpus") if entry else None) or "all"

    # Explicit per-request use_cloud (the Chat page always sends one) wins
    # over the Settings default -- and False means NOTHING goes to the cloud,
    # including the multi-query rewrite. Previously ignored here: live-
    # verified that use_cloud=false still made 2 cloud calls (rewrite +
    # answer) carrying the question and retrieved passages.
    use_cloud = body.use_cloud if body.use_cloud is not None else bool(gs.get("grounded_use_cloud", True))

    top_k = max(1, min(MAX_TOP_K, body.top_k if body.top_k is not None else int(gs.get("chat_top_k") or 6)))

    def _retrieve(for_cloud: bool):
        # Parent-child widening only when the CLOUD writes the answer.
        # Measured (scripts/eval_grounded.py, aprocrifos-v2): it raised the
        # cloud answerer's recall/accuracy, but for the local 0.5B model the
        # ~2.4x longer context LOWERED correctness (0.33 -> 0.20) and
        # out-of-corpus refusals (20% -> 0%). Local answers get the compact
        # context -- also the exact format the P4 grounded fine-tune uses.
        return rag_mod.retrieve_grounded(
            state, body.prompt, top_k=top_k, corpus=corpus,
            multi_query=bool(gs.get("chat_multi_query", False)) and for_cloud,
            parent_child=for_cloud,
            kgraph_enabled=bool(gs.get("chat_kgraph", False)),
            allow_cloud=for_cloud,
            graph_slot=bool(gs.get("chat_graph_slot", False)),
        )

    t0 = time.perf_counter()
    try:
        hits, dbg = _retrieve(use_cloud)
    except FileNotFoundError:
        # No indexes built at all: fall back to the plain path rather than
        # erroring -- grounding is an enhancement, not a hard requirement.
        return _plain_chat(body, state)

    sources = [
        {k: h.get(k) for k in ("corpus", "source", "id", "score")} | ({"kgraph": True} if h.get("kgraph") else {})
        for h in hits
    ]
    grounding = {**dbg, "corpus": corpus, "use_cloud": use_cloud, "fallback": None}

    if not hits and gs.get("grounded_refuse_when_empty", True):
        return ChatOut(
            model="grounded:none", prompt_tokens=0, completion_tokens=0,
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            text=rag_mod.GROUNDING_REFUSAL, sources=[], grounding=grounding,
        )

    augmented = rag_mod.build_prompt(body.prompt, hits)

    cloud_settings = get_cloud_settings(state)
    if use_cloud:
        try:
            text = cloud_generate(cloud_settings, augmented)
            if not text.strip():  # still empty after cloud_generate's own larger-budget retry
                raise RuntimeError("cloud provider returned an empty answer")
            return ChatOut(
                model=f"grounded:cloud:{cloud_settings['provider']}:{cloud_settings['model']}",
                prompt_tokens=len(augmented) // 4 + 1,
                completion_tokens=len(text) // 4 + 1,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                text=text, sources=sources, grounding=grounding,
            )
        except (ValueError, RuntimeError) as e:
            # provider disabled/misconfigured/offline mid-request (network
            # errors arrive as RuntimeError, see cloud_chat._post): degrade
            # to the local model instead of failing the whole grounded turn
            grounding["fallback"] = f"cloud failed, answered locally: {str(e)[:200]}"
            # re-retrieve in the compact local profile (no widening) -- a
            # couple of seconds, only on this fallback path
            hits, dbg = _retrieve(False)
            augmented = rag_mod.build_prompt(body.prompt, hits)
            sources = [
                {k: h.get(k) for k in ("corpus", "source", "id", "score")} | ({"kgraph": True} if h.get("kgraph") else {})
                for h in hits
            ]
            grounding.update(dbg)

    try:
        # Local answer = the measured best offline recipe: instructions in
        # the question's language (local_prompt_style, default "complete"),
        # compact context (see _retrieve), temperature 0, room for a
        # complete answer, and the script-drift guard.
        style = gs.get("local_prompt_style") or "complete"
        local_prompt = rag_mod.build_prompt(body.prompt, hits, style)
        lang = rag_mod.question_language(body.prompt, hits)
        model_name = body.model or state.config.default_model
        text = rag_mod.local_answer(state, local_prompt, model_name, lang,
                                    max_new_tokens=max(body.max_new_tokens, 350))
        if rag_mod.is_refusal(text):
            text = rag_mod.REFUSALS.get(lang, rag_mod.REFUSALS["English"])
        grounding["local_prompt_style"] = style
        return ChatOut(
            model=f"grounded:local:{model_name}",
            prompt_tokens=len(local_prompt) // 4 + 1, completion_tokens=len(text) // 4 + 1,
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            text=text, sources=sources, grounding=grounding,
        )
    except Exception as e:
        M.record_error(state)
        raise HTTPException(status_code=400, detail=str(e))


def _plain_chat(body: ChatIn, state: EngineState) -> ChatOut:
    """The original weights-only generation path, preserved verbatim so
    chat_mode='plain' behaves exactly as before grounding existed."""
    try:
        out = core_chat(
            state,
            body.prompt,
            model_name=body.model,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
        )
        return ChatOut(**out)
    except Exception as e:
        M.record_error(state)
        raise HTTPException(status_code=400, detail=str(e))


class EmbedIn(BaseModel):
    model: Optional[str] = None
    input: str | list[str]


@router.post("/api/embeddings")
def embeddings(
    body: EmbedIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    try:
        return core_embed(state, body.input, model_name=body.model)
    except Exception as e:
        M.record_error(state)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/models")
def models(state: EngineState = Depends(get_state)):
    return list_models(state)


class LoadIn(BaseModel):
    name: str


@router.post("/api/models/load")
def load(state: EngineState = Depends(get_state), body: LoadIn | None = None):
    name = (body.name if body else state.config.default_model)
    load_model(state, name)
    return {"loaded": name}


@router.post("/api/models/unload")
def unload(state: EngineState = Depends(get_state)):
    unload_model(state)
    return {"loaded": None}
