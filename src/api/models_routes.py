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

"""Model registry management endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..core.model_loader import list_models, register_model, set_rag_corpus, unregister_model
from ..core.state import EngineState
from .chat_routes import get_state, require_auth

router = APIRouter()


class RegisterIn(BaseModel):
    name: str
    provider: str = "minilm"
    path: str = ""
    device: Optional[str] = None
    # which RAG corpus /api/rag/chat should default to for this model when the
    # caller doesn't specify one explicitly -- see register_model()'s own
    # comment for the "point at a different model's corpus" use case
    rag_corpus: Optional[str] = None


@router.get("/api/registry")
def read_registry(state: EngineState = Depends(get_state)):
    return list_models(state)


@router.post("/api/registry/register")
def register(
    body: RegisterIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    try:
        return register_model(
            state, body.name, body.provider, body.path,
            rag_corpus=body.rag_corpus, device=body.device,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class SetRagCorpusIn(BaseModel):
    name: str
    rag_corpus: str


@router.post("/api/registry/set-rag-corpus")
def set_rag_corpus_route(
    body: SetRagCorpusIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    try:
        return set_rag_corpus(state, body.name, body.rag_corpus)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/registry/unregister")
def unregister(
    body: dict,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    try:
        unregister_model(state, body["name"])
        return {"ok": True}
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
