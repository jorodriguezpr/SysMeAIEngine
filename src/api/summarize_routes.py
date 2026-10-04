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

"""Whole-document summarization endpoint (map-reduce over chunks -- see
src/core/summarize.py for why a single prompt can't just hold a whole
ebook)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..core.state import EngineState
from ..core.summarize import summarize_document
from .chat_routes import get_state, require_auth

router = APIRouter()


class SummarizeIn(BaseModel):
    source: str
    model: Optional[str] = None
    chunk_size: int = 1200


@router.post("/api/summarize")
def summarize(
    body: SummarizeIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    try:
        return summarize_document(state, body.source, model_name=body.model, chunk_size=body.chunk_size)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
