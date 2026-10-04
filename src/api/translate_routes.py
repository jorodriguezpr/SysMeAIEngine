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

"""Spanish<->English translation endpoint via a dedicated MarianMT model
(see src/core/translate.py for why a dedicated model instead of prompting
the general chat LLM)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..core import translate as translate_mod
from ..core.state import EngineState
from .chat_routes import get_state, require_auth

router = APIRouter()


class TranslateIn(BaseModel):
    text: str
    direction: str  # "es-en" or "en-es"


@router.post("/api/translate")
def translate(
    body: TranslateIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    try:
        translation = translate_mod.translate(body.text, body.direction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"direction": body.direction, "text": body.text, "translation": translation}
