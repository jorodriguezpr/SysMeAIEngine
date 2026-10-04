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

"""Cloud AI provider settings -- enable/disable, provider, model, API key,
base URL override, and prompt customization (system prompt/temperature/
max_tokens). See src/core/cloud_settings.py and cloud_chat.py."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..core.cloud_settings import get_cloud_settings, masked, save_cloud_settings
from ..core.grounding_settings import get_grounding_settings, save_grounding_settings
from ..core.state import EngineState
from .chat_routes import get_state, require_auth

router = APIRouter()


@router.get("/api/settings/cloud")
def read_cloud_settings(state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    return masked(get_cloud_settings(state))


class CloudSettingsIn(BaseModel):
    enabled: Optional[bool] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@router.post("/api/settings/cloud")
def write_cloud_settings(
    body: CloudSettingsIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    # an empty-string api_key from the form means "leave the saved key alone"
    # (the GET endpoint never returns the real key, so the form can't safely
    # round-trip it) -- only overwrite when a real, non-empty value is sent.
    fields = body.model_dump()
    if fields.get("api_key") == "":
        fields["api_key"] = None
    saved = save_cloud_settings(state, **fields)
    return masked(saved)


class GroundingSettingsIn(BaseModel):
    chat_mode: Optional[str] = None
    chat_top_k: Optional[int] = None
    chat_multi_query: Optional[bool] = None
    chat_kgraph: Optional[bool] = None
    grounded_use_cloud: Optional[bool] = None
    grounded_refuse_when_empty: Optional[bool] = None


@router.get("/api/settings/grounding")
def read_grounding_settings(state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    return get_grounding_settings(state)


@router.post("/api/settings/grounding")
def write_grounding_settings(
    body: GroundingSettingsIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    try:
        return save_grounding_settings(state, **body.model_dump())
    except ValueError as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=str(e))
