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

"""Cloud AI provider settings UI."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from ..core.cloud_settings import get_cloud_settings, masked
from ..core.grounding_settings import get_grounding_settings
from ..core.state import EngineState
from ..api.chat_routes import get_state

router = APIRouter()


@router.get("/ui/settings", response_class=HTMLResponse)
def settings_page(request: Request, state: EngineState = Depends(get_state)):
    settings = masked(get_cloud_settings(state))
    return request.app.state.templates.TemplateResponse(
        request,
        name="settings.html",
        context={
            "title": "Settings", "settings": settings, "active": "settings",
            "grounding": get_grounding_settings(state),
        },
    )
