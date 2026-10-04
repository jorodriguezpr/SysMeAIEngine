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

"""build_app() — assemble the FastAPI application."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .admin import (agent_routes, chat_admin_routes, cluster_admin_routes,
                    dashboard_routes, help_routes, kgraph_admin_routes, mcp_admin_routes,
                    models_admin_routes, pdf_training_routes, settings_admin_routes,
                    storage_routes, training_admin_routes, weights_admin_routes)
from .api import (chat_routes, kgraph_routes, mcp_routes, metrics_routes, models_routes,
                  openai_routes, rag_routes, settings_routes, summarize_routes,
                  translate_routes)
from .cluster import agent_routes as cluster_agent_routes
from .core import config as config_mod
from .core import state as state_mod

TEMPLATES_DIR = Path(__file__).resolve().parents[0] / "web" / "templates"
STATIC_DIR = Path(__file__).resolve().parents[0] / "web" / "static"


def build_app(cfg: config_mod.Config | None = None) -> FastAPI:
    app = FastAPI(title="SysMeAIEngine", version="0.1.0")

    cfg = cfg or config_mod.Config.from_env()
    app.state.engine = state_mod.init_state(cfg)
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # API
    app.include_router(chat_routes.router)
    app.include_router(openai_routes.router)
    app.include_router(models_routes.router)
    app.include_router(metrics_routes.router)
    app.include_router(mcp_routes.router)
    app.include_router(rag_routes.router)
    app.include_router(kgraph_routes.router)
    app.include_router(summarize_routes.router)
    app.include_router(translate_routes.router)
    app.include_router(settings_routes.router)

    # Admin / UI
    app.include_router(dashboard_routes.router)
    app.include_router(chat_admin_routes.router)
    app.include_router(models_admin_routes.router)
    app.include_router(weights_admin_routes.router)
    app.include_router(training_admin_routes.router)
    app.include_router(pdf_training_routes.router)
    app.include_router(agent_routes.router)
    app.include_router(mcp_admin_routes.router)
    app.include_router(storage_routes.router)
    app.include_router(cluster_admin_routes.router)
    app.include_router(help_routes.router)
    app.include_router(kgraph_admin_routes.router)
    app.include_router(settings_admin_routes.router)

    # Cluster agent API — every instance can receive a job from a remote
    # master, not just ones set up as "the master" (see src/cluster/).
    app.include_router(cluster_agent_routes.router)

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/ui")

    return app
