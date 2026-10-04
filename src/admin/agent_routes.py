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

"""Self-learning agent UI — runs web searches and queues results for ingest."""
from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..core.state import EngineState
from ..api.chat_routes import get_state, require_auth

router = APIRouter()


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "topic"


@router.get("/ui/agent", response_class=HTMLResponse)
def agent_page(request: Request, state: EngineState = Depends(get_state)):
    return request.app.state.templates.TemplateResponse(
        request,
        name="agent.html",
        context={
            "title": "Self-Learning Agent",
            "agent": state.agent,
            "config": state.config,
            "active": "agent",
        },
    )


class AgentRunIn(BaseModel):
    topic: str
    max_pages: int | None = None
    max_depth: int = 1  # 1 = search results only; 2+ = also follow in-page links
    links_per_page: int = 3
    out_dir: str = ""


@router.post("/api/agent/run")
def run_agent(
    body: AgentRunIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    if state.agent.status == "running":
        raise HTTPException(status_code=409, detail="an agent run is already in progress")

    cfg = state.config
    # namespaced by topic (not dumped flat into data/raw) so different research
    # topics don't get mixed together, and "train on this" always has a clean,
    # well-defined folder to point at
    out_dir = Path(body.out_dir) if body.out_dir else cfg.raw_dir / "agent" / _slugify(body.topic)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts" / "agent_self_evolve.py"),
        "--topic", body.topic,
        "--max-pages", str(body.max_pages or cfg.agent_max_pages),
        "--max-depth", str(body.max_depth),
        "--links-per-page", str(body.links_per_page),
        "--out", str(out_dir),
    ]

    t0 = time.strftime("%Y-%m-%dT%H:%M:%S")
    log_path = cfg.data_dir / "agent_logs" / f"agent-{t0.replace(':', '-')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)

    state.agent.status = "running"
    state.agent.current_topic = body.topic
    state.agent.current_log_path = str(log_path)
    state.agent.current_started_at = t0
    state.agent.current_out_dir = str(out_dir)

    def _watch():
        try:
            rc = proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = -1
        log_f.close()

        files = []
        try:
            for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "WROTE " in line:
                    files.append(line.split("WROTE ", 1)[1].strip())
        except Exception:
            pass

        state.agent.last_topic = body.topic
        state.agent.last_run_at = t0
        state.agent.last_results = len(files)
        state.agent.last_files = files
        state.agent.last_out_dir = str(out_dir)
        state.agent.history.append(
            {"at": t0, "topic": body.topic, "files": files, "rc": rc, "out_dir": str(out_dir)}
        )
        state.agent.status = "idle"
        state.agent.current_topic = None
        state.agent.current_log_path = None
        state.agent.current_out_dir = None

    threading.Thread(target=_watch, daemon=True).start()
    return {"started": True, "topic": body.topic, "log_path": str(log_path), "out_dir": str(out_dir)}


@router.get("/api/agent/status")
def agent_status(state: EngineState = Depends(get_state)):
    return {
        "status": state.agent.status,
        "current_topic": state.agent.current_topic,
        "current_started_at": state.agent.current_started_at,
        "last_topic": state.agent.last_topic,
        "last_run_at": state.agent.last_run_at,
        "last_results": state.agent.last_results,
        "last_files": state.agent.last_files,
        "last_out_dir": state.agent.last_out_dir,
        "history": state.agent.history,
    }


@router.get("/api/agent/log")
def agent_log(state: EngineState = Depends(get_state)):
    if not state.agent.current_log_path:
        return {"status": state.agent.status, "log": None}
    path = Path(state.agent.current_log_path)
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    return {"status": state.agent.status, "log": text[-8192:]}
