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

"""Master-side admin UI + API: manage slave nodes, launch distributed jobs.

This lives on every instance (any node can act as master for a given job —
see src/cluster/) but is only meaningful once at least one other node has
been registered via /api/cluster/nodes.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..api.chat_routes import get_state, require_auth
from ..cluster import nodes as nodes_mod
from ..cluster import orchestrator
from ..cluster.client import NodeClient
from ..core.state import EngineState

router = APIRouter()


@router.get("/ui/cluster", response_class=HTMLResponse)
def cluster_page(request: Request, state: EngineState = Depends(get_state)):
    cfg = state.config
    nodes = nodes_mod.list_nodes(cfg)
    datasets = [p.name for p in sorted(cfg.datasets_dir.glob("*.jsonl"))]
    jobs = [
        {"id": j.id, "nodes": j.nodes, "args": j.args, "started_at": j.started_at}
        for j in state.cluster_jobs.values()
    ]
    return request.app.state.templates.TemplateResponse(
        request,
        name="cluster.html",
        context={
            "title": "Cluster",
            "nodes": nodes,
            "datasets": datasets,
            "jobs": jobs,
            "active": "cluster",
        },
    )


class NodeIn(BaseModel):
    name: str
    host: str
    port: int = 11534
    api_key: str | None = None
    scheme: str = "http"


@router.get("/api/cluster/nodes")
def list_nodes(state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    return nodes_mod.list_nodes(state.config)


@router.post("/api/cluster/nodes")
def add_node(
    body: NodeIn, state: EngineState = Depends(get_state), _: None = Depends(require_auth)
):
    try:
        return nodes_mod.add_node(
            state.config, body.name, body.host, body.port, body.api_key, body.scheme
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/api/cluster/nodes/{name}")
def remove_node(name: str, state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    try:
        nodes_mod.remove_node(state.config, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True}


@router.post("/api/cluster/nodes/{name}/ping")
def ping_node(name: str, state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    node = nodes_mod.get_node(state.config, name)
    if not node:
        raise HTTPException(status_code=404, detail="node not found")
    ok, msg = NodeClient(node).ping()
    return {"reachable": ok, "detail": msg}


class TrainIn(BaseModel):
    model: str
    dataset: str
    nodes: list[str]
    method: str = "lora"
    epochs: int = 1
    lr: float = 2e-4
    batch_size: int = 1
    max_seq: int = 1024
    output: str | None = None


@router.post("/api/cluster/train")
def start_distributed_train(
    body: TrainIn, state: EngineState = Depends(get_state), _: None = Depends(require_auth)
):
    try:
        job = orchestrator.launch_distributed_job(
            state, body.nodes, body.model, body.dataset,
            method=body.method, epochs=body.epochs, lr=body.lr,
            batch_size=body.batch_size, max_seq=body.max_seq, output=body.output,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"id": job.id, "nodes": job.nodes, "local_job_id": job.local_job_id, "remote": job.remote}


@router.get("/api/cluster/jobs")
def list_cluster_jobs(state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    return [
        {"id": j.id, "nodes": j.nodes, "args": j.args, "started_at": j.started_at}
        for j in state.cluster_jobs.values()
    ]


@router.get("/api/cluster/jobs/{run_name}")
def cluster_job_detail(
    run_name: str, state: EngineState = Depends(get_state), _: None = Depends(require_auth)
):
    try:
        return orchestrator.cluster_job_status(state, run_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/api/cluster/jobs/{run_name}/log")
def cluster_job_local_log(
    run_name: str, state: EngineState = Depends(get_state), _: None = Depends(require_auth)
):
    cj = state.cluster_jobs.get(run_name)
    if not cj or not cj.local_job_id:
        raise HTTPException(status_code=404, detail="job not found")
    job = state.train_jobs.get(cj.local_job_id)
    if not job or not job.log_path:
        raise HTTPException(status_code=404, detail="log not found")
    from pathlib import Path

    path = Path(job.log_path)
    if not path.exists():
        return {"log": ""}
    return {"log": path.read_text(encoding="utf-8", errors="ignore")[-8192:]}
