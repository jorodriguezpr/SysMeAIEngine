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

"""Knowledge-graph endpoints: build (background job), stats, facts, and
deletion. The graph is optional — every endpoint here fails soft with a
clear message when no graph has been built for a corpus, and nothing in
the RAG or chat path depends on this module existing."""
from __future__ import annotations

from typing import Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..core import kgraph
from ..core.state import EngineState
from .chat_routes import get_state, require_auth

router = APIRouter()

# in-memory job registry, same lifecycle as rag eval jobs (cleared on restart)
_build_jobs: dict[str, dict] = {}


class KGraphBuildIn(BaseModel):
    corpus: str
    max_chunks: Optional[int] = None  # bound the cloud cost while testing


@router.post("/api/kgraph/build")
def kgraph_build_start(
    body: KGraphBuildIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    import threading
    import time
    import uuid

    # fail fast if the provider isn't usable, before starting the job
    try:
        from ..core.cloud_distill import ensure_cloud_ready

        ensure_cloud_ready(state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rag_names = kgraph.list_rag_corpora(state)
    if body.corpus not in rag_names:
        raise HTTPException(
            status_code=404,
            detail=f"no RAG index built for corpus '{body.corpus}' (have: {rag_names or 'none'})",
        )

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id, "corpus": body.corpus, "status": "running",
        "params": body.model_dump(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished_at": None, "result": None, "error": None,
        "progress": {"done": 0, "total": None, "failed": 0},
    }
    _build_jobs[job_id] = job

    def _run() -> None:
        def _progress(done: int, total: int, failed: int) -> None:
            job["progress"] = {"done": done, "total": total, "failed": failed}

        try:
            result = kgraph.build_graph(
                state, body.corpus, max_chunks=body.max_chunks, on_progress=_progress,
            )
            job["result"] = result
            job["status"] = "done"
        except Exception as e:  # surfaced to the poller, not swallowed
            job["error"] = f"{type(e).__name__}: {e}"
            job["status"] = "failed"
        job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@router.get("/api/kgraph/build/{job_id}")
def kgraph_build_status(job_id: str, _: None = Depends(require_auth)):
    job = _build_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown build job id (jobs are kept in memory until server restart)")
    return job


@router.get("/api/kgraph/corpora")
def kgraph_corpora(state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    return {"graphs": kgraph.list_corpora(state)}


@router.get("/api/kgraph/stats/{corpus}")
def kgraph_stats(corpus: str, state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    try:
        return kgraph.graph_stats(state, corpus)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class KGraphQueryIn(BaseModel):
    question: str
    corpus: Union[str, list[str]] = "all"
    max_facts: int = 10


@router.post("/api/kgraph/query")
def kgraph_query(
    body: KGraphQueryIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    names = (
        kgraph.list_corpora(state) if body.corpus == "all"
        else [body.corpus] if isinstance(body.corpus, str) else list(body.corpus)
    )
    out = []
    try:
        for name in names:
            out.append(kgraph.graph_facts(state, name, body.question, max_facts=body.max_facts))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"results": out}


@router.delete("/api/kgraph/{corpus}")
def kgraph_delete(corpus: str, state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    try:
        deleted = kgraph.delete_graph(state, corpus)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"no graph built for corpus '{corpus}'")
    return {"deleted": corpus}