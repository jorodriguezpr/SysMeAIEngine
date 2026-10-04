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

"""Per-node agent API: what a remote master calls on this node.

Every SysMeAIEngine instance exposes these routes — there is no separate
"slave build". A node only ever receives calls here when some other instance
has registered it (src/cluster/nodes.py) and decided to include it in a job.
"""
from __future__ import annotations

import json
import platform
import shutil
import socket
import sys
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from ..api.chat_routes import get_state, require_auth
from ..core.device_select import device_report
from ..core.state import EngineState
from ..core.utils.device_probe import probe_cpu, probe_gpu
from . import runner

router = APIRouter(prefix="/cluster", tags=["cluster"])


@router.get("/info")
def cluster_info(state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    import torch  # lazy

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cpu": probe_cpu(),
        "gpu": probe_gpu(),
        "devices": device_report(),
    }


def _job_dir(state: EngineState, job_hint: str) -> Path:
    d = state.config.data_dir / "cluster_jobs" / job_hint
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.post("/jobs")
async def submit_job(
    job_spec: str = Form(...),
    dataset: UploadFile = File(...),
    base_model_zip: Optional[UploadFile] = File(default=None),
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    try:
        spec = json.loads(job_spec)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"invalid job_spec JSON: {e}")

    for required in ("rank", "world_size", "master_addr", "master_port", "model"):
        if required not in spec:
            raise HTTPException(status_code=400, detail=f"job_spec missing '{required}'")

    job_hint = f"r{spec['rank']}-{spec.get('run_name', 'job')}"
    job_dir = _job_dir(state, job_hint)

    dataset_path = job_dir / (dataset.filename or "dataset.jsonl")
    with open(dataset_path, "wb") as f:
        shutil.copyfileobj(dataset.file, f)

    model_override = None
    if base_model_zip is not None:
        zip_path = job_dir / "base_model.zip"
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(base_model_zip.file, f)
        extract_dir = job_dir / "base_model"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        model_override = str(extract_dir)

    job = runner.launch_job(state, spec, dataset_path, model_override=model_override)
    return {"job_id": job.id, "log_path": job.log_path}


@router.get("/jobs")
def list_jobs(state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    """Every job this node has ever run/is running, most recently started
    first -- lets a master (or an operator) find and cancel a job it lost
    track of, e.g. after the master-side process that launched it died
    without ever calling /cancel (confirmed live: a master crash left an
    orphaned rank running on a remote node with no way to identify or stop
    it, since job ids aren't otherwise discoverable from the master side)."""
    jobs = sorted(state.train_jobs.values(), key=lambda j: j.started_at or "", reverse=True)
    return [
        {
            "id": job.id, "status": job.status, "exit_code": job.exit_code,
            "started_at": job.started_at, "finished_at": job.finished_at, "args": job.args,
        }
        for job in jobs
    ]


@router.get("/jobs/{job_id}")
def job_status(job_id: str, state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    job = state.train_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "id": job.id, "status": job.status, "exit_code": job.exit_code,
        "started_at": job.started_at, "finished_at": job.finished_at, "args": job.args,
    }


@router.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str, state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    job = state.train_jobs.get(job_id)
    if not job or not job.log_path:
        raise HTTPException(status_code=404, detail="job not found")
    path = Path(job.log_path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[-8192:]


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    job = state.train_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    proc = state.running_processes.get(job_id)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        job.status = "cancelled"
    return {"id": job_id, "status": job.status}
