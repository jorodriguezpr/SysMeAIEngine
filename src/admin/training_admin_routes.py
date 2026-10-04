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

"""Training UI — create datasets, launch FT jobs, view history."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..core.state import EngineState, TrainJob
from ..core.train_progress import job_progress
from ..api.chat_routes import get_state, require_auth

router = APIRouter()


def _watch_job(proc: subprocess.Popen, job: TrainJob, log_f) -> None:
    """Wait for the subprocess in a background thread and record the result —
    without this, TrainJob.status just says "running" forever."""
    def _run():
        exit_code = proc.wait()
        log_f.close()
        job.exit_code = exit_code
        job.status = "done" if exit_code == 0 else "failed"
        job.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    threading.Thread(target=_run, daemon=True).start()


def _job_dict(j: TrainJob) -> dict:
    return {
        "id": j.id,
        "args": j.args,
        "started_at": j.started_at,
        "finished_at": j.finished_at,
        "status": j.status,
        "log_path": j.log_path,
        # Only worth reading the log for jobs still in flight -- a
        # finished job's own status/exit_code already says everything.
        "progress": job_progress(j.log_path) if j.status == "running" else None,
    }


@router.get("/ui/training", response_class=HTMLResponse)
def training_page(request: Request, state: EngineState = Depends(get_state)):
    cfg = state.config
    datasets = [
        {"name": p.name, "size": p.stat().st_size}
        for p in sorted(cfg.datasets_dir.glob("*.jsonl"))
    ]
    raw_files = sorted(p.name for p in cfg.raw_dir.glob("*"))
    chunks_files = sorted(p.name for p in cfg.chunks_dir.glob("*.txt"))
    jobs = [_job_dict(j) for j in state.train_jobs.values()]
    return request.app.state.templates.TemplateResponse(
        request,
        name="training.html",
        context={
            "title": "Training",
            "datasets": datasets,
            "raw_files": raw_files,
            "chunks_files": chunks_files,
            "jobs": jobs,
            "config": cfg,
            "active": "training",
        },
    )


class FinetuneIn(BaseModel):
    model: str = "minilm"
    dataset: str
    output: str = ""
    method: str = "lora"  # sft | lora | qlora
    epochs: int = 1
    lr: float = 2e-4
    batch_size: int = 1


@router.post("/api/train/finetune")
def start_finetune(
    body: FinetuneIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    dataset_path = Path(body.dataset)
    if not dataset_path.is_absolute():
        dataset_path = state.config.datasets_dir / body.dataset
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"dataset not found: {dataset_path}")

    output = body.output or str(
        state.config.checkpoints_dir / f"ft-{time.strftime('%Y%m%d-%H%M%S')}"
    )

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts" / "finetune_model.py"),
        "--model", body.model,
        "--dataset", str(dataset_path),
        "--output", output,
        "--method", body.method,
        "--epochs", str(body.epochs),
        "--lr", str(body.lr),
        "--batch-size", str(body.batch_size),
    ]
    job_id = uuid.uuid4().hex[:12]
    log_path = state.config.checkpoints_dir / f"job-{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")

    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    job = TrainJob(
        id=job_id,
        args={
            "model": body.model,
            "dataset": str(dataset_path),
            "output": output,
            "method": body.method,
            "epochs": body.epochs,
            "lr": body.lr,
            "batch_size": body.batch_size,
            "pid": proc.pid,
        },
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        log_path=str(log_path),
    )
    state.train_jobs[job_id] = job
    _watch_job(proc, job, log_f)
    return {"job_id": job_id, "log_path": str(log_path), "cmd": cmd}


@router.get("/api/train/jobs")
def list_jobs(state: EngineState = Depends(get_state)):
    return [_job_dict(j) for j in state.train_jobs.values()]


@router.get("/api/train/jobs/{job_id}/log")
def job_log(job_id: str, state: EngineState = Depends(get_state)):
    job = state.train_jobs.get(job_id)
    if not job or not job.log_path:
        raise HTTPException(status_code=404, detail="job not found")
    path = Path(job.log_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="log missing")
    return {"log": path.read_text(encoding="utf-8", errors="ignore")[-4096:]}
