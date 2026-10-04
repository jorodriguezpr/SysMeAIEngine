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

"""Shared job-launch logic: spawns scripts/distributed_finetune.py as a
subprocess with torch.distributed rendezvous env vars set, and tracks it the
same way training_admin_routes.py already tracks single-node jobs.

Used by both agent_routes.py (jobs arriving over HTTP from a remote master)
and orchestrator.py (the master's own local rank-0 job) — a node acting as
master runs its own rank exactly like this, no HTTP self-loop needed.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from ..core.state import EngineState, TrainJob

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "distributed_finetune.py"


def _watch(proc: subprocess.Popen, job: TrainJob, log_f, state: EngineState, job_id: str) -> None:
    def _run():
        exit_code = proc.wait()
        log_f.close()
        job.exit_code = exit_code
        job.status = "done" if exit_code == 0 else "failed"
        job.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        state.running_processes.pop(job_id, None)

    threading.Thread(target=_run, daemon=True).start()


def launch_job(
    state: EngineState,
    job_spec: dict[str, Any],
    dataset_path: Path,
    model_override: Optional[str] = None,
) -> TrainJob:
    """job_spec keys: rank, world_size, master_addr, master_port, model, method,
    epochs, lr, batch_size, max_seq. `model_override` replaces job_spec['model']
    when the caller already unzipped a locally-transferred base checkpoint."""
    model = model_override or job_spec["model"]
    rank = int(job_spec["rank"])
    world_size = int(job_spec["world_size"])
    job_id = uuid.uuid4().hex[:12]

    output = job_spec.get("output") or str(
        state.config.checkpoints_dir / f"cluster-{job_spec.get('run_name', job_id)}-rank{rank}"
    )

    cmd = [
        sys.executable, str(SCRIPT),
        "--model", model,
        "--dataset", str(dataset_path),
        "--output", output,
        "--method", str(job_spec.get("method", "lora")),
        "--epochs", str(job_spec.get("epochs", 1)),
        "--lr", str(job_spec.get("lr", 2e-4)),
        "--batch-size", str(job_spec.get("batch_size", 1)),
        "--max-seq", str(job_spec.get("max_seq", 1024)),
    ]

    env = {
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "LOCAL_RANK": "0",
        "MASTER_ADDR": str(job_spec["master_addr"]),
        "MASTER_PORT": str(job_spec["master_port"]),
    }
    import os
    full_env = {**os.environ, **env}

    log_path = state.config.checkpoints_dir / f"cluster-job-{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")

    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True, env=full_env)
    job = TrainJob(
        id=job_id,
        args={
            "rank": rank, "world_size": world_size, "model": model,
            "output": output, "pid": proc.pid, "rendezvous": f"{job_spec['master_addr']}:{job_spec['master_port']}",
        },
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        log_path=str(log_path),
    )
    state.train_jobs[job_id] = job
    state.running_processes[job_id] = proc
    _watch(proc, job, log_f, state, job_id)
    return job
