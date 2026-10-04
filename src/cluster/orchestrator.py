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

"""Master-side: fan a single distributed fine-tune job out across this node
(rank 0) and one or more registered slave nodes (rank 1..N).

Honest limitations (see docs/CLUSTER.md for the full writeup):
  * Synchronous DDP means every step waits for the slowest rank. Mixing a
    CPU rank with CUDA ranks can bottleneck the whole job to CPU speed.
  * Uses the Gloo backend unconditionally (not NCCL) specifically because
    Gloo is the only backend that works across a CPU+CUDA-mixed group —
    NCCL requires every rank to have a GPU.
  * Not benchmarked on real separate machines / real GPUs in this repo (no
    such cluster was available to test against) — verified so far via a
    same-machine loopback smoke test only (scripts/verify_cluster_loopback.py).
"""
from __future__ import annotations

import socket
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional

from ..core.state import ClusterJob, EngineState
from . import nodes as nodes_mod
from . import runner
from .client import NodeClient

DEFAULT_RENDEZVOUS_PORT = 29500


def detect_lan_ip() -> str:
    """Best-effort local LAN IP (no packet actually sent — UDP connect just
    asks the OS to pick the outbound route/interface)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _free_port(preferred: int = DEFAULT_RENDEZVOUS_PORT) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free rendezvous port found")


def preflight(cfg, node_names: list[str]) -> dict[str, Any]:
    """Ping every node and report reachability + torch/python version, so a
    mismatch is a clear warning before launch rather than a cryptic
    mid-training distributed-protocol error."""
    results = {}
    for name in node_names:
        node = nodes_mod.get_node(cfg, name)
        if not node:
            results[name] = {"reachable": False, "error": f"'{name}' not registered"}
            continue
        client = NodeClient(node)
        ok, msg = client.ping()
        if not ok:
            results[name] = {"reachable": False, "error": msg}
            continue
        info = client.info()
        results[name] = {"reachable": True, **info}
    return results


def _zip_dir(src: Path, dest_zip: Path) -> Path:
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src))
    return dest_zip


def launch_distributed_job(
    state: EngineState,
    node_names: list[str],
    model: str,
    dataset: str,
    method: str = "lora",
    epochs: int = 1,
    lr: float = 2e-4,
    batch_size: int = 1,
    max_seq: int = 1024,
    output: Optional[str] = None,
    rendezvous_port: Optional[int] = None,
) -> ClusterJob:
    cfg = state.config
    dataset_path = Path(dataset)
    if not dataset_path.is_absolute():
        dataset_path = cfg.datasets_dir / dataset
    if not dataset_path.exists():
        raise ValueError(f"dataset not found: {dataset_path}")

    nodes = []
    for name in node_names:
        node = nodes_mod.get_node(cfg, name)
        if not node:
            raise ValueError(f"node '{name}' is not registered (see: cluster node-add)")
        nodes.append(node)

    run_name = uuid.uuid4().hex[:8]
    world_size = 1 + len(nodes)
    master_addr = detect_lan_ip()
    master_port = rendezvous_port or _free_port()
    output = output or str(cfg.checkpoints_dir / f"cluster-{run_name}")

    # Base model transfer: only needed for a local checkpoint dir (an HF hub
    # id like "Qwen/Qwen3-0.6B" is downloaded independently by every rank —
    # nothing to transfer, and it works even if a node has no path in common
    # with this one).
    local_model_path = Path(model)
    base_model_zip: Optional[Path] = None
    if local_model_path.exists() and local_model_path.is_dir():
        zip_path = cfg.data_dir / "cluster_jobs" / f"basemodel-{run_name}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        base_model_zip = _zip_dir(local_model_path, zip_path)

    common = dict(
        world_size=world_size, master_addr=master_addr, master_port=master_port,
        model=model, method=method, epochs=epochs, lr=lr, batch_size=batch_size,
        max_seq=max_seq, run_name=run_name,
    )

    remote_job_ids: dict[str, str] = {}
    for i, node in enumerate(nodes, start=1):
        spec = {**common, "rank": i, "output": None}  # each remote node picks its own local output path
        client = NodeClient(node)
        resp = client.submit_job(spec, dataset_path, base_model_zip=base_model_zip)
        remote_job_ids[node["name"]] = resp["job_id"]

    # Master's own rank 0 — launched in-process, no HTTP self-call needed.
    local_model = str(local_model_path) if base_model_zip is not None else model
    local_job = runner.launch_job(
        state,
        {**common, "rank": 0, "output": output},
        dataset_path,
        model_override=local_model if base_model_zip is not None else None,
    )

    cluster_job = ClusterJob(
        id=run_name,
        local_job_id=local_job.id,
        remote=remote_job_ids,
        nodes=[n["name"] for n in nodes],
        args={**common, "output": output},
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    state.cluster_jobs[run_name] = cluster_job
    return cluster_job


def cluster_job_status(state: EngineState, run_name: str) -> dict[str, Any]:
    cj = state.cluster_jobs.get(run_name)
    if not cj:
        raise ValueError(f"cluster job '{run_name}' not found")
    local = state.train_jobs.get(cj.local_job_id) if cj.local_job_id else None
    result = {
        "id": cj.id,
        "args": cj.args,
        "started_at": cj.started_at,
        "master": {
            "status": local.status if local else "unknown",
            "exit_code": local.exit_code if local else None,
        },
        "nodes": {},
    }
    for name, remote_job_id in cj.remote.items():
        node = nodes_mod.get_node(state.config, name)
        entry: dict[str, Any] = {"job_id": remote_job_id}
        if node:
            try:
                entry.update(NodeClient(node).job_status(remote_job_id))
            except Exception as e:  # noqa: BLE001 - report, don't fail the whole view
                entry["error"] = str(e)
        result["nodes"][name] = entry
    return result
