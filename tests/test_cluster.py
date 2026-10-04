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

"""Cluster feature tests.

Node registry and orchestrator helpers are tested directly (fast, no
subprocess). The agent HTTP endpoints are exercised end-to-end via TestClient
with a real WORLD_SIZE=1 job (the TinyGPT provider, so it's fast and offline)
— this is the same "degrades to a normal single-process run" path documented
in scripts/distributed_finetune.py, and is as much of the real upload ->
subprocess -> job-tracking plumbing as can be verified without a second
machine. The actual multi-rank torch.distributed/DDP wiring is verified
separately and manually via scripts/verify_cluster_loopback.py (not part of
this suite — it takes tens of seconds and downloads a real tiny HF model).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cluster import nodes as nodes_mod  # noqa: E402
from src.cluster import orchestrator  # noqa: E402
from src.server import build_app  # noqa: E402


class _FakeConfig:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir


def test_node_registry_crud(tmp_path):
    cfg = _FakeConfig(tmp_path)
    assert nodes_mod.list_nodes(cfg) == []

    entry = nodes_mod.add_node(cfg, "laptop-a", "192.168.1.50", port=11434, api_key="secret")
    assert entry["name"] == "laptop-a"
    assert nodes_mod.base_url(entry) == "http://192.168.1.50:11434"

    assert len(nodes_mod.list_nodes(cfg)) == 1
    assert nodes_mod.get_node(cfg, "laptop-a")["host"] == "192.168.1.50"
    assert nodes_mod.get_node(cfg, "missing") is None

    with pytest.raises(ValueError):
        nodes_mod.add_node(cfg, "laptop-a", "10.0.0.1")

    nodes_mod.remove_node(cfg, "laptop-a")
    assert nodes_mod.list_nodes(cfg) == []

    with pytest.raises(ValueError):
        nodes_mod.remove_node(cfg, "laptop-a")


def test_detect_lan_ip_returns_an_ip_like_string():
    ip = orchestrator.detect_lan_ip()
    assert isinstance(ip, str) and ip.count(".") == 3


def test_free_port_is_actually_bindable():
    import socket

    port = orchestrator._free_port(preferred=29600)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", port))  # would raise OSError if not actually free


@pytest.fixture(scope="module")
def client():
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="sysadmin-engine-cluster-test-")
    os.environ["ENGINE_DATA_DIR"] = temp_dir
    os.environ["ENGINE_RAW_DIR"] = str(Path(temp_dir) / "raw")
    os.environ["ENGINE_CHUNKS_DIR"] = str(Path(temp_dir) / "chunks")
    os.environ["ENGINE_DATASETS_DIR"] = str(Path(temp_dir) / "datasets")
    os.environ["ENGINE_MODELS_DIR"] = str(Path(temp_dir) / "models")
    os.environ["ENGINE_CHECKPOINTS_DIR"] = str(Path(temp_dir) / "checkpoints")
    os.environ["ENGINE_WEIGHTS_DIR"] = str(Path(temp_dir) / "weights")
    return TestClient(build_app())


def test_cluster_info_endpoint(client):
    r = client.get("/cluster/info")
    assert r.status_code == 200
    j = r.json()
    assert "hostname" in j and "torch_version" in j and "devices" in j


def test_cluster_nodes_admin_api(client):
    r = client.post("/api/cluster/nodes", json={"name": "n1", "host": "127.0.0.1", "port": 9999})
    assert r.status_code == 200
    r = client.get("/api/cluster/nodes")
    assert any(n["name"] == "n1" for n in r.json())
    r = client.delete("/api/cluster/nodes/n1")
    assert r.status_code == 200
    r = client.get("/api/cluster/nodes")
    assert not any(n["name"] == "n1" for n in r.json())


def test_agent_job_submission_world_size_1(client):
    """Full round trip through the real HTTP endpoint: multipart dataset
    upload -> subprocess launch -> job tracking -> completion. Uses
    WORLD_SIZE=1 (TinyGPT) so it stays fast and offline, per the module
    docstring."""
    dataset_content = "\n".join(
        json.dumps({"input": f"sample line {i}"}) for i in range(4)
    )
    job_spec = {
        "rank": 0, "world_size": 1,
        "master_addr": "127.0.0.1", "master_port": 29700,
        "model": "minilm", "method": "sft", "epochs": 1,
        "batch_size": 1, "max_seq": 32, "run_name": "pytest",
    }
    r = client.post(
        "/cluster/jobs",
        data={"job_spec": json.dumps(job_spec)},
        files={"dataset": ("ds.jsonl", dataset_content, "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    deadline = time.time() + 60
    status = None
    while time.time() < deadline:
        r = client.get(f"/cluster/jobs/{job_id}")
        status = r.json()
        if status["status"] in ("done", "failed"):
            break
        time.sleep(1)

    log = client.get(f"/cluster/jobs/{job_id}/log").text
    assert status is not None and status["status"] == "done", (status, log)


def test_agent_job_status_404_for_unknown_id(client):
    r = client.get("/cluster/jobs/does-not-exist")
    assert r.status_code == 404


def test_agent_list_jobs_includes_previously_submitted_job(client):
    """The list-jobs endpoint (added after a master crash left an orphaned
    remote rank with no way to find/cancel it, since job ids were otherwise
    only known to the now-dead master process) -- relies on
    test_agent_job_submission_world_size_1 having already run in this
    module-scoped client to have at least one real job to find."""
    r = client.get("/cluster/jobs")
    assert r.status_code == 200
    jobs = r.json()
    assert isinstance(jobs, list) and len(jobs) >= 1
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids))  # no duplicates
    for j in jobs:
        assert set(j.keys()) >= {"id", "status", "exit_code", "started_at", "finished_at", "args"}
