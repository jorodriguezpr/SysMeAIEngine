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

"""Tests for scripts/train_from_folder.py's --nodes cluster wiring: the
minilm/tinygpt guard, the _poll_cluster_job polling loop, and the
_cleanup_cluster_job termination/cancellation it triggers on failure. These
are pure-Python unit tests against the script's own functions (loaded via
importlib since scripts/ isn't a package) using a fake orchestrator/state --
the full distributed path itself is exercised by tests/test_cluster.py
(WORLD_SIZE=1) and manually via scripts/verify_cluster_loopback.py; a
genuine multi-rank run needs real time/models and is out of scope here.

The cleanup tests exist because a real timed-out job was found, live, still
running hours later on both the master and the remote node -- _poll_cluster_job
used to just give up and return without touching either process.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("train_from_folder", ROOT / "scripts" / "train_from_folder.py")
train_from_folder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_from_folder)


class _FakeOrchestrator:
    """Simulates orchestrator.cluster_job_status()'s return shape across a
    sequence of polls, without spawning any real process."""

    def __init__(self, status_sequence):
        self._sequence = list(status_sequence)
        self.calls = 0

    def cluster_job_status(self, state, run_name):
        self.calls += 1
        idx = min(self.calls - 1, len(self._sequence) - 1)
        return self._sequence[idx]


def _fake_cluster_job(run_name="run1", local_job_id=None, remote=None):
    return SimpleNamespace(id=run_name, local_job_id=local_job_id, remote=remote or {})


def _fake_state(running_processes=None, config=None):
    return SimpleNamespace(running_processes=running_processes or {}, config=config)


def test_poll_cluster_job_succeeds_when_all_ranks_done():
    orch = _FakeOrchestrator([
        {"master": {"status": "running"}, "nodes": {"laptop-a": {"status": "running"}}},
        {"master": {"status": "done"}, "nodes": {"laptop-a": {"status": "done"}}},
    ])
    with patch.object(train_from_folder.time, "sleep"):
        ok = train_from_folder._poll_cluster_job(_fake_state(), orch, _fake_cluster_job(), poll_seconds=0, timeout_minutes=10)
    assert ok is True
    assert orch.calls == 2


def test_poll_cluster_job_fails_fast_on_any_rank_failure():
    # Both polls report the same failed node -- if the loop kept waiting
    # instead of failing fast, this would hang forever in a real run.
    orch = _FakeOrchestrator([
        {"master": {"status": "running"}, "nodes": {"laptop-a": {"status": "failed"}}},
        {"master": {"status": "running"}, "nodes": {"laptop-a": {"status": "failed"}}},
    ])
    with patch.object(train_from_folder.time, "sleep"):
        ok = train_from_folder._poll_cluster_job(_fake_state(), orch, _fake_cluster_job(), poll_seconds=0, timeout_minutes=10)
    assert ok is False
    assert orch.calls == 1  # stopped after the first poll, didn't keep waiting on a dead rank


def test_poll_cluster_job_fails_on_master_failure_even_if_nodes_ok():
    orch = _FakeOrchestrator([
        {"master": {"status": "failed"}, "nodes": {"laptop-a": {"status": "running"}}},
    ])
    with patch.object(train_from_folder.time, "sleep"):
        ok = train_from_folder._poll_cluster_job(_fake_state(), orch, _fake_cluster_job(), poll_seconds=0, timeout_minutes=10)
    assert ok is False


def test_poll_cluster_job_times_out_when_nothing_finishes():
    orch = _FakeOrchestrator([
        {"master": {"status": "running"}, "nodes": {"laptop-a": {"status": "running"}}},
    ])
    # Deterministic clock: deadline computed as 1000.0 + 60s = 1060.0, then
    # the in-loop check sees 1061.0 (past it) -- avoids flaking on real
    # clock resolution for a timeout this short.
    with patch.object(train_from_folder.time, "sleep"), \
         patch.object(train_from_folder.time, "time", side_effect=[1000.0, 1061.0]):
        ok = train_from_folder._poll_cluster_job(_fake_state(), orch, _fake_cluster_job(), poll_seconds=0, timeout_minutes=1)
    assert ok is False
    assert orch.calls == 1


def test_poll_cluster_job_does_not_cleanup_on_success():
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None  # still running
    cj = _fake_cluster_job(local_job_id="local-job-1", remote={})
    state = _fake_state(running_processes={"local-job-1": fake_proc})
    orch = _FakeOrchestrator([{"master": {"status": "done"}, "nodes": {}}])

    with patch.object(train_from_folder.time, "sleep"):
        ok = train_from_folder._poll_cluster_job(state, orch, cj, poll_seconds=0, timeout_minutes=10)

    assert ok is True
    fake_proc.terminate.assert_not_called()


def test_poll_cluster_job_terminates_local_process_on_timeout():
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None  # still running
    cj = _fake_cluster_job(local_job_id="local-job-1", remote={})
    state = _fake_state(running_processes={"local-job-1": fake_proc})
    orch = _FakeOrchestrator([{"master": {"status": "running"}, "nodes": {}}])

    with patch.object(train_from_folder.time, "sleep"), \
         patch.object(train_from_folder.time, "time", side_effect=[1000.0, 1061.0]):
        ok = train_from_folder._poll_cluster_job(state, orch, cj, poll_seconds=0, timeout_minutes=1)

    assert ok is False
    fake_proc.terminate.assert_called_once()


def test_poll_cluster_job_terminates_local_process_on_rank_failure():
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    cj = _fake_cluster_job(local_job_id="local-job-1", remote={})
    state = _fake_state(running_processes={"local-job-1": fake_proc})
    orch = _FakeOrchestrator([{"master": {"status": "running"}, "nodes": {"laptop-a": {"status": "failed"}}}])

    with patch.object(train_from_folder.time, "sleep"):
        ok = train_from_folder._poll_cluster_job(state, orch, cj, poll_seconds=0, timeout_minutes=10)

    assert ok is False
    fake_proc.terminate.assert_called_once()


def test_poll_cluster_job_does_not_terminate_already_finished_local_process():
    fake_proc = MagicMock()
    fake_proc.poll.return_value = 0  # already exited on its own
    cj = _fake_cluster_job(local_job_id="local-job-1", remote={})
    state = _fake_state(running_processes={"local-job-1": fake_proc})
    orch = _FakeOrchestrator([{"master": {"status": "running"}, "nodes": {"laptop-a": {"status": "failed"}}}])

    with patch.object(train_from_folder.time, "sleep"):
        ok = train_from_folder._poll_cluster_job(state, orch, cj, poll_seconds=0, timeout_minutes=10)

    assert ok is False
    fake_proc.terminate.assert_not_called()


def test_poll_cluster_job_cancels_remote_jobs_on_timeout():
    cj = _fake_cluster_job(local_job_id=None, remote={"laptop-a": "remote-job-1"})
    state = _fake_state()
    orch = _FakeOrchestrator([{"master": {"status": "running"}, "nodes": {"laptop-a": {"status": "running"}}}])
    fake_node = {"name": "laptop-a", "host": "1.2.3.4", "port": 11434}
    mock_client_instance = MagicMock()

    with patch.object(train_from_folder.time, "sleep"), \
         patch.object(train_from_folder.time, "time", side_effect=[1000.0, 1061.0]), \
         patch("src.cluster.nodes.get_node", return_value=fake_node), \
         patch("src.cluster.client.NodeClient", return_value=mock_client_instance):
        ok = train_from_folder._poll_cluster_job(state, orch, cj, poll_seconds=0, timeout_minutes=1)

    assert ok is False
    mock_client_instance.cancel_job.assert_called_once_with("remote-job-1")


def test_poll_cluster_job_skips_unregistered_remote_nodes_during_cleanup():
    # If a node was removed from the registry between launch and failure,
    # cleanup should skip it instead of raising.
    cj = _fake_cluster_job(local_job_id=None, remote={"gone-node": "remote-job-1"})
    state = _fake_state()
    orch = _FakeOrchestrator([{"master": {"status": "failed"}, "nodes": {}}])

    with patch.object(train_from_folder.time, "sleep"), \
         patch("src.cluster.nodes.get_node", return_value=None):
        ok = train_from_folder._poll_cluster_job(state, orch, cj, poll_seconds=0, timeout_minutes=10)

    assert ok is False


def test_main_rejects_nodes_with_minilm(capsys):
    argv = [
        "train_from_folder.py", "--source", "some/dir", "--job-name", "j",
        "--base-model", "minilm", "--nodes", "laptop-a",
    ]
    with patch.object(sys, "argv", argv):
        rc = train_from_folder.main()
    assert rc == 1
    assert "not supported" in capsys.readouterr().out


def test_main_rejects_nodes_with_tinygpt(capsys):
    argv = [
        "train_from_folder.py", "--source", "some/dir", "--job-name", "j",
        "--base-model", "tinygpt", "--nodes", "laptop-a",
    ]
    with patch.object(sys, "argv", argv):
        rc = train_from_folder.main()
    assert rc == 1
    assert "not supported" in capsys.readouterr().out
