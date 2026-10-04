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

"""Runtime singleton for the engine process."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # avoid circular import
    from .config import Config


@dataclass
class LoadedModel:
    name: str
    model: Any = None
    tokenizer: Any = None
    registry_entry: Any = None


@dataclass
class AgentState:
    last_topic: Optional[str] = None
    last_run_at: Optional[str] = None
    last_results: int = 0
    last_files: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    # live-run tracking — status flips to "running" the moment a job launches
    # and back to "idle" once the watcher thread sees the subprocess exit
    status: str = "idle"  # "idle" | "running"
    current_topic: Optional[str] = None
    current_log_path: Optional[str] = None
    current_started_at: Optional[str] = None
    current_out_dir: Optional[str] = None
    last_out_dir: Optional[str] = None


@dataclass
class TrainJob:
    id: str
    args: dict = field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    log_path: Optional[str] = None
    status: str = "running"
    exit_code: Optional[int] = None


@dataclass
class ClusterJob:
    """Master-side view of one distributed run: this node's own local rank-0
    TrainJob id, plus the remote job id returned by each participating node."""
    id: str
    local_job_id: Optional[str] = None
    remote: dict = field(default_factory=dict)  # {node_name: remote_job_id}
    nodes: list[str] = field(default_factory=list)
    args: dict = field(default_factory=dict)
    started_at: Optional[str] = None


class EngineState:
    """Mutable runtime state — one instance per process."""

    def __init__(self, config: "Config") -> None:
        self.config = config
        self.loaded: Optional[LoadedModel] = None
        self.metrics: dict[str, Any] = {
            "requests": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "chat_requests": 0,
            "embedding_requests": 0,
            "errors": 0,
        }
        self.agent = AgentState()
        self.train_jobs: dict[str, TrainJob] = {}
        # subprocess.Popen handles for running_processes — not part of TrainJob
        # since it isn't JSON-serializable and doesn't survive a restart anyway
        self.running_processes: dict[str, Any] = {}
        self.cluster_jobs: dict[str, ClusterJob] = {}


_state: Optional[EngineState] = None


def init_state(config: "Config") -> EngineState:
    global _state
    _state = EngineState(config)
    return _state


def get_state() -> EngineState:
    if _state is None:
        raise RuntimeError("Engine state not initialized; call init_state()")
    return _state
