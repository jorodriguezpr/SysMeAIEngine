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

"""Rolling usage metrics + `/api/metrics` payload builder."""
from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional

from .state import EngineState
from .utils.device_probe import probe_devices

_STARTED = time.time()
_LATENCIES_MS: deque[float] = deque(maxlen=256)


def record_request(state: EngineState) -> None:
    state.metrics["requests"] = state.metrics.get("requests", 0) + 1


def record_error(state: EngineState) -> None:
    state.metrics["errors"] = state.metrics.get("errors", 0) + 1


def record_chat(state: EngineState, tokens_in: int, tokens_out: int, elapsed_ms: float) -> None:
    record_request(state)
    state.metrics["chat_requests"] = state.metrics.get("chat_requests", 0) + 1
    state.metrics["tokens_in"] = state.metrics.get("tokens_in", 0) + tokens_in
    state.metrics["tokens_out"] = state.metrics.get("tokens_out", 0) + tokens_out
    _LATENCIES_MS.append(elapsed_ms)


def record_embed(state: EngineState, tokens_in: int) -> None:
    record_request(state)
    state.metrics["embedding_requests"] = state.metrics.get("embedding_requests", 0) + 1
    state.metrics["tokens_in"] = state.metrics.get("tokens_in", 0) + tokens_in


def token_estimate(text: str) -> int:
    """Rough 1-token ≈ 4-chars estimate — good enough for dashboards."""
    return max(1, len(text) // 4)


def recent_window(n: int = 20) -> list[float]:
    return list(_LATENCIES_MS)[-n:]


def avg_latency_ms() -> float:
    if not _LATENCIES_MS:
        return 0.0
    return round(sum(_LATENCIES_MS) / len(_LATENCIES_MS), 1)


def tok_s() -> float:
    """Crude sustained tokens-per-second over recent completions."""
    if not _LATENCIES_MS or len(_LATENCIES_MS) < 2:
        return 0.0
    # use the time span of the recorded latencies
    total_s = sum(_LATENCIES_MS) / 1000.0
    return round(len(_LATENCIES_MS) / total_s, 2) if total_s else 0.0


def snapshot(state: EngineState, full_devices: bool = True) -> dict[str, Any]:
    devices = probe_devices() if full_devices else {}
    return {
        "uptime_s": round(time.time() - _STARTED, 0),
        "host": state.config.host,
        "port": state.config.port,
        "loaded_model": state.loaded.name if state.loaded else None,
        "model_registry_size": len(list(state.config.models_dir.glob("*.json"))),
        "metrics": dict(state.metrics),
        "latency_ms_avg": avg_latency_ms(),
        "latency_ms_recent": recent_window(),
        "tokens_per_s": tok_s(),
        "devices": devices,
        "agent": {
            "last_topic": state.agent.last_topic,
            "last_run_at": state.agent.last_run_at,
            "last_results": state.agent.last_results,
            "history_len": len(state.agent.history),
        },
        "train_jobs": len(state.train_jobs),
    }
