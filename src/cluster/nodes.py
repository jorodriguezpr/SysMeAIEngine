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

"""Registry of known slave nodes for distributed training.

A "node" here is any other SysMeAIEngine instance reachable over HTTP with
its own API key — the same app this one is, just running elsewhere. There is
no separate "agent build"; every instance can act as master, slave, or both.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


def _registry_file(cfg) -> Path:
    return cfg.data_dir / "cluster_nodes.json"


def _read(cfg) -> list[dict]:
    f = _registry_file(cfg)
    if not f.exists():
        return []
    return json.loads(f.read_text(encoding="utf-8"))


def _write(cfg, entries: list[dict]) -> None:
    _registry_file(cfg).parent.mkdir(parents=True, exist_ok=True)
    _registry_file(cfg).write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def list_nodes(cfg) -> list[dict]:
    return _read(cfg)


def get_node(cfg, name: str) -> Optional[dict]:
    return next((n for n in _read(cfg) if n["name"] == name), None)


def add_node(
    cfg, name: str, host: str, port: int = 11534, api_key: Optional[str] = None,
    scheme: str = "http",
) -> dict:
    entries = _read(cfg)
    if any(n["name"] == name for n in entries):
        raise ValueError(f"Node '{name}' is already registered")
    entry = {
        "name": name,
        "host": host,
        "port": port,
        "scheme": scheme,
        "api_key": api_key,
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    entries.append(entry)
    _write(cfg, entries)
    return entry


def remove_node(cfg, name: str) -> None:
    entries = _read(cfg)
    new = [n for n in entries if n["name"] != name]
    if len(new) == len(entries):
        raise ValueError(f"Node '{name}' not found")
    _write(cfg, new)


def base_url(node: dict[str, Any]) -> str:
    return f"{node.get('scheme', 'http')}://{node['host']}:{node['port']}"
