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

"""Storage management — see disk usage and delete models / training
artifacts (raw extracted text, chunks, datasets, RAG indices) to free up
space. Deleting a model's files and deleting its training data are
independent actions: you can wipe a model but keep its source data for
retraining, or clear the ingested text while keeping the trained model."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..api.chat_routes import get_state, require_auth
from ..core.model_loader import (get_model_entry, list_models, unload_model,
                                  unregister_model)
from ..core.state import EngineState

router = APIRouter()


def _dir_size(path: Path) -> int:
    """Fast recursive size (os.scandir avoids per-file Path overhead —
    matters here since chunk folders can hold 20k+ small files)."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    else:
                        total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            pass
    return total


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _model_paths(state: EngineState, name: str) -> list[Path]:
    """Files a *model* owns — trained checkpoint / weights, not its source
    training data (see _job_paths)."""
    cfg = state.config
    paths = []
    ckpt = cfg.checkpoints_dir / name
    if ckpt.exists():
        paths.append(ckpt)
    weights = cfg.weights_dir / name
    if weights.exists():
        paths.append(weights)
    return paths


def _job_paths(state: EngineState, name: str) -> list[Path]:
    """Files a training *job* owns — extracted raw text, chunks, the built
    dataset, and its RAG index. Independent of whether a model with the
    same name is registered."""
    cfg = state.config
    paths = []
    for base in (cfg.raw_dir, cfg.chunks_dir, cfg.data_dir / "rag_index"):
        p = base / name
        if p.exists():
            paths.append(p)
    ds = cfg.datasets_dir / f"{name}-sft.jsonl"
    if ds.exists():
        paths.append(ds)
    return paths


def _all_job_names(state: EngineState) -> set[str]:
    cfg = state.config
    names: set[str] = set()
    for base in (cfg.raw_dir, cfg.chunks_dir, cfg.data_dir / "rag_index"):
        if not base.exists():
            continue
        for p in base.iterdir():
            if not p.is_dir():
                continue
            if p.name == "agent":
                # namespace container for per-topic Self-Learning Agent
                # scrapes (data/raw/agent/<topic>/) — list each topic as its
                # own deletable job, not the container itself
                names.update(f"agent/{sub.name}" for sub in p.iterdir() if sub.is_dir())
            else:
                names.add(p.name)
    if cfg.datasets_dir.exists():
        for p in cfg.datasets_dir.glob("*-sft.jsonl"):
            names.add(p.name[: -len("-sft.jsonl")])
    return names


@router.get("/ui/storage", response_class=HTMLResponse)
def storage_page(request: Request, state: EngineState = Depends(get_state)):
    cfg = state.config

    models_info: list[dict[str, Any]] = []
    for entry in list_models(state)["models"]:
        paths = _model_paths(state, entry["name"])
        size = sum(_dir_size(p) for p in paths)
        models_info.append({**entry, "size": _fmt_size(size), "has_files": bool(paths)})

    jobs_info: list[dict[str, Any]] = []
    for name in sorted(_all_job_names(state)):
        paths = _job_paths(state, name)
        size = sum(_dir_size(p) for p in paths)
        jobs_info.append({"name": name, "size": _fmt_size(size)})

    totals = {
        "models_dir": _fmt_size(_dir_size(cfg.models_dir)),
        "data_dir": _fmt_size(_dir_size(cfg.data_dir)),
    }

    return request.app.state.templates.TemplateResponse(
        request,
        name="storage.html",
        context={
            "title": "Storage",
            "active": "storage",
            "models": models_info,
            "jobs": jobs_info,
            "totals": totals,
        },
    )


class DeleteModelIn(BaseModel):
    name: str


@router.post("/api/storage/delete-model")
def delete_model(
    body: DeleteModelIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    entry = get_model_entry(state, body.name)
    if not entry:
        raise HTTPException(status_code=404, detail=f"model '{body.name}' not registered")
    if state.loaded and state.loaded.name == body.name:
        unload_model(state)
    for p in _model_paths(state, body.name):
        _rm(p)
    try:
        unregister_model(state, body.name)
    except ValueError:
        pass
    return {"deleted": body.name}


class DeleteJobIn(BaseModel):
    name: str


@router.post("/api/storage/delete-job")
def delete_job(
    body: DeleteJobIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    paths = _job_paths(state, body.name)
    if not paths:
        raise HTTPException(status_code=404, detail=f"no training data found for '{body.name}'")
    for p in paths:
        _rm(p)
    return {"deleted": body.name, "paths": [str(p) for p in paths]}
