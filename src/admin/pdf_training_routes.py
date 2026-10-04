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

"""Train-from-folder UI — point at a folder of PDFs, get a trained +
RAG-indexed model in one step. Job tracking (list/log) is shared with the
regular Training tab via the same `state.train_jobs` registry; jobs launched
here are tagged `kind: "pdf_folder"` so the UI can filter to just its own.
"""
from __future__ import annotations

import subprocess
import sys
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..cluster import nodes as nodes_mod
from ..core.device_select import device_report
from ..core.state import EngineState, TrainJob
from ..core.train_preflight import check_ram_preflight
from ..core.utils.device_probe import probe_memory
from ..api.chat_routes import get_state, require_auth
from .training_admin_routes import _job_dict, _watch_job

router = APIRouter()


@router.get("/ui/train-folder", response_class=HTMLResponse)
def train_folder_page(request: Request, state: EngineState = Depends(get_state)):
    from ..ingestion.ocr import ocr_status

    jobs = [
        _job_dict(j) for j in state.train_jobs.values()
        if j.args.get("kind") == "pdf_folder"
    ]
    return request.app.state.templates.TemplateResponse(
        request,
        name="train_folder.html",
        context={
            "title": "Train from Folder",
            "active": "train_folder",
            "devices": device_report(),
            "memory": probe_memory(),
            "ocr": ocr_status(),
            "jobs": jobs,
            "cluster_nodes": nodes_mod.list_nodes(state.config),
        },
    )


class TrainFolderIn(BaseModel):
    source: str
    job_name: str
    base_model: str = "gpt2"
    epochs: int = 1
    max_rows: int = 5000
    device: str = "auto"
    dataset_format: str = "lm"  # "lm" (default, honest continued-pretraining) | "qa"
    method: str = "sft"  # "sft" (default, full fine-tune) | "lora" | "qlora"
    chunking: str = "window"  # "window" (default) | "structured" | "paragraph"
    # Tokens per training row; anything longer is truncated. 128 = the
    # previous hardcoded value (unchanged default) -- the preview endpoint
    # reports how much text that actually cuts for a given folder/model.
    max_seq: int = 128
    nodes: list[str] = []  # registered cluster node names to distribute training across


class TrainFolderPreviewIn(BaseModel):
    source: str
    dataset_format: str = "lm"
    max_rows: int = 5000
    max_sample_files: int = 8
    chunking: str = "window"
    base_model: str = "gpt2"  # whose tokenizer to count with
    max_seq: int = 128


@router.post("/api/train/from-folder/preview")
def preview_train_from_folder(body: TrainFolderPreviewIn, _: None = Depends(require_auth)):
    """Approximate dataset size WITHOUT running the real job (which does
    full extraction inside a subprocess -- see start_train_from_folder).
    Extracts+chunks+filters just the first `max_sample_files` documents
    (sorted, for determinism) the same way the real job would, then scales
    the resulting row/token counts up proportionally for the rest.

    Token counts are exact for the sampled rows -- tokenized with
    `base_model`'s own tokenizer (files only, no weights), counting the
    same text build_tokenized_dataset() will -- and only the scale-up to
    the unsampled files is an estimate. Also reports how many rows/tokens
    `max_seq` would truncate, which silently discards text otherwise. See
    src/core/token_preview.py.
    """
    from ..core.token_preview import count_row_tokens
    from ..ingestion.pipeline import (
        build_lm_dataset, build_qa_dataset, chunk_text, chunk_text_paragraphs,
        chunk_text_structured, clean_text, extract_any, filter_chunks, find_documents,
        supported_globs,
    )

    chunker = {
        "window": chunk_text, "structured": chunk_text_structured, "paragraph": chunk_text_paragraphs,
    }.get(body.chunking, chunk_text)

    source = Path(body.source)
    if not source.is_dir():
        raise HTTPException(status_code=404, detail=f"folder not found: {source}")
    supported = supported_globs()
    docs_found = find_documents(source)
    if not docs_found:
        raise HTTPException(
            status_code=400,
            detail=f"no supported documents ({', '.join(supported)}) found under {source}",
        )

    sample = docs_found[: max(1, body.max_sample_files)]
    docs = []
    n_failed = 0
    for doc in sample:
        try:
            text = clean_text(extract_any(doc))
        except Exception:
            n_failed += 1
            continue
        if len(text) < 200:
            continue
        for c in chunker(text):
            docs.append({"id": c["id"], "source": doc.stem, "text": c["text"]})
    docs = filter_chunks(docs)
    # cloud_qa's real rows come from the cloud provider at job time (one API
    # call per chunk -- too slow/costly to actually do for a size preview),
    # so it's estimated with the same heuristic qa build "qa" uses; row
    # count should be comparable even though the real content will differ.
    rows = build_lm_dataset(docs) if body.dataset_format == "lm" else build_qa_dataset(docs)

    tc = count_row_tokens(rows, body.base_model, body.max_seq)
    scale = len(docs_found) / len(sample) if sample else 1.0
    estimated_rows_before_cap = round(len(rows) * scale)
    estimated_rows = min(estimated_rows_before_cap, body.max_rows)
    row_scale = estimated_rows / len(rows) if rows else 0.0
    lost_fraction = (
        1 - tc["sample_tokens_trained"] / tc["sample_tokens"] if tc["sample_tokens"] else 0.0
    )
    truncation_warning = None
    if tc["truncation_applies"] and lost_fraction > 0.10:
        truncation_warning = (
            f"{lost_fraction:.0%} of the text would be cut off: {tc['rows_truncated']}/{tc['sample_rows']} "
            f"sampled rows exceed max_seq={body.max_seq} tokens (longest {tc['max_tokens_per_row']}). "
            "Raise Max seq (costs RAM/time) or accept that only the start of each chunk is trained on."
        )

    return {
        "total_files": len(docs_found),
        "sampled_files": len(sample),
        "sample_extract_failures": n_failed,
        "estimated_rows": estimated_rows,
        "estimated_tokens": round(tc["sample_tokens"] * row_scale),
        "estimated_tokens_trained": round(tc["sample_tokens_trained"] * row_scale),
        "avg_tokens_per_row": round(tc["avg_tokens_per_row"], 1),
        "max_tokens_per_row": tc["max_tokens_per_row"],
        "rows_truncated_pct": round(100 * tc["rows_truncated"] / tc["sample_rows"], 1) if tc["sample_rows"] else 0.0,
        "text_truncated_pct": round(100 * lost_fraction, 1),
        "max_seq": body.max_seq,
        "token_count_method": tc["method"],
        "truncation_warning": truncation_warning,
        "capped_by_max_rows": estimated_rows_before_cap > body.max_rows,
        "note": (
            f"estimated from a sample of {len(sample)}/{len(docs_found)} files, scaled up proportionally"
            if len(sample) < len(docs_found)
            else "computed from all files (folder small enough, no sampling needed)"
        ),
    }


@router.post("/api/train/from-folder")
def start_train_from_folder(
    body: TrainFolderIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    source = Path(body.source)
    if not source.is_dir():
        raise HTTPException(status_code=404, detail=f"folder not found: {source}")
    from ..ingestion.pipeline import find_documents, supported_globs

    supported = supported_globs()
    if not find_documents(source):
        raise HTTPException(
            status_code=400,
            detail=f"no supported documents ({', '.join(supported)}) found under {source}",
        )
    if body.nodes and body.base_model in ("minilm", "tinygpt"):
        raise HTTPException(
            status_code=400,
            detail=f"cluster nodes not supported with base_model={body.base_model} -- "
            "TinyGPT is single-node only (see docs/CLUSTER.md)",
        )
    for name in body.nodes:
        if not nodes_mod.get_node(state.config, name):
            raise HTTPException(status_code=400, detail=f"node '{name}' is not registered (see cluster node-add)")
    if body.dataset_format == "cloud_qa":
        from ..core.cloud_distill import ensure_cloud_ready

        try:
            ensure_cloud_ready(state)
        except ValueError as e:
            # fail before spawning the subprocess and running extraction --
            # a job that would fail on its first cloud call anyway shouldn't
            # burn minutes of extraction/chunking first
            raise HTTPException(status_code=400, detail=str(e))

    mem = probe_memory()
    # train_from_folder.py's own --batch-size default (16, not user-settable
    # yet) plus the max_seq the subprocess will actually run with.
    preflight = check_ram_preflight(
        body.base_model, batch_size=16, max_seq=body.max_seq,
        total_ram_gb=mem["total_gb"], available_ram_gb=mem["available_gb"],
        method=body.method,
    )
    if preflight["block"]:
        raise HTTPException(status_code=400, detail=preflight["warning"])

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts" / "train_from_folder.py"),
        "--source", str(source),
        "--job-name", body.job_name,
        "--base-model", body.base_model,
        "--epochs", str(body.epochs),
        "--max-rows", str(body.max_rows),
        "--device", body.device,
        "--dataset-format", body.dataset_format,
        "--method", body.method,
        "--chunking", body.chunking,
        "--max-seq", str(body.max_seq),
    ]
    if body.nodes:
        cmd += ["--nodes", *body.nodes]
    job_id = uuid.uuid4().hex[:12]
    log_path = state.config.checkpoints_dir / f"job-{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)

    job = TrainJob(
        id=job_id,
        args={
            "kind": "pdf_folder",
            "source": str(source),
            "job_name": body.job_name,
            "base_model": body.base_model,
            "epochs": body.epochs,
            "max_rows": body.max_rows,
            "max_seq": body.max_seq,
            "device": body.device,
            "dataset_format": body.dataset_format,
            "method": body.method,
            "chunking": body.chunking,
            "nodes": body.nodes,
            "pid": proc.pid,
        },
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        log_path=str(log_path),
    )
    state.train_jobs[job_id] = job
    _watch_job(proc, job, log_f)
    result = {"job_id": job_id, "log_path": str(log_path)}
    if preflight["warning"]:
        result["ram_warning"] = preflight["warning"]
    return result
