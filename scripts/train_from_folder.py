#!/usr/bin/env python
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

"""One-shot pipeline: point at a folder of documents, get a trained +
RAG-indexed model. Scans the folder recursively for .pdf/.txt/.docx/.pptx/
.xlsx files, chunks the text (dropping low-signal/duplicate chunks —
see filter_chunks()), builds a training dataset (representatively sampled
if larger than --max-rows, so training finishes in a reasonable time),
fine-tunes a model, registers it, and builds a hybrid RAG index over the
same source text.

Dataset format defaults to plain continued-pretraining (--dataset-format lm)
rather than a fake instruction-wrapper — see build_lm_dataset()'s docstring
in src/ingestion/pipeline.py for why. --dataset-format qa is available for
real Q&A-style supervised training instead.

Pass --nodes to distribute the fine-tune step across registered cluster
nodes (this machine as rank 0 + one process per node) instead of running it
as a local-only subprocess — see docs/CLUSTER.md. Extraction/chunking/
dataset-build and the final RAG-index build always stay local; only
fine-tuning itself is distributed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cleanup_cluster_job(state, cluster_job) -> None:
    """Kills the local rank-0 process (if still running) and asks every
    remote node to cancel its rank. Called whenever _poll_cluster_job gives
    up on a job -- without this, a hung/stalled job just keeps running
    forever on every machine even after this script has already reported
    failure and exited (confirmed live: two orphaned distributed_finetune.py
    processes were still running hours after a timed-out job's own script
    had already exited)."""
    from src.cluster import nodes as nodes_mod
    from src.cluster.client import NodeClient

    proc = state.running_processes.get(cluster_job.local_job_id)
    if proc is not None and proc.poll() is None:
        log(f"cleanup: terminating local rank-0 process (job {cluster_job.local_job_id})")
        try:
            proc.terminate()
        except Exception as e:
            log(f"cleanup: WARNING: failed to terminate local rank-0 process: {e}")

    for name, remote_job_id in cluster_job.remote.items():
        node = nodes_mod.get_node(state.config, name)
        if not node:
            continue
        log(f"cleanup: cancelling remote job on '{name}' (job {remote_job_id})")
        try:
            NodeClient(node).cancel_job(remote_job_id)
        except Exception as e:
            log(f"cleanup: WARNING: failed to cancel job on '{name}': {e}")


def _poll_cluster_job(state, orchestrator, cluster_job, poll_seconds: int, timeout_minutes: int) -> bool:
    """Blocks until every rank (master + every remote node) of the cluster
    job reaches a terminal state, or `timeout_minutes` elapses. Returns True
    only if every rank finished with status "done" -- on any other outcome,
    cleans up every rank (see _cleanup_cluster_job) before returning False,
    so a failed/timed-out job doesn't keep running unattended.

    Synchronous DDP means one dead/cancelled rank stalls every other rank
    with no error of its own (see docs/CLUSTER.md's "known gaps" and the
    real two-machine test's sleep-kills-job finding) -- so a failed/
    cancelled rank here is treated as a reason to stop waiting immediately,
    not as something the other ranks might still recover from.
    """
    run_name = cluster_job.id
    deadline = time.time() + timeout_minutes * 60
    while True:
        status = orchestrator.cluster_job_status(state, run_name)
        master_status = status["master"]["status"]
        node_statuses = {name: info.get("status", "unknown") for name, info in status["nodes"].items()}
        summary = ", ".join(f"{name}={s}" for name, s in node_statuses.items())
        log(f"cluster job {run_name}: master={master_status}" + (f", {summary}" if summary else ""))

        all_statuses = [master_status, *node_statuses.values()]
        if any(s in ("failed", "cancelled") for s in all_statuses):
            log("ERROR: at least one rank failed or was cancelled -- synchronous DDP means the "
                "rest are likely stalled waiting on it too, not continuing to wait")
            _cleanup_cluster_job(state, cluster_job)
            return False
        if all(s == "done" for s in all_statuses):
            return True
        if time.time() > deadline:
            log(f"ERROR: cluster job did not finish within --cluster-timeout-minutes ({timeout_minutes})")
            _cleanup_cluster_job(state, cluster_job)
            return False
        time.sleep(poll_seconds)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="folder of .pdf and/or .txt files, scanned recursively")
    p.add_argument("--job-name", required=True, help="namespaces data/raw, data/chunks, data/datasets, models/checkpoints")
    p.add_argument("--base-model", default="gpt2")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-rows", type=int, default=2000, help="representative-sample cap so training finishes in a reasonable time")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seq", type=int, default=128)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "directml"])
    p.add_argument(
        "--dataset-format", default="lm", choices=["lm", "qa", "cloud_qa"],
        help="'lm' (default): plain continued-pretraining on the document text -- "
        "the honest format for 'train on my documents'. 'qa': instruction/answer "
        "rows via the first-sentence-as-question heuristic, with masked labels "
        "so loss concentrates on the answer. 'cloud_qa': same masked format, but "
        "each row's question/answer is generated by the configured cloud AI "
        "provider actually reading the chunk, instead of the cheap heuristic -- "
        "needs Settings > Cloud AI provider enabled with a working API key, "
        "and makes one real API call per chunk (bounded by --max-rows).",
    )
    p.add_argument(
        "--chunking", default="window", choices=["window", "structured", "paragraph"],
        help="'window' (default, unchanged behavior): fixed character sliding window, "
        "can cut mid-sentence. 'structured': packs whole sentences per chunk instead. "
        "'paragraph': packs whole paragraphs per chunk (falls back to sentence-packing "
        "for any single paragraph bigger than the chunk size).",
    )
    p.add_argument(
        "--method", default="sft", choices=["sft", "lora", "qlora"],
        help="'sft' (default, unchanged behavior): full fine-tune. 'lora'/'qlora': now "
        "usable on any architecture finetune_model.py's LoRA target-module "
        "auto-detection recognizes (gpt2 included -- previously hardcoded "
        "target modules only existed on Llama/Qwen-style models), needs peft installed.",
    )
    p.add_argument(
        "--nodes", nargs="+", default=[],
        help="registered cluster node names (see 'cluster node-add') to distribute the "
        "fine-tune step across, in addition to this machine as rank 0. When omitted "
        "(default), fine-tuning runs entirely locally as before. Not supported with "
        "--base-model minilm/tinygpt -- see docs/CLUSTER.md.",
    )
    p.add_argument("--cluster-poll-seconds", type=int, default=15)
    p.add_argument(
        "--cluster-timeout-minutes", type=int, default=240,
        help="give up waiting on the cluster job after this long and report failure -- "
        "synchronous DDP means one dead/stalled rank hangs every other rank with no "
        "error (see docs/CLUSTER.md), so this is the only thing that stops an "
        "unrecoverable cluster job from hanging this script forever.",
    )
    args = p.parse_args()

    if args.nodes and args.base_model in ("minilm", "tinygpt"):
        log(f"ERROR: --nodes is not supported with --base-model {args.base_model} -- "
            "TinyGPT is single-node only (see docs/CLUSTER.md)")
        return 1

    from src.core.config import Config
    from src.core import state as state_mod
    from src.core import rag as rag_mod
    from src.core.model_loader import register_model
    from src.ingestion.pipeline import (
        build_lm_dataset, build_qa_dataset, chunk_folder, chunk_text,
        chunk_text_paragraphs, chunk_text_structured, clean_text, extract_any,
        filter_chunks, find_documents, supported_globs,
    )

    _CHUNKERS = {"window": chunk_text, "structured": chunk_text_structured, "paragraph": chunk_text_paragraphs}
    chunker = _CHUNKERS[args.chunking]
    from src.ingestion.manifest import is_stale, load_manifest, record, save_manifest, sweep_removed

    cfg = Config.from_env()
    state = state_mod.init_state(cfg)

    if args.dataset_format == "cloud_qa":
        from src.core.cloud_distill import ensure_cloud_ready

        try:
            ensure_cloud_ready(state)
        except ValueError as e:
            log(f"ERROR: {e}")
            return 1

    raw_dir = cfg.raw_dir / args.job_name
    dataset_path = cfg.datasets_dir / f"{args.job_name}-sft.jsonl"
    checkpoint_dir = cfg.checkpoints_dir / args.job_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1) scan + extract — one dispatcher (extract_any) handles every
    # supported type: PDF (pdfminer), Word/.docx, PowerPoint/.pptx,
    # Excel/.xlsx, and .txt (e.g. pages the Self-Learning Agent already
    # scraped and cleaned, read as-is)
    # Scanned PDFs and image files (.png/.jpg/.tif/...) are OCR'd when the
    # optional OCR packages are installed -- see src/ingestion/ocr.py.
    from src.ingestion.ocr import ocr_status

    SUPPORTED_EXTS = supported_globs()
    docs_found = find_documents(args.source)
    _ocr = ocr_status()
    log("OCR: enabled (scanned PDF pages + image files)" if _ocr["available"] else f"OCR: off ({_ocr['reason']})")
    if not docs_found:
        log(f"ERROR: no supported documents ({', '.join(SUPPORTED_EXTS)}) found under {args.source}")
        return 1
    log(f"found {len(docs_found)} document(s) under {args.source}")

    manifest_path = raw_dir / "_manifest.json"
    manifest = load_manifest(manifest_path)
    n_extracted = n_skipped = n_failed = 0
    for i, doc in enumerate(docs_found, 1):
        out_file = raw_dir / (doc.stem + ".txt")
        if out_file.exists() and not is_stale(manifest, doc):
            log(f"[{i}/{len(docs_found)}] skip (unchanged since last extraction): {doc.name}")
            n_skipped += 1
            continue
        log(f"[{i}/{len(docs_found)}] extracting: {doc.name}")
        try:
            text = clean_text(extract_any(doc))
        except Exception as e:
            log(f"  FAILED: {e}")
            n_failed += 1
            continue
        if len(text) < 200:
            log(f"  skip: only {len(text)} chars (too short / no usable text)")
            continue
        if out_file.resolve() != doc.resolve():
            out_file.write_text(text, encoding="utf-8")
        record(manifest, doc, out_file)
        n_extracted += 1

    for stale_out in sweep_removed(manifest):
        if stale_out.exists():
            stale_out.unlink()
            log(f"removed extracted text for deleted source: {stale_out.name}")
    save_manifest(manifest_path, manifest)
    log(f"extraction: {n_extracted} extracted/updated, {n_skipped} unchanged (skipped), {n_failed} failed")

    # 2) chunk
    log(f"chunking extracted text (--chunking {args.chunking})...")
    docs = chunk_folder(raw_dir, chunker=chunker, tag_source=True)
    if not docs:
        log("ERROR: no usable text extracted from any PDF in this folder")
        return 1
    raw_chunk_count = len(docs)
    docs = filter_chunks(docs)
    log(
        f"{len(docs)} chunks from {len(docs_found)} document(s) "
        f"({raw_chunk_count - len(docs)} dropped: low-signal or exact-duplicate)"
    )

    # 3) build dataset — representative sample (evenly spread) if too large
    if len(docs) > args.max_rows:
        step = max(1, len(docs) // args.max_rows)
        sampled = docs[::step][: args.max_rows]
        log(f"sampling {len(sampled)}/{len(docs)} chunks (every {step}th, spread across all sources) to keep training time reasonable")
    else:
        sampled = docs
    if args.dataset_format == "lm":
        rows = build_lm_dataset(sampled)
    elif args.dataset_format == "qa":
        rows = build_qa_dataset(sampled)
    else:  # cloud_qa
        from src.core.cloud_distill import DEFAULT_PAIRS_PER_CHUNK, build_cloud_qa_dataset

        log(f"cloud_qa: generating up to {DEFAULT_PAIRS_PER_CHUNK} differently-phrased question/answer "
            f"pairs per chunk ({len(sampled)} chunks, one real API call each -- this can take a while "
            "and will incur provider cost)")

        def _log_progress(done: int, total: int, failed: int) -> None:
            if done % 25 == 0 or done == total:
                log(f"  cloud_qa progress: {done}/{total} chunks processed ({failed} failed/skipped so far)")

        rows = build_cloud_qa_dataset(state, sampled, on_progress=_log_progress)
        log(f"cloud_qa: {len(rows)} question/answer pairs from {len(sampled)} chunks "
            f"(avg {round(len(rows) / len(sampled), 1) if sampled else 0} pairs/chunk)")
    log(f"dataset format: {args.dataset_format}")
    if not rows:
        # confirmed live: without this check, an all-failed cloud_qa run
        # (bad key, provider outage, every response unparseable) wrote an
        # empty dataset file and then crashed several minutes later inside
        # HF's `datasets` loader with a bare "StopIteration" -- correct but
        # useless as a diagnostic. Fail here instead, immediately, with a
        # message that actually says what happened.
        log(f"ERROR: dataset build produced 0 rows (format={args.dataset_format}) -- nothing to train on. "
            + ("Check the cloud AI provider is reachable and the API key/model are valid."
               if args.dataset_format == "cloud_qa" else
               "Check the source documents actually contain extractable text."))
        return 1
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"wrote dataset: {dataset_path} ({len(rows)} rows)")

    # 4) fine-tune. Default is full SFT (these models are small enough that
    # full fine-tuning is practical) -- --method lora/qlora is now usable on
    # any architecture finetune_model.py's target-module auto-detection
    # recognizes, not just Llama/Qwen-style models as before.
    if args.nodes:
        from src.cluster import orchestrator

        log(f"preflight: pinging {len(args.nodes)} node(s) ({', '.join(args.nodes)})...")
        preflight = orchestrator.preflight(cfg, args.nodes)
        unreachable = [name for name, info in preflight.items() if not info.get("reachable")]
        if unreachable:
            log(f"ERROR: node(s) not reachable, aborting before launch: {unreachable}")
            for name in unreachable:
                log(f"  {name}: {preflight[name].get('error')}")
            return 1
        log(f"preflight OK: {preflight}")

        log(
            f"starting DISTRIBUTED fine-tune across {1 + len(args.nodes)} rank(s) "
            f"(this machine + {args.nodes}): base_model={args.base_model} method={args.method} "
            f"epochs={args.epochs}"
        )
        cluster_job = orchestrator.launch_distributed_job(
            state, args.nodes, args.base_model, str(dataset_path),
            method=args.method, epochs=args.epochs, batch_size=args.batch_size,
            max_seq=args.max_seq, output=str(checkpoint_dir),
        )
        log(f"cluster job launched: run_name={cluster_job.id}")
        ok = _poll_cluster_job(
            state, orchestrator, cluster_job,
            poll_seconds=args.cluster_poll_seconds, timeout_minutes=args.cluster_timeout_minutes,
        )
        if not ok:
            log("ERROR: distributed fine-tune failed")
            return 1
        log("distributed fine-tune complete")
    else:
        log(f"starting fine-tune: base_model={args.base_model} method={args.method} epochs={args.epochs} device={args.device}")
        cmd = [
            sys.executable, str(ROOT / "scripts" / "finetune_model.py"),
            "--model", args.base_model,
            "--dataset", str(dataset_path),
            "--output", str(checkpoint_dir),
            "--method", args.method,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--max-seq", str(args.max_seq),
            "--device", args.device,
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            log(f"ERROR: fine-tune failed (exit {rc})")
            return rc
        log("fine-tune complete")

    # 5) register the model (HF path also needs the tokenizer saved alongside
    # the weights, which finetune_model.py already does)
    model_name = args.job_name
    try:
        register_model(state, model_name, "hf" if args.base_model not in ("minilm", "tinygpt") else "minilm", str(checkpoint_dir))
        log(f"registered model '{model_name}'")
    except ValueError as e:
        log(f"note: {e}")

    # 6) build a hybrid (semantic + TF-IDF) RAG index over the same source
    # text, so answers can be grounded in the actual documents instead of
    # only what training baked in
    log(f"building hybrid RAG index over the source folder (corpus='{model_name}')...")
    try:
        result = rag_mod.build_index(state, raw_dir, method="hybrid", corpus=model_name)
        embed_note = f", {result['embedded']} newly embedded / {result['reused']} reused" if "embedded" in result else ""
        log(f"RAG index built: {result['chunks']} chunks across {len(result['sources'])} source(s){embed_note}")
    except Exception as e:
        log(f"WARNING: RAG index build failed (model is still trained and registered): {e}")

    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
