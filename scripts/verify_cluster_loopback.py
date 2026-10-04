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

"""Sanity-check the distributed training wiring on THIS machine alone, before
trying it across a real cluster.

Launches N local processes (default 2) of distributed_finetune.py against a
tiny local dataset and a tiny local HF model config (built on the fly — no
network download needed), all talking to each other over 127.0.0.1 via the
Gloo backend. Confirms torch.distributed rendezvous + DDP gradient sync +
rank-0-only checkpoint saving actually complete successfully end to end.

This does NOT verify real multi-machine networking, Windows Firewall rules,
or real CUDA participation — only that the distributed protocol wiring
itself (env vars, backend choice, save-on-rank-0) is correct. See
docs/CLUSTER.md for what still needs verifying on a real cluster.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, default=2)
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="cluster_loopback_"))
    try:
        # tiny local model — no network needed
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

        model_dir = work / "tiny_model"
        # vocab_size must match the real GPT2 tokenizer below (50257) — token
        # ids it produces would index out of range against a smaller embedding
        # table. Only depth/width are shrunk for speed.
        cfg = GPT2Config(n_embd=16, n_layer=1, n_head=1, n_positions=64)
        GPT2LMHeadModel(cfg).save_pretrained(model_dir)
        # a real vocab/merges file is still needed for tokenization — reuse
        # GPT2's public tokenizer files (small, cached after first download)
        # rather than the randomly-initialized weights above.

        AutoTokenizer.from_pretrained("gpt2").save_pretrained(model_dir)

        dataset_path = work / "tiny.jsonl"
        dataset_path.write_text(
            "\n".join(
                json.dumps({"input": f"sample training line number {i}"}) for i in range(8)
            ),
            encoding="utf-8",
        )

        port = _free_port()
        procs = []
        logs = []
        for rank in range(args.world_size):
            out_dir = work / f"out-rank{rank}"
            log_path = work / f"rank{rank}.log"
            logs.append(log_path)
            env = {
                "RANK": str(rank), "WORLD_SIZE": str(args.world_size), "LOCAL_RANK": "0",
                "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
            }
            import os

            cmd = [
                sys.executable, str(ROOT / "scripts" / "distributed_finetune.py"),
                "--model", str(model_dir), "--dataset", str(dataset_path),
                "--output", str(out_dir), "--method", "sft", "--epochs", "1",
                "--batch-size", "1", "--max-seq", "16",
            ]
            log_f = open(log_path, "w", encoding="utf-8")
            procs.append(subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env={**os.environ, **env}))

        ok = True
        for rank, proc in enumerate(procs):
            code = proc.wait(timeout=300)
            if code != 0:
                ok = False
                print(f"[verify] rank {rank} FAILED (exit {code}) — see {logs[rank]}")
            else:
                print(f"[verify] rank {rank} OK")

        if ok:
            saved = list((work / "out-rank0").glob("*.safetensors")) + list((work / "out-rank0").glob("pytorch_model*.bin"))
            if not saved:
                ok = False
                print("[verify] rank 0 finished but no model weights were saved")
            else:
                print(f"[verify] rank 0 saved: {[p.name for p in saved]}")
            # distributed_finetune.py itself raises if model_wrapped isn't
            # DistributedDataParallel post-train (world_size>1), so a clean
            # exit already proves this — this just surfaces the confirmation
            # line so it's visible without opening the per-rank logs.
            for rank, lp in enumerate(logs):
                text = lp.read_text(encoding="utf-8", errors="ignore")
                line = next((l for l in text.splitlines() if "model_wrapped=" in l), None)
                print(f"[verify] rank {rank}: {line or '(no model_wrapped line found)'}")

        print("\n[verify] RESULT:", "PASS" if ok else "FAIL")
        if not ok:
            for lp in logs:
                print(f"\n--- {lp} (tail) ---")
                print(lp.read_text(encoding="utf-8", errors="ignore")[-2000:])
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
