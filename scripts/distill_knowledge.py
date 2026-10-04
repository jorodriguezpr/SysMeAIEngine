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

"""Distill knowledge — turn a document into a QA set, then run a short FT pass
that 'teaches' the model the QA pairs (knowledge grounding)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.pipeline import build_qa_dataset, chunk_text, clean_text, extract_any  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path, help="pdf/html/txt/md")
    p.add_argument("--model", required=True, help="model name or HF id")
    p.add_argument("--out", required=True, type=Path, help="checkpoint dir to save")
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()

    if not args.source.exists():
        print(f"ERROR: {args.source} not found", file=sys.stderr)
        return 1

    raw = clean_text(extract_any(args.source))
    chunks = chunk_text(raw)
    qa = build_qa_dataset(chunks)
    print(f"Generated {len(qa)} QA rows from {args.source.name}")

    # write a temporary QA dataset
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)
    qa_path = outdir / f"distill-{args.source.stem}-{int(time.time())}.jsonl"
    with qa_path.open("w", encoding="utf-8") as f:
        for r in qa:
            # reuse SFT format: instruction + Q input + A output
            f.write(
                json.dumps(
                    {
                        "instruction": "Answer the question using ground truth.",
                        "input": r["question"],
                        "output": r["answer"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # run the standard fine-tune script on the QA dataset
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "finetune_model.py"),
        "--model", args.model,
        "--dataset", str(qa_path),
        "--output", str(outdir),
        "--epochs", str(args.epochs),
        "--method", "lora",
    ]
    print("RUN:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc == 0:
        print(f"DISTILL OK -> {outdir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
