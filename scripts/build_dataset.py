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

"""Build a JSONL dataset from a folder of .txt chunks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.pipeline import build_qa_dataset, build_sft_dataset, chunk_text  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--instruction", default="Using the passage, answer:")
    p.add_argument("--chunk-size", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--qa", action="store_true", help="emit question/answer rows instead of sft")
    args = p.parse_args()

    chunks: list[dict] = []
    for txt in sorted(args.chunks.glob("*.txt")):
        text = txt.read_text(encoding="utf-8", errors="ignore")
        for c in chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap):
            c["source"] = txt.name
            chunks.append(c)

    rows = build_qa_dataset(chunks) if args.qa else build_sft_dataset(chunks, args.instruction)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"WROTE {args.out} — {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
