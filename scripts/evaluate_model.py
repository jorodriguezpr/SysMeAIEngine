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

"""Quick benchmark — latency + tokens/s over a few prompts, reported as JSON."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="minilm")
    p.add_argument("--prompts", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    from src.core import config as config_mod
    from src.core import state as state_mod
    from src.core.model_loader import chat as core_chat

    cfg = config_mod.Config.from_env()
    state = state_mod.init_state(cfg)

    samples = []
    for i in range(args.prompts):
        prompt = f"Test prompt {i}: explain briefly what you know about AI."
        t0 = time.perf_counter()
        out = core_chat(state, prompt, model_name=args.model, max_new_tokens=args.max_tokens)
        dt = time.perf_counter() - t0
        tok = out["completion_tokens"]
        samples.append(
            {"prompt": i, "seconds": dt, "tokens": tok, "tok_per_s": tok / dt if dt > 0 else 0}
        )

    result = {
        "model": args.model,
        "samples": samples,
        "avg_tok_per_s": statistics.mean(s["tok_per_s"] for s in samples),
        "avg_seconds": statistics.mean(s["seconds"] for s in samples),
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
