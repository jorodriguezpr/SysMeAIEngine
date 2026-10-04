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

"""RAG-QA evaluation loop — measures whether a corpus's RAG index actually
retrieves the right chunk (recall@k), and optionally whether a given model's
generated answers overlap with the expected answer text. Needs no separate
labeled eval set: reuses build_qa_dataset()'s question-generation heuristic
against a held-out slice of the corpus's own already-indexed chunks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--method", default="hybrid", choices=["tfidf", "semantic", "hybrid"])
    p.add_argument("--held-out-fraction", type=float, default=0.1)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--question-mode", default="first_sentence", choices=["first_sentence", "keywords"])
    p.add_argument("--model", default=None, help="if given, also scores generated-answer keyword overlap")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    from src.core import config as config_mod
    from src.core import state as state_mod
    from src.core.rag import evaluate_answers, evaluate_retrieval

    cfg = config_mod.Config.from_env()
    state = state_mod.init_state(cfg)

    result = {
        "retrieval": evaluate_retrieval(
            state, args.corpus, method=args.method,
            held_out_fraction=args.held_out_fraction, top_k=args.top_k, seed=args.seed,
            question_mode=args.question_mode,
        )
    }
    if args.model:
        result["answers"] = evaluate_answers(
            state, args.corpus, args.model, method=args.method,
            held_out_fraction=args.held_out_fraction, top_k=args.top_k,
            max_new_tokens=args.max_new_tokens, seed=args.seed, question_mode=args.question_mode,
        )

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
