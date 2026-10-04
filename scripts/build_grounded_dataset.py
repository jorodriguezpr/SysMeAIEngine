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

"""P4: build a "grounded exchanges" fine-tuning set -- teach the local model
the FORMAT of grounded answering (use the passages, answer briefly in the
question's language, refuse when they don't cover it), not the book facts.

Each row: {"prompt": rag.build_prompt(question, retrieved hits),
           "completion": <cloud-written faithful answer>}
-- exactly what the local model sees at chat time (see
tokenize_dataset.chat_training_ids for how it's tokenized).

Questions come only from chunks that are NOT gold passages of the grounded
eval set (data/eval/<corpus>-grounded.json), so before/after eval numbers
aren't inflated by training on the test questions. Retrieval uses the same
settings the offline answerer will use (--top-k, baseline features).

  python scripts/build_grounded_dataset.py --corpus aprocrifos-v2 --n 120 --n-refusal 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONCISE = (
    "\n\n(Answer in 1-3 sentences, in the same language as the question, "
    "plain text without markdown.)"
)

REFUSAL_TOPICS_PROMPT = (
    "Write {n} diverse, natural questions in Spanish that a person might ask a "
    "chatbot, about everyday topics completely unrelated to ancient religious "
    "or apocryphal texts (science, sports, cooking, geography, technology, "
    "history of other eras...). One question per line, no numbering, nothing else."
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--n", type=int, default=120, help="answerable training questions")
    p.add_argument("--n-refusal", type=int, default=20, help="out-of-corpus questions (answer = refusal)")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    from src.core import config as config_mod
    from src.core import grounded_eval as ge
    from src.core import rag
    from src.core import state as state_mod

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    state = state_mod.init_state(config_mod.Config.from_env())
    chunks = json.loads(rag._paths(state, "semantic", args.corpus)["meta"].read_text(encoding="utf-8"))
    try:
        held_out = {g for it in ge.load_eval_set(state, args.corpus)["items"] for g in it["gold"]}
    except FileNotFoundError:
        held_out = set()
        log("WARNING: no grounded eval set found -- nothing held out")
    pool = [c for c in chunks if c["id"] not in held_out and len(c["text"]) >= 300]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    log(f"{len(pool)} candidate chunks ({len(held_out)} eval gold chunks held out)")

    def make_question(c: dict) -> str | None:
        qa = ge._parse_json(ge._cloud(state, ge._single_prompt(c["text"])))
        return qa.get("question") if qa else None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        questions = [q for q in ex.map(make_question, pool[: int(args.n * 1.2)]) if q][: args.n]
    log(f"{len(questions)} answerable questions")

    raw = ge._cloud(state, REFUSAL_TOPICS_PROMPT.format(n=args.n_refusal))
    refusal_qs = [ln.strip("-• ").strip() for ln in raw.splitlines() if ln.strip().endswith("?")][: args.n_refusal]
    log(f"{len(refusal_qs)} out-of-corpus questions")

    # retrieval sequential (shared CPU models), answering parallel (cloud)
    prompts = []
    for q in questions + refusal_qs:
        hits, _ = rag.retrieve_grounded(state, q, top_k=args.top_k, corpus=args.corpus,
                                        multi_query=False, parent_child=False,
                                        kgraph_enabled=False, mmr=False, allow_cloud=False)
        prompts.append((q, rag.build_prompt(q, hits)))

    def answer(item: tuple[str, str]) -> dict | None:
        q, prompt = item
        for _ in range(3):
            a = ge._cloud(state, prompt + CONCISE).strip()
            if a:
                return {"prompt": prompt, "completion": a, "question": q,
                        "refusal_expected": q in refusal_qs}
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for r in ex.map(answer, prompts) if r]

    out = args.out or state.config.datasets_dir / f"{args.corpus}-grounded-chat.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}, ensure_ascii=False) + "\n")
    refused = sum(1 for r in rows if rag.GROUNDING_REFUSAL.lower()[:40] in r["completion"].lower())
    log(f"wrote {out}: {len(rows)} rows ({refused} refusals; "
        f"{sum(1 for r in rows if r['refusal_expected'])} were out-of-corpus questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
