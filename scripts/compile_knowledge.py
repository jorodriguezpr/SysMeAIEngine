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

"""Build the Knowledge Compiler base for a corpus (see
src/core/knowledge_compiler.py). Resumable: re-running continues where it
stopped.

  # fast build with the cloud model (chat stays 100% local afterwards):
  python scripts/compile_knowledge.py --corpus aprocrifos-v2
  # fully offline build with a local Ollama model (slow on CPU):
  python scripts/compile_knowledge.py --corpus aprocrifos-v2 --generator local:qwen7b-local --workers 1
  # derive a copy WITHOUT some chunks (e.g. the eval set's gold passages) --
  # no new generation, just filter + re-embed:
  python scripts/compile_knowledge.py --corpus aprocrifos-v2 --derive-heldout aprocrifos-v2-heldout
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--generator", default="cloud", help="'cloud' or 'local:<registered model>'")
    p.add_argument("--per-chunk", type=int, default=8)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--derive-heldout", default=None,
                   help="write a filtered copy named this, excluding the grounded eval set's gold chunks")
    args = p.parse_args()

    import numpy as np

    from src.core import config as config_mod
    from src.core import grounded_eval as ge
    from src.core import knowledge_compiler as kc
    from src.core import state as state_mod
    from src.core.rag import _embed_texts

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    state = state_mod.init_state(config_mod.Config.from_env())
    if args.derive_heldout:
        src = state.config.data_dir / "compiled" / args.corpus
        items = json.loads((src / "items.json").read_text(encoding="utf-8"))
        gold = {g for it in ge.load_eval_set(state, args.corpus)["items"] for g in it["gold"]}
        kept = [i for i in items if i["chunk_id"] not in gold]
        dst = state.config.data_dir / "compiled" / args.derive_heldout
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "items.json").write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
        np.save(dst / "q_emb.npy", _embed_texts([i["question"] for i in kept]).astype(np.float32))
        (dst / "meta.json").write_text(json.dumps({"derived_from": args.corpus, "excluded_eval_gold_chunks": len(gold),
                                                    "items": len(kept), "dropped": len(items) - len(kept)}, indent=2))
        log(f"derived {args.derive_heldout}: {len(kept)} items ({len(items) - len(kept)} dropped from {len(gold)} eval chunks)")
        return 0

    kc.compile_corpus(state, args.corpus, args.generator, args.per_chunk, workers=args.workers, log=log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
