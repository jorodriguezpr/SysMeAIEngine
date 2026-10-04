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

"""End-to-end grounded-answer evaluation (see src/core/grounded_eval.py).

  # once per corpus (cloud provider must be enabled):
  python scripts/eval_grounded.py --corpus aprocrifos-v2 --build-set
  # feature ablation, cloud answerer:
  python scripts/eval_grounded.py --corpus aprocrifos-v2 --configs all
  # offline quality: a local model answers, cloud judges:
  python scripts/eval_grounded.py --corpus aprocrifos-v2 --configs baseline --answerer local:aprocrifos-v2
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
    p.add_argument("--build-set", action="store_true", help="(re)generate the eval set first")
    p.add_argument("--set", default="grounded", choices=["grounded", "hard"],
                   help="which saved question set: grounded (v1) or hard (paraphrase/chained/why/near-OOC)")
    p.add_argument("--n-single", type=int, default=20)
    p.add_argument("--n-multi", type=int, default=5)
    p.add_argument("--configs", nargs="+", default=["all"],
                   help="config names from grounded_eval.CONFIGS, or 'all' for every one")
    p.add_argument("--answerer", default="cloud", help="'cloud' or 'local:<registered model>'")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--prompt-style", default="default", choices=["default", "native", "complete", "quote"])
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--kb-threshold", type=float, default=None, help="compiled answerer: rerank score to trust a KB match")
    p.add_argument("--kb-verify", default="strict", choices=["strict", "lenient", "off"])
    p.add_argument("--kb-strategy", default="lookup", choices=["lookup", "facts", "facts+raw"])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    from src.core import config as config_mod
    from src.core import state as state_mod
    from src.core import grounded_eval as ge

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    state = state_mod.init_state(config_mod.Config.from_env())
    if args.build_set and args.set == "hard":
        ev = ge.build_hard_eval_set(state, args.corpus, log=log)
    elif args.build_set:
        ev = ge.build_eval_set(state, args.corpus, args.n_single, args.n_multi, log=log)
    else:
        ev = ge.load_eval_set(state, args.corpus, args.set)
    if args.configs == ["none"]:
        return 0

    names = list(ge.CONFIGS) if args.configs == ["all"] else args.configs
    report: dict = {"corpus": args.corpus, "answerer": args.answerer, "top_k": args.top_k,
                    "prompt_style": args.prompt_style,
                    "n_items": len(ev["items"]), "configs": {}}
    for name in names:
        log(f"config {name} ({args.answerer})...")
        results = ge.run_config(state, ev, ge.CONFIGS[name], args.answerer, args.top_k, args.workers, log,
                                prompt_style=args.prompt_style, max_new_tokens=args.max_new_tokens,
                                kb_threshold=args.kb_threshold, kb_verify=args.kb_verify,
                                kb_strategy=args.kb_strategy)
        report["configs"][name] = {"summary": ge.summarize(results), "results": results}
        log(f"  {name}: {json.dumps(report['configs'][name]['summary'])}")

    out = args.out or (state.config.data_dir / "eval" /
                       f"{args.corpus}-results-{args.answerer.replace(':', '_')}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    cols = ["context_recall_single", "context_recall_multi", "correctness_single", "correctness_multi",
            "accuracy_answerable", "false_refusal_rate", "ooc_refusal_rate", "avg_retrieval_s",
            "avg_answer_s", "avg_context_chars"]
    print("\n| config | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, r in report["configs"].items():
        print(f"| {name} | " + " | ".join(str(r["summary"][c]) for c in cols) + " |")
    print(f"\nWROTE {out}\nNote: the judge is the cloud model; compare configs, don't over-read absolute scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
