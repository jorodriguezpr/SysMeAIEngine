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

"""Unified operator CLI for SysMeAIEngine: engine/model/chat/ingest/train/weights/agent/metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import config as config_mod  # noqa: E402
from src.core import state as state_mod  # noqa: E402
from src.core.model_loader import (  # noqa: E402
    chat as core_chat,
    embeddings as core_embed,
    list_models,
    load_model,
    register_model,
    tensor_stats,
    unload_model,
)
from src.core.metrics import snapshot  # noqa: E402


def _state():
    return state_mod.init_state(config_mod.Config.from_env())


# ── subcommands ─────────────────────────────────────────────────────────────
def cmd_engine(args: argparse.Namespace) -> int:
    if args.action == "serve":
        import uvicorn

        cfg = config_mod.Config.from_env()
        uvicorn.run(
            "src.app:app",
            host=cfg.host,
            port=cfg.port,
            reload=False,
            log_level=cfg.log_level.lower(),
        )
        return 0
    raise SystemExit(f"unknown engine action: {args.action}")


def cmd_model(args: argparse.Namespace) -> int:
    state = _state()
    if args.action == "register":
        m = register_model(
            state, args.name, args.provider, args.path,
            rag_corpus=args.rag_corpus, device=args.device,
        )
        print(json.dumps(m, indent=2))
    elif args.action == "unregister":
        from src.core.model_loader import unregister_model
        unregister_model(state, args.name)
        print("ok")
    elif args.action == "list":
        print(json.dumps(list_models(state), indent=2, default=str))
    elif args.action == "load":
        load_model(state, args.name)
        print(f"loaded {args.name}")
    elif args.action == "unload":
        unload_model(state)
        print("ok")
    elif args.action == "set-rag-corpus":
        from src.core.model_loader import set_rag_corpus
        if not args.rag_corpus:
            raise SystemExit("--rag-corpus is required for `model set-rag-corpus`")
        m = set_rag_corpus(state, args.name, args.rag_corpus)
        print(json.dumps(m, indent=2))
    else:
        raise SystemExit(f"unknown model action: {args.action}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    state = _state()
    out = core_chat(
        state,
        args.prompt,
        model_name=args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(out["text"])
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    state = _state()
    result = core_embed(state, args.text, model_name=args.model)
    print(json.dumps({"dimension": result["dimension"], "data_len": len(result["data"])}, indent=2))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from src.ingestion.pipeline import (
        build_qa_dataset,
        build_sft_dataset,
        chunk_folder,
        clean_text,
        extract_any,
        extract_pdf,
    )
    if args.action == "pdf":
        text = clean_text(extract_pdf(args.file))
        args.out.mkdir(parents=True, exist_ok=True)
        out_file = args.out / (args.file.stem + ".txt")
        out_file.write_text(text, encoding="utf-8")
        print(f"WROTE {out_file}")
    elif args.action == "html":
        text = clean_text(extract_any(args.file))
        args.out.mkdir(parents=True, exist_ok=True)
        out_file = args.out / (args.file.stem + ".txt")
        out_file.write_text(text, encoding="utf-8")
        print(f"WROTE {out_file}")
    elif args.action == "chunk":
        chunks = chunk_folder(args.in_)
        args.out.mkdir(parents=True, exist_ok=True)
        for c in chunks:
            (args.out / f"{c['id']}.txt").write_text(c["text"], encoding="utf-8")
        print(f"WROTE {len(chunks)} chunks -> {args.out}")
    elif args.action == "build-dataset":
        chunks = chunk_folder(args.chunks)
        rows = (
            build_qa_dataset(chunks)
            if args.qa
            else build_sft_dataset(chunks, args.instruction)
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"WROTE {args.out} ({len(rows)} rows)")
    else:
        raise SystemExit(f"unknown ingest action: {args.action}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    import subprocess
    script = ROOT / "scripts" / f"{'finetune_model' if args.action == 'finetune' else 'distill_knowledge'}.py"
    cmd = [sys.executable, str(script)]
    for k, v in vars(args).items():
        if k in ("command", "action", "fn") or v is None:
            continue
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    return subprocess.call(cmd)


def cmd_cluster(args: argparse.Namespace) -> int:
    from src.cluster import nodes as nodes_mod
    from src.cluster import orchestrator
    from src.cluster.client import NodeClient

    state = _state()
    if args.action == "node-add":
        n = nodes_mod.add_node(state.config, args.name, args.host, args.port, args.api_key)
        print(json.dumps(n, indent=2))
    elif args.action == "node-remove":
        nodes_mod.remove_node(state.config, args.name)
        print("ok")
    elif args.action == "node-list":
        print(json.dumps(nodes_mod.list_nodes(state.config), indent=2))
    elif args.action == "node-ping":
        node = nodes_mod.get_node(state.config, args.name)
        if not node:
            raise SystemExit(f"node '{args.name}' not registered")
        ok, msg = NodeClient(node).ping()
        print(json.dumps({"reachable": ok, "detail": msg}, indent=2))
    elif args.action == "node-jobs":
        node = nodes_mod.get_node(state.config, args.name)
        if not node:
            raise SystemExit(f"node '{args.name}' not registered")
        print(json.dumps(NodeClient(node).list_jobs(), indent=2))
    elif args.action == "node-cancel-job":
        if not args.job_id:
            raise SystemExit("--job-id is required for `cluster node-cancel-job`")
        node = nodes_mod.get_node(state.config, args.name)
        if not node:
            raise SystemExit(f"node '{args.name}' not registered")
        print(json.dumps(NodeClient(node).cancel_job(args.job_id), indent=2))
    elif args.action == "train":
        if not args.dataset:
            raise SystemExit("--dataset is required for `cluster train`")
        job = orchestrator.launch_distributed_job(
            state, args.nodes, args.model, args.dataset,
            method=args.method, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, max_seq=args.max_seq, output=args.output,
        )
        print(json.dumps(
            {"id": job.id, "nodes": job.nodes, "local_job_id": job.local_job_id, "remote": job.remote},
            indent=2,
        ))
    elif args.action == "status":
        print(json.dumps(orchestrator.cluster_job_status(state, args.run_name), indent=2))
    else:
        raise SystemExit(f"unknown cluster action: {args.action}")
    return 0


def cmd_weights(args: argparse.Namespace) -> int:
    state = _state()
    if args.action == "inspect":
        rows = tensor_stats(state)
        for r in rows:
            print(
                f"{r['name']:50s} shape={str(r['shape']):20s} mean={r['mean']:.4f} std={r['std']:.4f}"
            )
        return 0
    if args.action == "edit":
        import torch  # lazy
        from src.core.providers.registry import load_provider

        load_model(state, args.model or state.config.default_model)
        loaded = state.loaded
        assert loaded is not None
        for name, p in loaded.model.named_parameters():
            if name == args.tensor:
                with __import__("torch").no_grad():
                    if args.op == "scale":
                        p.mul_(args.value)
                    elif args.op == "bias":
                        p.add_(args.value)
                    elif args.op == "clip":
                        p.clamp_(-abs(args.value), abs(args.value))
                    elif args.op == "sparsify":
                        flat = p.abs().flatten()
                        k = max(1, int(flat.numel() * args.value))
                        thresh = torch.kthvalue(flat, k).values
                        p.mul_((p.abs() >= thresh).float())
                if hasattr(loaded.model, "tokenizer"):
                    ckpt = state.config.weights_dir / loaded.name / "minilm.pt"
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(loaded.model.state_dict(), ckpt)
                print(f"ok — {args.op} on {args.tensor} value={args.value}")
                return 0
        print(f"tensor {args.tensor} not found", file=sys.stderr)
        return 1
    raise SystemExit(f"unknown weights action: {args.action}")


def cmd_agent(args: argparse.Namespace) -> int:
    from scripts.agent_self_evolve import main as agent_main  # type: ignore

    if args.action == "run":
        rc_map = [
            "--topic", args.topic,
            "--max-pages", str(args.max_pages),
            "--out", args.out,
        ]
        sys.argv = ["agent_self_evolve.py"] + rc_map
        return agent_main()
    raise SystemExit(f"unknown agent action: {args.action}")


def cmd_rag(args: argparse.Namespace) -> int:
    from src.core import rag as rag_mod

    state = _state()
    if args.action == "build":
        result = rag_mod.build_index(state, args.source, method=args.method, corpus=args.corpus)
        print(json.dumps(result, indent=2))
    elif args.action == "query":
        if not args.query:
            raise SystemExit("--query is required for `rag query`")
        hits = rag_mod.retrieve(state, args.query, args.top_k, method=args.method, corpus=args.corpus)
        for h in hits:
            print(f"[{h['score']:.3f}] {h['source']}: {h['text'][:150]}...")
    elif args.action == "eval":
        result = {"retrieval": rag_mod.evaluate_retrieval(
            state, args.corpus, method=args.method, held_out_fraction=args.held_out_fraction,
            top_k=args.top_k, question_mode=args.question_mode,
        )}
        if args.model:
            result["answers"] = rag_mod.evaluate_answers(
                state, args.corpus, args.model, method=args.method,
                held_out_fraction=args.held_out_fraction, question_mode=args.question_mode,
            )
        if not args.details:
            for part in result.values():
                part.pop("details", None)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        raise SystemExit(f"unknown rag action: {args.action}")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    state = _state()
    print(json.dumps(snapshot(state), indent=2, default=str))
    return 0


def cmd_kgraph(args: argparse.Namespace) -> int:
    from src.core import kgraph

    state = _state()
    if args.action == "build":
        result = kgraph.build_graph(state, args.corpus, max_chunks=args.max_chunks)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.action == "stats":
        print(json.dumps(kgraph.graph_stats(state, args.corpus), indent=2, ensure_ascii=False))
    elif args.action == "query":
        if not args.query:
            raise SystemExit("--query is required for `kgraph query`")
        names = kgraph.list_corpora(state) if args.corpus == "all" else [args.corpus]
        for name in names:
            res = kgraph.graph_facts(state, name, args.query, max_facts=args.max_facts)
            if res["entities"]:
                print(f"# {name}: entities: {', '.join(e['name'] for e in res['entities'])}")
                for f in res["facts"]:
                    print(f"  {f['subject']} --[{f['relation']}]--> {f['object']}  ({f['source']})")
    elif args.action == "list":
        print(json.dumps(kgraph.list_corpora(state), indent=2))
    elif args.action == "delete":
        print("deleted" if kgraph.delete_graph(state, args.corpus) else "not found")
    else:
        raise SystemExit(f"unknown kgraph action: {args.action}")
    return 0


# ── parser wiring ──────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sysadmin-ai-engine", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("engine")
    e.add_argument("action", choices=["serve"])
    e.set_defaults(fn=cmd_engine)

    m = sub.add_parser("model")
    m.add_argument("action", choices=["register", "unregister", "list", "load", "unload", "set-rag-corpus"])
    m.add_argument("--name", default="minilm")
    m.add_argument("--provider", default="minilm")
    m.add_argument("--path", default="")
    m.add_argument("--rag-corpus", default=None, help="which RAG corpus /api/rag/chat should default to for this model")
    m.add_argument(
        "--device", default=None, choices=["auto", "cpu", "cuda"],
        help="inference device for this model; defaults to ENGINE_DEFAULT_DEVICE ('auto')",
    )
    m.set_defaults(fn=cmd_model)

    c = sub.add_parser("chat")
    c.add_argument("--model", default=None)
    c.add_argument("--prompt", required=True)
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--temperature", type=float, default=0.7)
    c.set_defaults(fn=cmd_chat)

    em = sub.add_parser("embed")
    em.add_argument("--model", default=None)
    em.add_argument("--text", required=True)
    em.set_defaults(fn=cmd_embed)

    ig = sub.add_parser("ingest")
    ig.add_argument("action", choices=["pdf", "html", "chunk", "build-dataset"])
    ig.add_argument("--file", type=Path, required=False)
    ig.add_argument("--out", type=Path, default=Path("data/raw"))
    ig.add_argument("--in_", dest="in_", type=Path, default=Path("data/raw"))
    ig.add_argument("--chunks", type=Path, default=Path("data/chunks"))
    ig.add_argument("--instruction", default="Using the passage, answer:")
    ig.add_argument("--qa", action="store_true")
    ig.set_defaults(fn=cmd_ingest)

    tr = sub.add_parser("train")
    tr.add_argument("action", choices=["finetune", "distill"])
    tr.add_argument("--model", default="minilm")
    tr.add_argument("--dataset", default=None)
    tr.add_argument("--output", default=None)
    tr.add_argument("--source", default=None)
    tr.add_argument("--method", default="lora", choices=["sft", "lora", "qlora"])
    tr.add_argument("--epochs", type=int, default=1)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--batch-size", type=int, default=1)
    tr.add_argument("--max-seq", type=int, default=1024)
    tr.set_defaults(fn=cmd_train)

    w = sub.add_parser("weights")
    w.add_argument("action", choices=["inspect", "edit"])
    w.add_argument("--model", default=None)
    w.add_argument("--tensor", default=None)
    w.add_argument("--op", default="scale", choices=["scale", "bias", "clip", "sparsify"])
    w.add_argument("--value", type=float, default=0.5)
    w.set_defaults(fn=cmd_weights)

    ag = sub.add_parser("agent")
    ag.add_argument("action", choices=["run"])
    ag.add_argument("--topic", required=True)
    ag.add_argument("--max-pages", type=int, default=6)
    ag.add_argument("--out", default="data/raw")
    ag.set_defaults(fn=cmd_agent)

    rg = sub.add_parser("rag")
    rg.add_argument("action", choices=["build", "query", "eval"])
    rg.add_argument("--source", type=Path, default=Path("data/raw"))
    rg.add_argument("--query", default=None)
    rg.add_argument("--top-k", type=int, default=5)
    rg.add_argument("--method", default="hybrid", choices=["tfidf", "semantic", "hybrid"])
    rg.add_argument("--corpus", default="default", help="namespaces the index so multiple trained models don't clobber each other's RAG data")
    rg.add_argument("--question-mode", default="first_sentence", choices=["first_sentence", "keywords"],
                    help="`rag eval`: keywords = 6 exact words from the chunk (lexical test: easy for tfidf, hard for semantic)")
    rg.add_argument("--held-out-fraction", type=float, default=0.1, help="`rag eval`: fraction of chunks to evaluate on")
    rg.add_argument("--model", default=None, help="`rag eval`: also score this model's RAG-grounded answers (slow)")
    rg.add_argument("--details", action="store_true", help="`rag eval`: include per-question details")
    rg.set_defaults(fn=cmd_rag)

    me = sub.add_parser("metrics")
    me.set_defaults(fn=cmd_metrics)

    kg = sub.add_parser("kgraph")
    kg.add_argument("action", choices=["build", "stats", "query", "list", "delete"])
    kg.add_argument("--corpus", default="default", help="corpus name, or 'all' for query/list")
    kg.add_argument("--query", default=None)
    kg.add_argument("--max-chunks", type=int, default=None, help="build: cap cloud cost while testing")
    kg.add_argument("--max-facts", type=int, default=10)
    kg.set_defaults(fn=cmd_kgraph)

    cl = sub.add_parser("cluster")
    cl.add_argument(
        "action",
        choices=[
            "node-add", "node-remove", "node-list", "node-ping", "node-jobs",
            "node-cancel-job", "train", "status",
        ],
    )
    cl.add_argument("--name", default=None, help="node name (node-add/remove/ping/jobs/cancel-job)")
    cl.add_argument("--host", default=None, help="node host/IP (node-add)")
    cl.add_argument("--port", type=int, default=11534, help="node port (node-add)")
    cl.add_argument("--api-key", default=None, help="node's ENGINE_API_KEY, if it has one (node-add)")
    cl.add_argument("--job-id", default=None, help="a job id from `node-jobs` (node-cancel-job)")
    cl.add_argument("--nodes", nargs="*", default=[], help="registered node names to include (train)")
    cl.add_argument("--model", default="minilm", help="HF hub id or local checkpoint path (train)")
    cl.add_argument("--dataset", default=None, help="dataset file under data/datasets/ (train)")
    cl.add_argument("--output", default=None)
    cl.add_argument("--method", default="lora", choices=["sft", "lora", "qlora"])
    cl.add_argument("--epochs", type=int, default=1)
    cl.add_argument("--lr", type=float, default=2e-4)
    cl.add_argument("--batch-size", type=int, default=1)
    cl.add_argument("--max-seq", type=int, default=1024)
    cl.add_argument("--run-name", default=None, help="cluster job id returned by `train` (status)")
    cl.set_defaults(fn=cmd_cluster)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
