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

"""Fine-tune entry point launched by src/cluster/runner.py, on every rank.

Reads standard torch.distributed rendezvous env vars (RANK, WORLD_SIZE,
LOCAL_RANK, MASTER_ADDR, MASTER_PORT — the same ones `torchrun` sets, so
HuggingFace's Trainer/Accelerate auto-initializes the process group from
them with no extra code here). WORLD_SIZE<=1 (or unset) degrades to exactly
the same single-process training as scripts/finetune_model.py — this is how
the plumbing (job upload, subprocess launch, log tracking) gets exercised in
tests without needing a real multi-node cluster.

WORLD_SIZE>1 is HF-only (TinyGPT is a toy model with no real payoff from
distributing it — see finetune_model.py's own docstring). This path always:
  * uses the Gloo backend (`ddp_backend="gloo"`), not NCCL — Gloo is the only
    backend that works across a CPU+CUDA-mixed group; NCCL requires every
    rank to have a GPU, which a cluster with a CPU-only master doesn't have.
  * forces float32 everywhere and disables fp16, regardless of what any
    individual rank's local hardware could otherwise use — DDP's gradient
    all-reduce averages raw tensor bytes across ranks, so every rank's
    trainable parameters must share one dtype or the sync breaks. Mixing
    fp16-on-GPU-ranks with fp32-on-the-CPU-rank is exactly the kind of
    cross-rank inconsistency that produces obscure, hard-to-debug distributed
    failures — not worth the speed for a first implementation. See
    docs/CLUSTER.md.

Honesty note: verified via a same-machine loopback smoke test
(scripts/verify_cluster_loopback.py, all-CPU, 2 processes) — not on real
separate machines or real GPUs, since none were available while building
this (see [[project_sysmeaiengine_cuda_support]] for the same caveat on
plain single-node CUDA).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Must happen before numpy/torch is ever imported -- see finetune_model.py's
# matching comment for the live-verified crash this avoids. This file has
# its own copy since it's an independent entry point (not every caller goes
# through finetune_model.py's WORLD_SIZE<=1 fallback import).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import finetune_model`


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v else default


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--method", choices=["sft", "lora", "qlora"], default="lora")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-seq", type=int, default=1024)
    p.add_argument(
        "--grad-accum", type=int, default=None,
        help="defaults to ENGINE_TRAIN_GRAD_ACCUM (Config.train_grad_accum) when omitted",
    )
    args = p.parse_args()

    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)

    if args.grad_accum is None:
        from src.core.config import Config

        args.grad_accum = Config.from_env().train_grad_accum

    args.output.mkdir(parents=True, exist_ok=True)

    if world_size <= 1:
        print("[distributed_finetune] WORLD_SIZE<=1 — delegating to single-process finetune_model.py")
        import finetune_model  # noqa: E402  (scripts/ dir already on sys.path via ROOT insert above... see below)

        return finetune_model.main()

    if args.model in ("minilm", "tinygpt"):
        raise SystemExit(
            "distributed (WORLD_SIZE>1) training is HF-only — TinyGPT is a toy "
            "model with no real payoff from distributing it. Run it single-node."
        )

    for required in ("MASTER_ADDR", "MASTER_PORT"):
        if not os.environ.get(required):
            raise SystemExit(f"WORLD_SIZE={world_size} but {required} is not set")

    import torch
    from datasets import load_dataset  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    import torch.distributed as dist

    from src.core.device_select import cuda_available
    from src.core.finetune_plan import detect_lora_target_modules, resolve_finetune_plan, resolve_grad_accum
    from src.core.tokenize_dataset import build_tokenized_dataset, make_collator

    # Relying on Trainer/Accelerate to auto-detect a distributed job purely
    # from RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT env vars turned out NOT to
    # work reliably (verified live: it silently ran each rank as an
    # independent single-process job with no gradient sync at all, only
    # printing a warning about parallel_mode != DISTRIBUTED — no crash, which
    # is what makes this dangerous to leave to auto-detection). Initializing
    # the process group explicitly, before TrainingArguments/Trainer exist,
    # makes Accelerate detect and reuse this already-initialized group
    # instead of guessing — verified live to actually wrap the model in DDP
    # (see scripts/verify_cluster_loopback.py).
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=rank,
        world_size=world_size,
    )

    use_cuda = cuda_available()
    if not use_cuda:
        # Divide across ranks so N ranks sharing one physical machine (e.g.
        # the loopback verification script) don't each claim every core.
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(1, world_size)))

    args.method, use_peft, dtype, _ = resolve_finetune_plan(args.method, use_cuda)
    # fp16 is deliberately dropped even when resolve_finetune_plan would allow
    # it on this rank's own GPU — see module docstring: every rank must agree
    # on one dtype for DDP's gradient all-reduce to be well-defined.
    dtype = torch.float32

    print(
        f"[distributed_finetune] rank={rank}/{world_size} "
        f"local_device={'cuda' if use_cuda else 'cpu'} method={args.method} "
        f"master={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']} "
        f"backend=gloo dtype={dtype}"
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # No device_map here: that's Accelerate's single-process model-parallel
    # sharding feature and is not what DDP wants — Trainer places this
    # rank's single model replica on its own local device automatically once
    # it sees the distributed environment.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)

    if use_peft:
        from peft import LoraConfig, get_peft_model  # type: ignore

        target_modules = detect_lora_target_modules(model)
        if rank == 0:
            print(f"[distributed_finetune] LoRA target_modules (auto-detected): {target_modules}")
        lcfg = LoraConfig(
            task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lcfg)
        if rank == 0:
            model.print_trainable_parameters()

    ds = load_dataset("json", data_files=str(args.dataset))["train"]
    ds_tok = build_tokenized_dataset(ds, tok, args.max_seq)
    collator = make_collator(tok)

    # Trainer's DistributedSampler shards `ds` across ranks at train() time,
    # so each rank effectively only ever sees ~len(ds)/world_size rows --
    # that's the number that actually determines real step count, not the
    # full dataset size. See finetune_model.py's matching comment/docs/TRAINING.md.
    per_rank_rows = max(1, len(ds) // world_size)
    resolved_grad_accum = resolve_grad_accum(per_rank_rows, args.batch_size, args.grad_accum, args.epochs)
    if resolved_grad_accum != args.grad_accum:
        print(
            f"[distributed_finetune] rank={rank} auto-lowered --grad-accum {args.grad_accum} -> "
            f"{resolved_grad_accum} (~{per_rank_rows} rows/rank x {args.epochs} epoch(s) / batch_size "
            f"{args.batch_size} would otherwise yield too few real training steps -- see docs/TRAINING.md)"
        )
        args.grad_accum = resolved_grad_accum

    targs = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        fp16=False,
        ddp_backend="gloo",
        # Accelerate's PartialState only checks its MULTI_CPU distributed
        # path when cpu=True is explicitly passed through — otherwise a
        # CPU-only rank falls all the way through to DistributedType.NO with
        # no error, which is exactly the "training runs but never syncs
        # gradients" failure caught while building this (see the
        # init_process_group comment above). CUDA ranks get MULTI_GPU
        # correctly either way, so this only needs to flip for CPU ranks.
        use_cpu=not use_cuda,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tok, data_collator=collator)
    print(f"[distributed_finetune] rank={rank} parallel_mode={targs.parallel_mode}")
    t0 = time.time()
    trainer.train()
    # Trainer only actually wraps the model in DDP once train() starts (it's
    # still the plain model at __init__ time) — confirms real gradient sync
    # happened rather than each rank silently training independently, which
    # is exactly the failure mode caught while building this (see the
    # init_process_group comment above: it produced no error, only a log
    # warning, so this check exists to fail loudly instead).
    wrapped = type(trainer.model_wrapped).__name__
    print(f"[distributed_finetune] rank={rank} model_wrapped={wrapped}")
    if world_size > 1 and wrapped != "DistributedDataParallel":
        raise RuntimeError(
            f"expected DistributedDataParallel wrapping with world_size={world_size}, "
            f"got {wrapped} — distributed training did not actually engage"
        )
    # Trainer.save_model()/save_pretrained() already guard on
    # is_world_process_zero() internally — only rank 0 actually writes files,
    # every other rank's call here is a safe no-op.
    trainer.save_model(str(args.output))
    if rank == 0:
        tok.save_pretrained(str(args.output))
    print(f"[distributed_finetune] rank={rank} DONE in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
