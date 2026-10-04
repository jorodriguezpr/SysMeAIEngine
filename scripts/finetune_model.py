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

"""Fine-tune a model with SFT / LoRA / QLoRA.

For TinyGPT (minilm) this trains the full model — it's tiny so qlora is meaningless.
For HF models, uses PEFT when method != "sft".
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must happen before numpy/torch is ever imported (MKL/OpenMP size their own
# thread pools at load time, not on a later Python-level call) -- confirmed
# live: without this, CPU training on this project's default (LoRA/SFT,
# torch.set_num_threads(os.cpu_count())) crashed with a Windows
# STATUS_ACCESS_VIOLATION (exit code 3221225477 / 0xC0000005) right at the
# first real training step, every time, on a 16-logical-core Windows
# machine -- consistent with PyTorch's own intra-op thread pool and MKL's
# independent BLAS thread pool oversubscribing each other. Capping both to 1
# fixed it immediately and reproducibly. `setdefault` so an operator who has
# already verified a higher value works on their own machine isn't overridden.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    p.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "directml"],
        help="'auto'/'cuda' use a real NVIDIA GPU when present (both TinyGPT and "
        "HF paths); 'directml' is TinyGPT-only and disabled (see device_select.py)",
    )
    args = p.parse_args()

    if args.grad_accum is None:
        from src.core.config import Config

        args.grad_accum = Config.from_env().train_grad_accum

    args.output.mkdir(parents=True, exist_ok=True)

    # ── TinyGPT path (fast, no HF needed) ───────────────────────────────────
    if args.model in ("minilm", "tinygpt"):
        from src.core.device_select import get_torch_device
        from src.core.providers.minilm import TinyGPT
        import torch
        from torch.nn import functional as F

        device_label, device = get_torch_device(args.device)
        model = TinyGPT().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        rows = [json.loads(l) for l in args.dataset.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"[finetune/TinyGPT] {len(rows)} rows, {args.epochs} epochs, method={args.method} (sft-equivalent for TinyGPT), device={device_label}")
        for epoch in range(args.epochs):
            for i, row in enumerate(rows):
                text = row.get("input") or row.get("text") or ""
                ids = model.encode(text)[: model.cfg.block_size]
                if len(ids) < 2:
                    continue
                x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
                y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                if (i + 1) % args.batch_size == 0:
                    opt.step()
                    opt.zero_grad()
            print(f"[epoch {epoch+1}] loss={loss.item():.4f}")
        out_file = args.output / "minilm.pt"
        __import__("torch").save(model.cpu().state_dict(), out_file)
        print(f"WROTE {out_file}")
        return 0

    # ── HF path ─────────────────────────────────────────────────────────────
    import os

    import torch
    from datasets import load_dataset  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from src.core.device_select import resolve_hf_device_map
    from src.core.finetune_plan import detect_lora_target_modules, resolve_finetune_plan, resolve_grad_accum
    from src.core.tokenize_dataset import build_tokenized_dataset, make_collator

    if args.device == "directml":
        raise SystemExit(
            "--device=directml is TinyGPT-only and disabled for HF models — "
            "HuggingFace's Trainer has no DirectML support. Use 'auto' or 'cuda'."
        )
    device_map = resolve_hf_device_map(args.device)
    use_cuda = device_map == "cuda"
    if not use_cuda:
        # CPU-first project; without this, torch defaults to a thread count
        # that doesn't reliably match the actual core count on every machine.
        torch.set_num_threads(os.cpu_count() or 1)

    args.method, use_peft, dtype, use_fp16 = resolve_finetune_plan(args.method, use_cuda)
    print(f"[finetune/HF] model={args.model} method={args.method} device={device_map} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if use_peft:
        from peft import LoraConfig, get_peft_model  # type: ignore

        target_modules = detect_lora_target_modules(model)
        print(f"[finetune/HF] LoRA target_modules (auto-detected): {target_modules}")
        lcfg = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    ds = load_dataset("json", data_files=str(args.dataset))["train"]
    ds_tok = build_tokenized_dataset(ds, tok, args.max_seq)
    collator = make_collator(tok)

    resolved_grad_accum = resolve_grad_accum(len(ds), args.batch_size, args.grad_accum, args.epochs)
    if resolved_grad_accum != args.grad_accum:
        print(
            f"[finetune/HF] auto-lowered --grad-accum {args.grad_accum} -> {resolved_grad_accum} "
            f"({len(ds)} rows x {args.epochs} epoch(s) / batch_size {args.batch_size} would otherwise "
            "yield too few real training steps to learn anything meaningful -- see docs/TRAINING.md)"
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
        fp16=use_fp16,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tok, data_collator=collator)
    est_steps = max(1, (len(ds) * args.epochs) // (args.batch_size * args.grad_accum))
    print(f"[finetune/HF] ~{est_steps} optimizer step(s) expected "
          f"({len(ds)} rows x {args.epochs} epoch(s), effective batch {args.batch_size * args.grad_accum})")
    t0 = time.time()
    trainer.train()
    trainer.save_model(str(args.output))
    tok.save_pretrained(str(args.output))
    print(f"DONE in {time.time()-t0:.1f}s — saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
