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

"""Prune a checkpoint — magnitude-based pruning on the MLP blocks (TinyGPT)
or a uniform tensor-level pruning for HF checkpoints (saved via safetensors).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path, help="path to minilm.pt or HF dir")
    p.add_argument("--amount", type=float, default=0.1, help="fraction of weights to drop (0-1)")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    import torch

    args.out.mkdir(parents=True, exist_ok=True)

    # TinyGPT checkpoint
    if args.checkpoint.is_file() and args.checkpoint.suffix == ".pt":
        from src.core.providers.minilm import TinyGPT

        model = TinyGPT()
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        with torch.no_grad():
            for name, p in model.named_parameters():
                # skip layer-norm + biases
                if p.ndim < 2:
                    continue
                flat = p.abs().flatten()
                k = max(1, int(flat.numel() * args.amount))
                thresh = torch.kthvalue(flat, k).values
                p.mul_((p.abs() >= thresh).float())
        out_file = args.out / "minilm-pruned.pt"
        torch.save(model.state_dict(), out_file)
        print(f"WROTE {out_file}")
        return 0

    # HF checkpoint dir — use torch/pruning if available
    if args.checkpoint.is_dir():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(str(args.checkpoint))
        try:
            import torch.nn.utils.prune as prune

            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=args.amount)
            model.save_pretrained(args.out)
            print(f"WROTE {args.out} (HF pruned)")
        except ImportError:
            print("torch.nn.utils.prune unavailable — copy checkpoint instead", file=sys.stderr)
            model.save_pretrained(args.out)
        return 0

    print("ERROR: checkpoint must be a .pt file or a HF dir", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
