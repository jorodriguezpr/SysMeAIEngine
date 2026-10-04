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

"""Export a registered HF causal LM to ONNX for GPU (DirectML) or NPU (Vitis
AI) inference via onnx_directml.py / onnx_npu.py.

Exports a single-forward-pass graph (input_ids -> logits) — matching what
those providers' naive greedy-decode loop already expects (no KV-cache
reuse, so generation is O(n^2) over length, same as the existing providers).
Deliberately avoids the `optimum` library: it wants to downgrade transformers
to a range incompatible with sentence-transformers/RAG in this project (see
project notes) — plain torch.onnx.export needs nothing beyond what's already
installed.

Uses the legacy TorchScript-based exporter (dynamo=False), not torch's newer
default. Verified directly: the newer exporter produces a graph that crashes
DirectML at both runtime ("Reshape" node, dynamic shapes) and session-init
("parameter is incorrect", static shapes) — the legacy exporter's graph runs
correctly on DmlExecutionProvider (logits match CPU output; argmax identical).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# torch's own ONNX exporter prints unicode (checkmarks etc.) that crashes on
# Windows' default cp1252 console encoding — reconfigure before it gets a
# chance to print anything (same class of bug as cli.py's earlier fix, but
# this one is inside third-party code we can't edit directly)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _LogitsOnly:
    """Wraps a HF causal LM so its forward() returns a plain tensor —
    torch.onnx.export can't trace a ModelOutput dataclass return value."""

    def __new__(cls, base):
        import torch

        class _Wrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = base

            def forward(self, input_ids):
                return self.base(input_ids=input_ids).logits

        return _Wrapped()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local checkpoint path")
    p.add_argument("--output", required=True, type=Path, help="output .onnx file path")
    p.add_argument("--seq-len", type=int, default=32, help="dummy sequence length used for tracing")
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    wrapped = _LogitsOnly(model)

    vocab_size = getattr(tok, "vocab_size", None) or model.config.vocab_size
    dummy_input = torch.randint(0, vocab_size, (1, args.seq_len), dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"exporting to {args.output} (this can take a minute)...", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy_input,),
            str(args.output),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=14,
            dynamo=False,
        )

    # onnx_directml.py / onnx_npu.py load the tokenizer from the same folder
    tok.save_pretrained(str(args.output.parent))
    print(f"DONE — wrote {args.output} and tokenizer files to {args.output.parent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
