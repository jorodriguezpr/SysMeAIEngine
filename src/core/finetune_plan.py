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

"""Shared HF fine-tune precision/method resolution.

Used by both the single-node (scripts/finetune_model.py) and distributed
(scripts/distributed_finetune.py) training entry points so the fp16/
GradScaler-safety reasoning below is decided in exactly one place instead of
drifting between two copies.
"""
from __future__ import annotations

from typing import Any


def resolve_grad_accum(
    num_rows: int, batch_size: int, requested_grad_accum: int, epochs: int, min_steps: int = 20,
) -> int:
    """Caps gradient_accumulation_steps so a small dataset still gets a
    meaningful number of real optimizer steps, instead of silently
    collapsing to just 1-2 for the whole run.

    Confirmed live: the defaults (batch_size=16, grad_accum=8 -> effective
    batch 128) against a 201-row dataset produced exactly 2 steps total for
    the entire epoch -- nowhere near enough for a LoRA adapter (or any
    fine-tune) to learn anything meaningful, and the resulting model's
    chat answers were consequently barely different from the untouched
    base model. See docs/TRAINING.md.

    Only ever lowers grad_accum, never raises it -- a value already at or
    below the floor this would compute is left alone. If even grad_accum=1
    can't reach `min_steps` (a very small dataset relative to batch_size),
    this returns 1 rather than pretending a lower value exists -- reducing
    batch_size or raising epochs are the remaining levers, both left to
    the caller since they trade off memory/wall-clock differently.
    """
    if requested_grad_accum <= 1:
        return requested_grad_accum
    effective_batch = batch_size * requested_grad_accum
    steps_at_requested = max(1, (num_rows * epochs) // effective_batch)
    if steps_at_requested >= min_steps:
        return requested_grad_accum
    capped = (num_rows * epochs) // (batch_size * min_steps)
    return min(requested_grad_accum, max(1, capped))


def resolve_finetune_plan(method: str, use_cuda: bool) -> tuple[str, bool, Any, bool]:
    """Returns (method, use_peft, dtype, fp16).

    fp16 base weights are fine when only a small LoRA adapter is trained on
    top (the frozen base never needs a gradient), but for a full ("sft")
    fine-tune every trainable param would be fp16, which crashes Trainer's
    GradScaler ("Attempting to unscale FP16 gradients"). peft missing is
    exactly the case that silently downgrades lora -> sft, so that fallback
    must be resolved before dtype/fp16 are decided, not after.
    """
    import torch  # lazy

    use_peft = method in ("lora", "qlora")
    if use_peft:
        try:
            import peft  # noqa: F401
        except ImportError:
            print("[warn] peft not installed; falling back to SFT")
            method = "sft"
            use_peft = False

    full_finetune = not use_peft
    dtype = torch.float32 if (not use_cuda or full_finetune) else torch.float16
    fp16 = use_cuda and not full_finetune
    return method, use_peft, dtype, fp16


# Ordered by which architecture family each name pattern belongs to, not by
# preference -- the first family whose modules are ALL present in the model
# wins, since a partial match (e.g. only "query" without "value") isn't
# usable for LoRA either way.
_LORA_TARGET_FAMILIES: list[tuple[str, ...]] = [
    ("q_proj", "v_proj"),          # Llama/Qwen/Mistral-style
    ("c_attn",),                   # GPT-2-style (fused qkv in one Linear)
    ("query_key_value",),          # GPT-NeoX/Pythia-style (fused qkv)
    ("query", "value"),            # BERT/RoBERTa-style attention naming
]


def detect_lora_target_modules(model: Any) -> list[str]:
    """Inspects the model's actual module names instead of assuming Llama/
    Qwen-style q_proj/v_proj, which silently doesn't exist on gpt2-family
    models (a hardcoded target_modules list here previously meant LoRA was
    only ever reachable for one architecture family -- see
    scripts/train_from_folder.py's own comment on why it never used LoRA at
    all as a result).

    Raises ValueError with the actual leaf module names found, rather than
    letting PEFT fail later with a generic "target modules not found" error
    that gives no hint about what modules this specific model actually has.
    """
    leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    for family in _LORA_TARGET_FAMILIES:
        if all(name in leaf_names for name in family):
            return list(family)
    raise ValueError(
        "Could not find a known LoRA target-module pattern (tried "
        f"{_LORA_TARGET_FAMILIES}) on this model. Actual module names found: "
        f"{sorted(leaf_names)[:40]}{'...' if len(leaf_names) > 40 else ''}. "
        "Use --method sft (full fine-tune) for this architecture instead."
    )
