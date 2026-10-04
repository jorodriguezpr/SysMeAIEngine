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

"""Tests for src.core.finetune_plan -- specifically detect_lora_target_modules(),
which replaced a hardcoded q_proj/v_proj target list that silently made LoRA
unreachable on gpt2-family models (see scripts/train_from_folder.py's own
former comment on why it never used LoRA at all as a result).

Uses real tiny model architectures (not full pretrained weights -- just the
module structure, built from config classes) so this tests against genuine
HF module-naming conventions rather than a guessed/fictional structure."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.finetune_plan import detect_lora_target_modules, resolve_grad_accum  # noqa: E402


def test_detects_gpt2_style_c_attn():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(n_embd=16, n_layer=1, n_head=1, n_positions=64)
    model = transformers.GPT2LMHeadModel(cfg)
    assert detect_lora_target_modules(model) == ["c_attn"]


def test_detects_qwen_style_q_proj_v_proj():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen2Config(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, intermediate_size=32, vocab_size=100,
    )
    model = transformers.Qwen2ForCausalLM(cfg)
    assert detect_lora_target_modules(model) == ["q_proj", "v_proj"]


def test_raises_clear_error_for_unrecognized_architecture():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Weird(nn.Module):
        def __init__(self):
            super().__init__()
            self.some_unrelated_layer = nn.Linear(4, 4)

    with pytest.raises(ValueError, match="Could not find a known LoRA target-module pattern"):
        detect_lora_target_modules(Weird())


def test_resolve_grad_accum_regression_the_real_bug():
    # The exact real-world numbers that produced only 2 optimizer steps for
    # a whole training run: 201 rows, batch_size=16, grad_accum=8, 1 epoch.
    # ceil(201/16) = 13 steps is the best possible with batch_size=16, so
    # the fix should lower grad_accum to 1 (its floor) rather than pretend
    # it can hit the full min_steps target when the dataset is this small.
    result = resolve_grad_accum(num_rows=201, batch_size=16, requested_grad_accum=8, epochs=1)
    assert result == 1


def test_resolve_grad_accum_leaves_large_dataset_alone():
    # 10,000 rows easily clears min_steps=20 at the requested grad_accum --
    # nothing should change.
    result = resolve_grad_accum(num_rows=10_000, batch_size=16, requested_grad_accum=8, epochs=1)
    assert result == 8


def test_resolve_grad_accum_never_raises_grad_accum():
    # A user-supplied grad_accum=1 is already at the floor -- must not be
    # "corrected" upward even though it doesn't reach min_steps either.
    result = resolve_grad_accum(num_rows=10, batch_size=16, requested_grad_accum=1, epochs=1)
    assert result == 1


def test_resolve_grad_accum_picks_middle_value_when_possible():
    # 2000 rows, batch_size=16, requested grad_accum=8 -> effective batch
    # 128 -> ceil(2000/128) = 15 steps, below min_steps=20. Lowering to
    # grad_accum=6 -> effective batch 96 -> ceil(2000/96) = 21 steps, which
    # clears the floor, so the resolver should land at 6, not collapse all
    # the way to 1.
    result = resolve_grad_accum(num_rows=2000, batch_size=16, requested_grad_accum=8, epochs=1, min_steps=20)
    assert result == 6
    # Sanity-check the claim above against the actual formula.
    assert (2000 * 1) // (16 * result) >= 20
    assert (2000 * 1) // (16 * (result + 1)) < 20


def test_resolve_grad_accum_respects_epochs():
    # More epochs means more total steps at the same grad_accum, so the
    # resolver should land on a higher (less aggressively lowered)
    # grad_accum once epochs push the achievable step count up, even
    # though 5 epochs of 201 rows still doesn't fully clear min_steps=20.
    result_1_epoch = resolve_grad_accum(num_rows=201, batch_size=16, requested_grad_accum=8, epochs=1)
    result_5_epochs = resolve_grad_accum(num_rows=201, batch_size=16, requested_grad_accum=8, epochs=5)
    assert result_1_epoch == 1
    assert result_5_epochs == 3
    assert result_1_epoch < result_5_epochs
