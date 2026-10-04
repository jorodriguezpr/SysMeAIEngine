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

"""Tests for src.core.train_preflight -- the RAM pre-flight estimate used
before launching a fine-tuning job."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.train_preflight import check_ram_preflight, estimate_training_ram_gb  # noqa: E402


def test_estimate_known_model_plausible_order_of_magnitude():
    est = estimate_training_ram_gb("gpt2", batch_size=16, max_seq=128)
    assert est["matched_known_model"] is True
    assert 1 <= est["estimated_gb"] <= 10


def test_estimate_unknown_model_uses_fallback_without_raising():
    est = estimate_training_ram_gb("some/totally-unknown-model-id", batch_size=16, max_seq=128)
    assert est["matched_known_model"] is False
    assert est["estimated_gb"] > 0


def test_preflight_blocks_when_estimate_exceeds_total_ram():
    result = check_ram_preflight("gpt2", batch_size=16, max_seq=128, total_ram_gb=0.5, available_ram_gb=0.3)
    assert result["block"] is True
    assert "exceeds this machine's total RAM" in result["warning"]


def test_preflight_soft_warns_when_close_to_available_but_not_total():
    # gpt2 estimate is a few GB; give it a total that's comfortably above
    # the estimate but available RAM tight enough to trigger the soft warn.
    est = estimate_training_ram_gb("gpt2", batch_size=16, max_seq=128)
    total = est["estimated_gb"] * 10
    available = est["estimated_gb"] * 1.1  # just above the estimate, under the 70% margin check
    result = check_ram_preflight("gpt2", batch_size=16, max_seq=128, total_ram_gb=total, available_ram_gb=available)
    assert result["block"] is False
    assert result["warning"] is not None


def test_preflight_no_warning_when_plenty_of_ram():
    result = check_ram_preflight("gpt2", batch_size=16, max_seq=128, total_ram_gb=1000, available_ram_gb=900)
    assert result["block"] is False
    assert result["warning"] is None


def test_qwen_1_5b_is_a_known_model():
    est = estimate_training_ram_gb("Qwen/Qwen2.5-1.5B-Instruct", batch_size=16, max_seq=128, method="lora")
    assert est["matched_known_model"] is True
    assert est["assumed_params_m"] == 1540


def test_lora_estimate_is_much_lower_than_sft_for_the_same_model():
    """Regression test: this project defaults to full fine-tuning, but a
    LoRA job only needs gradients/optimizer state for the tiny adapter, not
    every base-model weight -- estimating LoRA with the full-fine-tune
    formula (16 bytes/param) is 3-4x too pessimistic and could hard-block a
    job that would actually run fine (verified live: qwen2.5-1.5b-instruct
    loaded and ran real inference at ~6GB)."""
    sft_est = estimate_training_ram_gb("Qwen/Qwen2.5-1.5B-Instruct", method="sft")
    lora_est = estimate_training_ram_gb("Qwen/Qwen2.5-1.5B-Instruct", method="lora")
    assert lora_est["estimated_gb"] < sft_est["estimated_gb"] / 2


def test_lora_preflight_does_not_block_where_sft_would():
    # 1.5B params * 16 bytes/param (sft) ~= 24.6GB -- exceeds a 16GB machine
    # and should hard-block; the same model with method="lora" (~4 bytes/
    # param ~= 6.2GB plus a small activation term) should not.
    sft_result = check_ram_preflight(
        "Qwen/Qwen2.5-1.5B-Instruct", batch_size=16, max_seq=128,
        total_ram_gb=16, available_ram_gb=8, method="sft",
    )
    lora_result = check_ram_preflight(
        "Qwen/Qwen2.5-1.5B-Instruct", batch_size=16, max_seq=128,
        total_ram_gb=16, available_ram_gb=8, method="lora",
    )
    assert sft_result["block"] is True
    assert lora_result["block"] is False
