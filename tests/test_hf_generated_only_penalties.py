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

"""Anti-repetition must apply to generated tokens only. HF's built-in
repetition_penalty/no_repeat_ngram_size also count the prompt, which for
grounded answers (copying names from retrieved passages) garbled output:
"comadrona Zelomi" became "comadrona Zalomi" with a Cyrillic letter."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.providers.hf import REPETITION_PENALTY, generated_only_penalties  # noqa: E402


def _apply(input_ids, prompt_len, vocab=10):
    scores = torch.full((1, vocab), 2.0)
    procs = generated_only_penalties(prompt_len)
    for p in procs:
        scores = p(torch.tensor([input_ids]), scores)
    return scores[0]


def test_prompt_tokens_are_not_penalized():
    # tokens 1,2,3 only occur in the prompt -> untouched
    s = _apply([1, 2, 3], prompt_len=3)
    assert torch.all(s == 2.0)


def test_generated_tokens_are_penalized():
    s = _apply([1, 2, 3, 5], prompt_len=3)  # 5 was generated
    assert abs(float(s[5]) - 2.0 / REPETITION_PENALTY) < 1e-5
    assert s[1] == 2.0  # prompt token still free to copy


def test_no_repeat_ngram_only_over_generated_tokens():
    # generated: 4 5 6 4 5 -> the 3-gram (4,5,6) would repeat: 6 banned
    s = _apply([7, 8, 4, 5, 6, 4, 5], prompt_len=2)
    assert s[6] == float("-inf")
    # the same pattern living in the PROMPT bans nothing
    s = _apply([4, 5, 6, 4, 5], prompt_len=5)
    assert torch.isfinite(s).all()
