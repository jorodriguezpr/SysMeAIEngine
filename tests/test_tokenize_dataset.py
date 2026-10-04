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

"""Tests for src.core.tokenize_dataset -- specifically the label-masking
math, which is the correctness-critical part (see the module's own
docstring for the bug this fixes). Kept as a pure-function test with no
real tokenizer/model needed, so it runs fast and offline."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.tokenize_dataset import mask_prompt_labels  # noqa: E402


def test_mask_prompt_labels_masks_prompt_and_pad():
    # 3 prompt tokens, 4 answer tokens, 2 pad tokens (attention_mask=0)
    input_ids = [1, 2, 3, 4, 5, 6, 7, 0, 0]
    attention_mask = [1, 1, 1, 1, 1, 1, 1, 0, 0]
    prompt_len = 3
    labels = mask_prompt_labels(input_ids, attention_mask, prompt_len)
    assert labels[:3] == [-100, -100, -100]
    assert labels[3:7] == [4, 5, 6, 7]
    assert labels[7:] == [-100, -100]


def test_mask_prompt_labels_no_padding():
    input_ids = [1, 2, 3]
    attention_mask = [1, 1, 1]
    labels = mask_prompt_labels(input_ids, attention_mask, prompt_len=1)
    assert labels == [-100, 2, 3]


def test_mask_prompt_labels_prompt_exceeds_sequence_masks_everything():
    # Truncation edge case: prompt_len clamped to len(input_ids) by the
    # caller -- every label becomes -100, a safe zero-loss row, not a crash.
    input_ids = [1, 2, 3]
    attention_mask = [1, 1, 1]
    labels = mask_prompt_labels(input_ids, attention_mask, prompt_len=3)
    assert labels == [-100, -100, -100]


def test_chat_rows_match_inference_format_and_mask_prompt():
    import pytest
    from src.core.tokenize_dataset import chat_prompt_text, chat_training_ids

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as e:  # offline, uncached
        pytest.skip(f"Qwen tokenizer unavailable: {e}")
    p, c = chat_training_ids(tok, "Contexto.\nPregunta: ¿Quién?", "Pedro.", 512)
    # prompt tokens == exactly what HFProvider.generate() feeds at inference
    assert tok.decode(p) == chat_prompt_text(tok, "Contexto.\nPregunta: ¿Quién?")
    assert c[-1] == tok.eos_token_id  # learns to stop
    # overlong prompt: cut from the LEFT, answer and question kept
    p2, c2 = chat_training_ids(tok, "palabra " * 2000 + "\nPregunta final?", "Respuesta.", 128)
    assert len(p2) + len(c2) == 128
    assert tok.decode(c2) == "Respuesta." + tok.eos_token
    assert "Pregunta final?" in tok.decode(p2)


def test_chat_rows_tokenized_dataset_labels_masked():
    import pytest
    from datasets import Dataset
    from src.core.tokenize_dataset import build_tokenized_dataset

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as e:
        pytest.skip(f"Qwen tokenizer unavailable: {e}")
    ds = Dataset.from_list([{"prompt": "Q?", "completion": "A."}])
    row = build_tokenized_dataset(ds, tok, 256)[0]
    kept = [t for t, l in zip(row["input_ids"], row["labels"]) if l != -100]
    assert tok.decode(kept) == "A." + tok.eos_token
