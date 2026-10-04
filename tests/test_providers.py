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

"""Provider tests — minilm must work offline; others may be skipped."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.providers.minilm import MiniLMProvider  # noqa: E402
from src.core.providers.registry import available_providers  # noqa: E402


def test_provider_registry_lists_minilm():
    assert "minilm" in available_providers()


def test_minilm_load_generate(tmp_path):
    provider = MiniLMProvider()
    model, tok = provider.load("minilm-test", tmp_path)
    assert model is not None and tok is not None
    out = provider.generate(model, tok, "hello", max_new_tokens=4)
    assert isinstance(out, str)


def test_minilm_embed(tmp_path):
    provider = MiniLMProvider()
    model, tok = provider.load("minilm-test", tmp_path)
    vectors = provider.embed(model, tok, ["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 0


def test_minilm_tensor_stats(tmp_path):
    provider = MiniLMProvider()
    model, _ = provider.load("minilm-test", tmp_path)
    stats = model.tensor_stats()
    assert stats and all("name" in s and "shape" in s for s in stats)


def _fake_hf_model_and_tokenizer():
    """A model/tokenizer pair good enough to exercise HFProvider.generate()'s
    real control flow (chat-template check, truncation, generate(), decode,
    EOS-stop detection) without downloading a real model. Uses real torch
    tensors (cheap, no weights involved) so the actual slicing/`.to(device)`
    logic in generate() runs unmodified rather than being mocked away."""
    torch = pytest.importorskip("torch")

    tokenizer = MagicMock()
    tokenizer.chat_template = None  # skip the chat-template branch
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.return_value = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    tokenizer.decode.return_value = "generated text"

    model = MagicMock()
    model.device = "cpu"
    model.config = MagicMock(n_positions=None, max_position_embeddings=1024)
    # 3 prompt tokens + 2 new tokens, the second being EOS -- generate()
    # should recognize this as a natural stop after one round.
    model.generate.return_value = torch.tensor([[10, 11, 12, 99, 1]])
    return model, tokenizer


def test_hf_generate_passes_repetition_guards():
    """Regression test for a real bug found live: a chat answer repeated
    the exact same sentence 15+ times because model.generate() was called
    with no repetition_penalty or no_repeat_ngram_size at all."""
    from src.core.providers.hf import HFProvider

    model, tokenizer = _fake_hf_model_and_tokenizer()
    provider = HFProvider()

    out = provider.generate(model, tokenizer, "hello", max_new_tokens=10, temperature=0.2)

    assert out == "generated text"
    model.generate.assert_called_once()
    _, kwargs = model.generate.call_args
    # guards are now custom logits processors that skip the prompt (see
    # hf.generated_only_penalties / tests/test_hf_generated_only_penalties.py);
    # the prompt-counting built-ins must NOT be used any more
    names = [type(p).__name__ for p in kwargs["logits_processor"]]
    assert names == ["_RepetitionPenalty", "_NoRepeatNGram"]
    assert "repetition_penalty" not in kwargs and "no_repeat_ngram_size" not in kwargs


def test_hf_generate_passes_top_p_top_k_when_sampling():
    """top_p/top_k must reach model.generate() when do_sample is True --
    with neither set, sampling draws from the full vocab weighted only by
    temperature, which is what produced garbled words like "Nicodmo" for
    "Nicodemo" in a real RAG answer."""
    from src.core.providers.hf import HFProvider

    model, tokenizer = _fake_hf_model_and_tokenizer()
    provider = HFProvider()

    provider.generate(model, tokenizer, "hello", max_new_tokens=10, temperature=0.2, top_p=0.85, top_k=40)

    _, kwargs = model.generate.call_args
    assert kwargs["top_p"] == 0.85
    assert kwargs["top_k"] == 40


def test_hf_generate_omits_top_p_top_k_when_greedy():
    """temperature=0 means do_sample=False -- passing top_p/top_k anyway
    would trigger a spurious HF warning that they're being ignored."""
    from src.core.providers.hf import HFProvider

    model, tokenizer = _fake_hf_model_and_tokenizer()
    provider = HFProvider()

    provider.generate(model, tokenizer, "hello", max_new_tokens=10, temperature=0.0)

    _, kwargs = model.generate.call_args
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs
