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

"""Exact token counts for the Train-from-Folder dataset preview.

The preview previously estimated tokens as chars/4. Measured against the
real tokenizers on this project's own corpora that was off by -36% (gpt2,
Spanish text: 2.55 chars/token) to +19% (Qwen2.5, English: 4.78
chars/token) -- and it could not say the thing that actually matters for
training: how much of each row gets cut off at --max-seq. Loading just the
tokenizer (no weights) takes ~1s and tokenizes ~200K chars in ~0.1s, so
exact counts are cheap.
"""
from __future__ import annotations

from typing import Any

from .tokenize_dataset import qa_prompt

_TINYGPT_MODELS = ("minilm", "tinygpt")
_tok_cache: dict[str, Any] = {}


def row_training_text(row: dict[str, Any]) -> str:
    """The exact string build_tokenized_dataset() tokenizes for this row."""
    if "prompt" in row and "completion" in row:
        # chat-format rows; the chat template adds a handful of tokens on top
        return row["prompt"] + row["completion"]
    if "text" in row:
        return row["text"]
    return qa_prompt(row.get("question", "")) + row.get("answer", "")


def load_preview_tokenizer(base_model: str) -> tuple[Any, str | None]:
    """(tokenizer, None) on success, (None, reason) otherwise. Tokenizer
    files only -- never the model weights."""
    if base_model in _tok_cache:
        return _tok_cache[base_model], None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(base_model)
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:200]}"
    _tok_cache[base_model] = tok
    return tok, None


def count_row_tokens(rows: list[dict[str, Any]], base_model: str, max_seq: int) -> dict[str, Any]:
    """Per-sample token statistics for `rows`. Falls back to the old chars/4
    estimate (clearly labeled) when the tokenizer can't be loaded, e.g.
    offline with an uncached hub model."""
    texts = [row_training_text(r) for r in rows]
    if base_model in _TINYGPT_MODELS:
        # TinyGPT is byte-level: one token per UTF-8 byte, exactly.
        lengths = [len(t.encode("utf-8")) for t in texts]
        method = "exact (TinyGPT byte-level: 1 token per UTF-8 byte)"
        truncation_applies = False
    else:
        tok, err = load_preview_tokenizer(base_model)
        if tok is None:
            lengths = [len(t) // 4 for t in texts]
            method = f"estimate (chars/4 -- tokenizer for '{base_model}' unavailable: {err})"
        else:
            # verbose=False: rows longer than the model's own max length are
            # expected here (that's what we're measuring), not an error.
            lengths = [len(ids) for ids in tok(texts, verbose=False)["input_ids"]] if texts else []
            method = f"exact ({base_model} tokenizer)"
        truncation_applies = True

    total = sum(lengths)
    kept = sum(min(n, max_seq) for n in lengths) if truncation_applies else total
    n_truncated = sum(1 for n in lengths if n > max_seq) if truncation_applies else 0
    return {
        "method": method,
        "sample_rows": len(lengths),
        "sample_tokens": total,
        "sample_tokens_trained": kept,
        "avg_tokens_per_row": (total / len(lengths)) if lengths else 0.0,
        "max_tokens_per_row": max(lengths) if lengths else 0,
        "rows_truncated": n_truncated,
        "truncation_applies": truncation_applies,
    }
