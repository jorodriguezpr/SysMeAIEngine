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

"""Shared HF dataset tokenization + label construction for fine-tuning.

Used by both scripts/finetune_model.py and scripts/distributed_finetune.py
so this logic — and its correctness-critical label masking — lives in
exactly one place instead of drifting between two near-identical copies.

Three dataset row shapes are auto-detected from column names:
  * {"text", "id"}                            -- src.ingestion.pipeline
    .build_lm_dataset() rows: plain continued-pretraining. Labels are the
    full (unmasked) input_ids -- correct here, since there's no prompt/
    answer split to preserve.
  * {"question", "answer", "id"}               -- build_qa_dataset() rows:
    real instruction-tuning. Labels are masked (-100) over the prompt span
    and any padding, so loss is computed only on the answer tokens.
  * {"instruction", "input", "output", "id"}    -- legacy build_sft_dataset()
    rows. Unmasked, full-sequence loss, unchanged from previous behavior --
    kept only for backward compatibility with existing callers
    (src/cli.py, scripts/build_dataset.py, scripts/distill_knowledge.py).

Historical note: this project's "SFT" format previously always produced
`output=""` and computed unmasked full-sequence loss regardless of what was
in `output` -- meaning it was accidentally doing plain continued-pretraining
with a wasted, constant instruction prefix, not real instruction-tuning.
This module is the fix: document training now gets an honest LM format
(no fake instruction wrapper), and real Q&A-style training gets actual
prompt masking -- previously entirely absent, even when `answer` held real
text.
"""
from __future__ import annotations

from typing import Any


def mask_prompt_labels(
    input_ids: list[int], attention_mask: list[int], prompt_len: int
) -> list[int]:
    """labels = input_ids, except -100 for the prompt span (< prompt_len)
    and any padding position (attention_mask == 0) -- so loss is only ever
    computed on real answer tokens, never the prompt or pad tokens.

    Edge case: if the encoded sequence was truncated shorter than the
    prompt alone (a very long question relative to max_seq), prompt_len is
    clamped by the caller to len(input_ids) -- every label in that row
    becomes -100 (zero loss contribution from that row, not a crash).
    """
    return [
        -100 if (i < prompt_len or mask == 0) else tok_id
        for i, (tok_id, mask) in enumerate(zip(input_ids, attention_mask))
    ]


def qa_prompt(question: str) -> str:
    """The prompt prefix QA rows are trained with -- shared with
    src/core/token_preview.py so the preview counts the exact same text."""
    return "Using the passage, answer:\n" + question + "\n"


def chat_prompt_text(tok: Any, prompt: str) -> str:
    """The exact text HFProvider.generate() feeds the model for a plain
    prompt: wrapped as one user turn in the model's chat template, with the
    assistant generation marker -- or the raw prompt for models without a
    template (gpt2). Training on anything else teaches a format the model
    never sees at inference."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    return prompt


def chat_training_ids(tok: Any, prompt: str, completion: str, max_seq: int) -> tuple[list[int], list[int]]:
    """(prompt_ids, completion_ids) for a {"prompt", "completion"} row.

    Completion ends with the end-of-turn token so the model learns to stop.
    When too long, the PROMPT is cut from the left -- same as inference
    (hf.py truncates left), keeping the question at the end of the prompt
    and never losing the answer tokens the loss is computed on."""
    p_ids = tok(chat_prompt_text(tok, prompt), add_special_tokens=False)["input_ids"]
    end = tok.eos_token or ""
    c_ids = tok(completion + end, add_special_tokens=False)["input_ids"]
    c_ids = c_ids[:max_seq]
    room = max_seq - len(c_ids)
    p_ids = p_ids[-room:] if room > 0 else []
    return p_ids, c_ids


def build_tokenized_dataset(ds: Any, tok: Any, max_seq: int) -> Any:
    """ds: a `datasets.Dataset` with one of the three known column-name
    shapes (see module docstring). Returns a tokenized Dataset with an
    explicit `labels` column. No padding is applied here -- padding happens
    dynamically at collation time via `make_collator()`, which is what
    makes this faster than the old fixed `padding="max_length"` approach.
    """
    cols = set(ds.column_names)

    if "prompt" in cols and "completion" in cols:
        def tokenize(batch: dict[str, list]) -> dict[str, list]:
            ids_out: list[list[int]] = []
            labels_out: list[list[int]] = []
            for prompt, completion in zip(batch["prompt"], batch["completion"]):
                p_ids, c_ids = chat_training_ids(tok, prompt, completion, max_seq)
                ids_out.append(p_ids + c_ids)
                labels_out.append([-100] * len(p_ids) + c_ids)
            return {"input_ids": ids_out, "attention_mask": [[1] * len(i) for i in ids_out],
                    "labels": labels_out}

    elif "text" in cols:
        def tokenize(batch: dict[str, list]) -> dict[str, list]:
            enc = tok(batch["text"], truncation=True, max_length=max_seq)
            enc["labels"] = [list(ids) for ids in enc["input_ids"]]
            return enc

    elif "question" in cols and "answer" in cols:
        def tokenize(batch: dict[str, list]) -> dict[str, list]:
            input_ids_out: list[list[int]] = []
            attn_out: list[list[int]] = []
            labels_out: list[list[int]] = []
            for q, a in zip(batch["question"], batch["answer"]):
                prompt = qa_prompt(q)
                prompt_ids = tok(prompt, truncation=True, max_length=max_seq)["input_ids"]
                enc = tok(prompt + a, truncation=True, max_length=max_seq)
                ids = enc["input_ids"]
                mask = enc["attention_mask"]
                prompt_len = min(len(prompt_ids), len(ids))
                input_ids_out.append(ids)
                attn_out.append(mask)
                labels_out.append(mask_prompt_labels(ids, mask, prompt_len))
            return {"input_ids": input_ids_out, "attention_mask": attn_out, "labels": labels_out}

    else:
        # legacy build_sft_dataset shape: instruction/input/output, unmasked
        # (unchanged behavior -- see module docstring)
        def tokenize(batch: dict[str, list]) -> dict[str, list]:
            n = len(batch.get("input", []))
            instructions = batch.get("instruction", [""] * n)
            outputs = batch.get("output", [""] * n)
            texts = [
                (ins + "\n" + inp + "\n" + out).strip()
                for ins, inp, out in zip(instructions, batch.get("input", []), outputs)
            ]
            enc = tok(texts, truncation=True, max_length=max_seq)
            enc["labels"] = [list(ids) for ids in enc["input_ids"]]
            return enc

    return ds.map(tokenize, batched=True, remove_columns=ds.column_names)


def make_collator(tok: Any) -> Any:
    """DataCollatorForSeq2Seq pads `labels` with label_pad_token_id=-100
    without recomputing already-set label values -- unlike
    DataCollatorForLanguageModeling, which would clobber the masked labels
    build_tokenized_dataset() already constructed for the QA path."""
    from transformers import DataCollatorForSeq2Seq

    return DataCollatorForSeq2Seq(tok, label_pad_token_id=-100)
