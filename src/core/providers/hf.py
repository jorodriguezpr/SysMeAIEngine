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

"""HuggingFace Transformers provider — works on CPU/GPU (DirectML is separate)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM = 3


def generated_only_penalties(prompt_len: int) -> Any:
    """Repetition penalty + no-repeat-ngram applied ONLY to tokens the model
    generates, never to the prompt.

    Why: without anti-repetition a small model locks onto a phrase and
    re-emits it for the whole budget (confirmed live: one sentence repeated
    15+ times). But HF's built-in repetition_penalty/no_repeat_ngram_size
    count the PROMPT too -- for grounded answers, whose job is to copy names
    and phrases from the retrieved passages, that penalizes exactly the
    right tokens and bans every 3-token span of the passages. Measured on
    the grounded eval: "comadronа Zalomi" (Cyrillic 'а') for "comadrona
    Zelomi", "judíacos", "mismopuesiónmente", "comadRONA ZelOMI descUBRIÓ";
    with the prompt excluded the same model spells them correctly.

    prompt_len: number of leading input tokens that are prompt (on a
    continuation round, earlier generated text is part of the input and
    IS penalized).
    """
    import torch  # lazy
    from transformers import LogitsProcessor, LogitsProcessorList

    class _RepetitionPenalty(LogitsProcessor):
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            gen = input_ids[:, prompt_len:]
            if gen.shape[1] == 0:
                return scores
            picked = torch.gather(scores, 1, gen)
            picked = torch.where(picked < 0, picked * REPETITION_PENALTY, picked / REPETITION_PENALTY)
            return scores.scatter(1, gen, picked)

    class _NoRepeatNGram(LogitsProcessor):
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            n = NO_REPEAT_NGRAM
            for b in range(input_ids.shape[0]):
                gen = input_ids[b, prompt_len:].tolist()
                if len(gen) < n:
                    continue
                prefix = tuple(gen[-(n - 1):])
                banned = [gen[i + n - 1] for i in range(len(gen) - n + 1) if tuple(gen[i:i + n - 1]) == prefix]
                if banned:
                    scores[b, banned] = float("-inf")
            return scores

    return LogitsProcessorList([_RepetitionPenalty(), _NoRepeatNGram()])


def _raise_if_unmaterialized(model: Any, model_id: str) -> None:
    """Refuse a model whose weights were never actually loaded.

    Live incident: a server produced multilingual garbage for every prompt
    (even plain chat, even after unload/reload) while the same checkpoint
    loaded fine in any other process. The console showed the real cause
    only on /ui/weights: "Cannot copy out of meta tensor; no data!" --
    from_pretrained() had returned a model with some tensors still on the
    'meta' device (placeholders, no data), e.g. from incomplete base-model
    files in the Hugging Face cache the process used. Generation doesn't
    fail on that; it silently emits noise. Fail loudly instead."""
    import itertools

    meta = [
        name for name, t in itertools.chain(model.named_parameters(), model.named_buffers())
        if t.device.type == "meta"
    ]
    if not meta:
        return
    try:
        from huggingface_hub import constants

        cache = constants.HF_HUB_CACHE
    except Exception:
        cache = "the Hugging Face cache"
    raise RuntimeError(
        f"model '{model_id}' loaded with {len(meta)} weight tensor(s) that contain no data "
        f"(still on the 'meta' device), e.g. {meta[:3]}. Its files -- or, for a LoRA adapter, "
        f"its base model's files -- are incomplete in {cache}. Delete that model's folder "
        "there and load again to re-download it."
    )


class HFProvider:
    name = "hf"

    def load(self, name: str, registry_path: Path, **kwargs: Any) -> tuple[Any, Any]:
        import torch  # lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
        from ..device_select import resolve_hf_device_map  # lazy

        model_id = kwargs.get("path") or name
        device_map = resolve_hf_device_map(kwargs.get("device_map", "auto"))
        # Checkpoints saved in bf16 (e.g. Qwen2.5-Instruct) produce NaN
        # probabilities during CPU sampling — torch.multinomial then crashes
        # every do_sample=True generation. CPU inference needs fp32. On CUDA,
        # fp16 is a safe default across GPU generations (bf16 needs Ampere+
        # and isn't worth detecting for here); GPU sampling doesn't hit the
        # CPU NaN bug either way.
        if device_map == "cpu":
            dtype = torch.float32
        elif device_map == "cuda":
            dtype = torch.float16
        else:
            dtype = "auto"
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
        )
        _raise_if_unmaterialized(model, model_id)
        return model, tok

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        import torch  # lazy

        max_ctx = (
            getattr(model.config, "n_positions", None)
            or getattr(model.config, "max_position_embeddings", None)
            or 1024
        )
        max_input_len = max(1, max_ctx - max_new_tokens)

        # Instruction-tuned models (Qwen, Llama-Instruct, etc.) were trained on
        # their own specific chat format with special role tokens — feeding
        # them a raw string instead of that format understates what they can
        # actually do. Base models like gpt2 have no chat_template, so this is
        # a no-op for them and behavior is unchanged. When the caller passes
        # structured `messages` (and, for models that support it, `tools`),
        # those are used directly instead of wrapping `prompt` as a single
        # user turn — this is what lets a tool schema reach the model's
        # native tool-calling format (e.g. Qwen's <tool_call> blocks).
        text = prompt
        has_template = getattr(tokenizer, "chat_template", None)
        if messages and has_template:
            text = tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True
            )
        elif has_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )

        # truncate from the left: for RAG-style prompts the actual question
        # lives at the end, so if anything has to go, it's the earliest context
        tokenizer.truncation_side = "left"
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id

        # A fixed max_new_tokens silently truncates answers mid-thought (e.g.
        # mid code-block) whenever a reply genuinely needs more room than
        # asked. Detect that (the model used its whole budget without
        # reaching a natural stop) and keep going, bounded so it can't run
        # away — respects the caller's budget as the normal case, only
        # extends when generation actually got cut off.
        hard_cap = min(max_new_tokens * 4, 1024)
        remaining = hard_cap
        round_budget = max_new_tokens
        all_new_ids: list[int] = []

        while remaining > 0:
            max_input_len = max(1, max_ctx - round_budget)
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_input_len
            )
            # tokenizer output is always CPU tensors; when the model was
            # loaded onto CUDA (device_map="cuda") these must move to match
            # it or model.generate() raises a device-mismatch error.
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            do_sample = temperature > 0
            sampling_kwargs: dict[str, Any] = {}
            if do_sample:
                # With no top_p/top_k, sampling draws from the entire vocab
                # weighted only by temperature -- a small model then reliably
                # picks a rare/wrong subword often enough to produce garbled
                # words (e.g. "Nicodmo" instead of "Nicodemo"). Restricting to
                # the plausible high-probability tokens cuts that down without
                # otherwise changing the sampling temperature behavior.
                sampling_kwargs["top_p"] = top_p
                sampling_kwargs["top_k"] = top_k
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=round_budget,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=pad_id,
                    # Anti-repetition over the GENERATED tokens only -- see
                    # generated_only_penalties() for why not HF's built-in
                    # repetition_penalty / no_repeat_ngram_size.
                    logits_processor=generated_only_penalties(
                        max(0, inputs["input_ids"].shape[1] - len(all_new_ids))
                    ),
                    **sampling_kwargs,
                )
            # decode only the newly generated tokens — robust regardless of
            # whether a chat template altered the input text
            new_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
            all_new_ids.extend(new_ids)
            remaining -= len(new_ids)

            hit_eos = bool(new_ids) and eos_id is not None and new_ids[-1] == eos_id
            used_full_budget = len(new_ids) >= round_budget
            if hit_eos or not used_full_budget or remaining <= 0:
                break

            # cut off, not finished — feed what we generated back in and continue
            text = text + tokenizer.decode(new_ids, skip_special_tokens=True)
            round_budget = min(round_budget, remaining)

        return tokenizer.decode(all_new_ids, skip_special_tokens=True)

    def embed(
        self, model: Any, tokenizer: Any, text: str | list[str]
    ) -> list[list[float]]:
        """Mean-pool the last hidden state as an embedding. Not all HF causal LMs
        are embedders, but it's reliable enough for dashboards and RAG-lite use."""
        import torch  # lazy

        texts = [text] if isinstance(text, str) else text
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last = outputs.hidden_states[-1]  # [B, T, H]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.tolist()

    def info(self, model: Any) -> dict[str, Any]:
        cfg = getattr(model, "config", None)
        return {
            "kind": type(model).__name__,
            "params": sum(p.numel() for p in model.parameters()),
            "hidden_size": getattr(cfg, "hidden_size", None),
            "num_layers": getattr(cfg, "num_hidden_layers", None),
            "vocab_size": getattr(cfg, "vocab_size", None),
        }
