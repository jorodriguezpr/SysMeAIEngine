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

"""RAM pre-flight estimate for a fine-tuning job.

Separate from src/core/utils/device_probe.py (that module's docstring
scopes it to Windows hardware probing, not training math).

The estimate is deliberately a rough order-of-magnitude, not a precise
prediction -- there's no dynamic way to know a model's real parameter count
without either a network call to fetch its config (fragile: fails offline,
adds latency to every job-launch request) or having already downloaded it,
so this uses a small static lookup for the models this project's UI
actually offers, plus a conservative fallback for anything else (including
arbitrary model ids passed directly via the API, which the UI dropdown
doesn't expose but the endpoint doesn't prevent).
"""
from __future__ import annotations

from typing import Any

# Known base models (UI dropdown + commonly used via the API), param counts
# in millions. Source: each model's published config (hidden_size etc.) or
# its name -- e.g. "0.5B" -- not independently verified against a loaded
# checkpoint for every entry.
_KNOWN_MODEL_PARAMS_M: dict[str, float] = {
    "gpt2": 124,
    "qwen/qwen2.5-0.5b-instruct": 494,
    "qwen/qwen2.5-0.5b": 494,
    "qwen/qwen2.5-1.5b-instruct": 1540,
    "qwen/qwen3-0.6b": 600,
    "minilm": 2,
    "tinygpt": 2,
}
_FALLBACK_PARAMS_M = 500  # unrecognized model id -- assume "small-ish" HF model


def estimate_training_ram_gb(
    base_model: str, batch_size: int = 16, max_seq: int = 128, method: str = "sft",
) -> dict[str, Any]:
    """CPU fp32. Two different formulas depending on `method`, because they
    have genuinely different memory shapes -- this used to assume full
    fine-tune math unconditionally (see the project's own note that this
    project defaults to SFT, not LoRA), which meant a LoRA job on a bigger
    model than this table knew about could get hard-blocked over an
    estimate 3-4x too pessimistic for what LoRA actually needs. Verified
    directly: qwen2.5-1.5b-instruct loaded and ran real inference at ~6GB,
    consistent with the LoRA-shaped estimate below, not the full-fine-tune one.

    'sft' (full fine-tune): params * 4 bytes (weights) + params * 4 bytes
    (gradients) + params * 8 bytes (Adam's two fp32 moment buffers) =
    params * 16 bytes -- every weight needs a gradient and optimizer state.

    'lora'/'qlora': the base model is frozen -- only ~0.1% of params (the
    adapter) need gradients/optimizer state, negligible next to the base
    model itself. Dominant cost is just holding the frozen fp32 weights:
    params * 4 bytes, plus the same activation-memory term either way.

    Both are rough order-of-magnitude estimates for a pre-flight check, not
    a guarantee -- actual usage varies with architecture, tokenizer, and
    whatever else is resident in memory at the time.
    """
    key = base_model.strip().lower()
    params_m = _KNOWN_MODEL_PARAMS_M.get(key, _FALLBACK_PARAMS_M)
    params = params_m * 1_000_000

    bytes_per_param = 4 if method in ("lora", "qlora") else 16
    weights_grads_optim_gb = (params * bytes_per_param) / 1e9
    # Coarse activation-memory term: proportional to batch*seq*hidden-ish;
    # without loading the model we don't know hidden size, so approximate
    # hidden_size from param count (very rough) and keep this term small
    # relative to the dominant weights/gradients/optimizer term above.
    approx_hidden = max(64, int((params / 12) ** 0.5))
    activation_gb = (batch_size * max_seq * approx_hidden * 4 * 4) / 1e9

    estimated_gb = round(weights_grads_optim_gb + activation_gb, 1)
    return {
        "base_model": base_model,
        "method": method,
        "matched_known_model": key in _KNOWN_MODEL_PARAMS_M,
        "assumed_params_m": params_m,
        "estimated_gb": estimated_gb,
        "note": f"rough order-of-magnitude estimate for {method} fp32 training, not a guarantee",
    }


def check_ram_preflight(
    base_model: str, batch_size: int, max_seq: int, total_ram_gb: float, available_ram_gb: float,
    method: str = "sft",
) -> dict[str, Any]:
    """Returns {"estimate": {...}, "block": bool, "warning": str|None}.

    block=True only when the estimate exceeds *total* physical RAM (a
    guaranteed-OOM job should never even start). A softer warning (job
    still allowed to launch) fires past ~70% of *available* RAM -- avoids
    false-positive blocks on a moderately loaded machine.
    """
    estimate = estimate_training_ram_gb(base_model, batch_size, max_seq, method)
    gb = estimate["estimated_gb"]
    if gb > total_ram_gb:
        return {
            "estimate": estimate,
            "block": True,
            "warning": (
                f"Estimated RAM need (~{gb} GB) exceeds this machine's total RAM "
                f"({total_ram_gb} GB) -- this job would almost certainly fail with an "
                f"out-of-memory error. Try a smaller model, smaller --batch-size/--max-seq, "
                f"or LoRA instead of full fine-tuning."
            ),
        }
    if gb > 0.7 * available_ram_gb:
        return {
            "estimate": estimate,
            "block": False,
            "warning": (
                f"Estimated RAM need (~{gb} GB) is close to what's currently available "
                f"({available_ram_gb} GB) -- this may run slowly or risk OOM if other "
                f"programs are using memory at the same time."
            ),
        }
    return {"estimate": estimate, "block": False, "warning": None}
