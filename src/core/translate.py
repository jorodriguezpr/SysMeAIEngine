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

"""Spanish<->English translation via dedicated MarianMT models (Helsinki-NLP
opus-mt), not the general chat LLM -- same reasoning already applied to the
RAG embedder/reranker (see providers/sentence_embed.py's docstring): a small
model built specifically for one task consistently beats prompting a
general instruct model to do that task, and these are real, verified models
(BLEU 59.6 es->en, 54.9 en->es on their own published benchmarks).

Loaded and cached the same way rag.py caches its embedder/reranker --
module-level, independent of `state.loaded`, so translating never evicts
whatever chat model the server currently has loaded.
"""
from __future__ import annotations

import re
from typing import Any, Optional

MODEL_IDS = {
    "es-en": "Helsinki-NLP/opus-mt-es-en",
    "en-es": "Helsinki-NLP/opus-mt-en-es",
}

_CACHE: dict[str, tuple[Any, Any]] = {}


def _load(direction: str) -> tuple[Any, Any]:
    if direction not in MODEL_IDS:
        raise ValueError(f"unsupported translation direction '{direction}' (expected 'es-en' or 'en-es')")
    if direction not in _CACHE:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # lazy

        model_id = MODEL_IDS[direction]
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        _CACHE[direction] = (model, tok)
    return _CACHE[direction]


def translate(text: str, direction: str) -> str:
    import torch  # lazy

    model, tok = _load(direction)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=512)
    return tok.decode(out_ids[0], skip_special_tokens=True)


# ── chat-side intent detection ────────────────────────────────────────────
# Deliberately conservative, same philosophy as summarize.py's detector: a
# bare keyword is too weak a signal (translation requests are phrased too
# many different ways to regex reliably), so this only fires when the
# target language is explicit AND the text to translate is structurally
# identifiable (quoted, or after a colon) -- ambiguous phrasings fall
# through to normal chat rather than guessing at what to translate.
_TRANSLATE_KEYWORDS = ("translate", "traducir", "traduce")
_TO_ENGLISH_MARKERS = ("to english", "into english", "al ingl", "en ingl")
_TO_SPANISH_MARKERS = ("to spanish", "into spanish", "al espa", "en espa")
_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')


def detect_translate_request(prompt: str) -> Optional[tuple[str, str]]:
    """Returns (text_to_translate, direction) or None."""
    low = prompt.lower()
    if not any(k in low for k in _TRANSLATE_KEYWORDS):
        return None
    if any(m in low for m in _TO_ENGLISH_MARKERS):
        direction = "es-en"
    elif any(m in low for m in _TO_SPANISH_MARKERS):
        direction = "en-es"
    else:
        return None

    m = _QUOTED_RE.search(prompt)
    if m:
        text = (m.group(1) or m.group(2)).strip()
        return (text, direction) if text else None

    if ":" in prompt:
        text = prompt.split(":", 1)[1].strip()
        return (text, direction) if text else None

    return None
