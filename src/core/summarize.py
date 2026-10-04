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

"""Whole-document summarization via map-reduce.

A real ebook is far beyond any of this project's models' usable context
window (and HFProvider truncates from the left when a prompt is too long --
see hf.py -- so naively stuffing a whole book into one prompt would silently
summarize only its last few pages). This instead: extracts + chunks the
document (map) summarizes each chunk individually, then combines those
chunk-summaries into one final summary, recursively re-summarizing batches
of summaries if there are too many to fit in a single prompt (reduce).

Prompts ask the model to reply in the same language as the source text
rather than hardcoding English or Spanish, since this project's documents
are routinely one or the other (see providers/sentence_embed.py's docstring
on why the RAG embedder/reranker are multilingual for the same reason).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from ..ingestion.pipeline import chunk_text_paragraphs, clean_text, extract_any
from .model_loader import chat as core_chat
from .state import EngineState

# Deliberately compound: a keyword alone (especially "resum", which also
# matches the unrelated English verb "resume") is too weak a signal on its
# own -- requiring an explicit, extractable file path alongside it keeps
# false positives (e.g. "resume the download") from ever reaching here,
# since those don't also contain a path ending in a supported extension.
_SUMMARIZE_KEYWORDS = ("summarize", "summarise", "summary", "resum")
_PATH_RE = re.compile(
    r'["\']?((?:[A-Za-z]:)?[\\/][^"\'\n]+?\.(?:pdf|txt|docx|pptx|xlsx))["\']?',
    re.IGNORECASE,
)


def detect_summarize_request(prompt: str) -> Optional[str]:
    """Best-effort chat-side router: if the message looks like "summarize
    <path>" (English or Spanish), return the path so the caller can run the
    real map-reduce pipeline instead of a normal generation call -- the
    small models this project uses aren't reliable enough at native
    tool-calling to trust that judgment call to the model itself (see this
    project's own findings on small-model reliability), so this is a plain
    deterministic check, the same pattern try_calculate() already uses for
    arithmetic in this codebase."""
    low = prompt.lower()
    if not any(k in low for k in _SUMMARIZE_KEYWORDS):
        return None
    m = _PATH_RE.search(prompt)
    return m.group(1) if m else None

CHUNK_SUMMARY_MAX_NEW_TOKENS = 150
FINAL_SUMMARY_MAX_NEW_TOKENS = 350
# characters, not tokens (this project's own chars/4 approximation elsewhere
# would put this around ~1500 tokens) -- keeps each reduce-round prompt fast
# on CPU rather than maximizing how much fits in the model's real context.
REDUCE_BATCH_CHAR_LIMIT = 6000
MAX_REDUCE_ROUNDS = 5

# A vague "reply in the same language as the text" instruction was
# live-tested against a real Spanish document and the model answered in
# English anyway -- small instruct models don't reliably infer source
# language from a long pasted block of text well enough to act on it. A
# cheap marker-count heuristic (English/Spanish only -- the two languages
# this project's documents actually come in) lets the prompt name the
# target language explicitly instead of asking the model to figure it out.
_SPANISH_MARKERS = (
    " de ", " que ", " el ", " la ", " en ", " los ", " las ", " del ", " un ",
    " una ", " para ", " con ", " por ", " se ", " su ", " es ", " no ",
)
_ENGLISH_MARKERS = (
    " the ", " and ", " of ", " to ", " in ", " that ", " with ", " for ",
    " is ", " was ", " it ", " as ", " on ",
)


def _detect_language(text: str) -> str:
    sample = f" {text[:3000].lower()} "
    spanish_score = sample.count("ñ") + sample.count("¿") + sample.count("¡")
    spanish_score += sum(sample.count(m) for m in _SPANISH_MARKERS)
    english_score = sum(sample.count(m) for m in _ENGLISH_MARKERS)
    return "Spanish" if spanish_score > english_score else "English"


def _summarize_text(
    state: EngineState, text: str, model_name: Optional[str], max_new_tokens: int, language: str
) -> str:
    prompt = (
        f"Summarize the following text in a few clear sentences, in {language}. "
        "Do not add information that isn't in the text.\n\nText:\n" + text + "\n\nSummary:"
    )
    out = core_chat(state, prompt, model_name=model_name, max_new_tokens=max_new_tokens, temperature=0.2)
    return out["text"].strip()


def _batch_by_chars(items: list[str], limit: int) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for item in items:
        if current and current_len + len(item) > limit:
            batches.append(current)
            current, current_len = [], 0
        current.append(item)
        current_len += len(item)
    if current:
        batches.append(current)
    return batches


def summarize_document(
    state: EngineState,
    source: str | Path,
    model_name: Optional[str] = None,
    chunk_size: int = 1200,
) -> dict[str, Any]:
    """Extract + chunk `source`, summarize each chunk, then reduce those
    chunk-summaries into one final summary. Returns chunk_count/reduce_rounds
    alongside the summary so a caller can tell a one-paragraph memo from a
    genuinely multi-round reduction over a long book."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"document not found: {path}")

    raw = clean_text(extract_any(path))
    chunks = chunk_text_paragraphs(raw, chunk_size=chunk_size)
    if not chunks:
        raise ValueError(f"no extractable text in {path}")

    # detected once from the whole document rather than per-chunk: cheaper,
    # and keeps every chunk/reduce summary in the same language even if one
    # short chunk's own marker counts would have tipped the other way.
    language = _detect_language(raw)

    summaries = [
        _summarize_text(state, c["text"], model_name, CHUNK_SUMMARY_MAX_NEW_TOKENS, language)
        for c in chunks
    ]

    reduce_rounds = 0
    while (
        len(summaries) > 1
        and sum(len(s) for s in summaries) > REDUCE_BATCH_CHAR_LIMIT
        and reduce_rounds < MAX_REDUCE_ROUNDS
    ):
        batches = _batch_by_chars(summaries, REDUCE_BATCH_CHAR_LIMIT)
        summaries = [
            _summarize_text(state, "\n\n".join(b), model_name, CHUNK_SUMMARY_MAX_NEW_TOKENS, language)
            for b in batches
        ]
        reduce_rounds += 1

    if len(summaries) == 1:
        final_summary = summaries[0]
    else:
        final_summary = _summarize_text(
            state, "\n\n".join(summaries), model_name, FINAL_SUMMARY_MAX_NEW_TOKENS, language
        )

    return {
        "source": str(path),
        "model": model_name or state.config.default_model,
        "chunk_count": len(chunks),
        "reduce_rounds": reduce_rounds,
        "language": language,
        "summary": final_summary,
    }
