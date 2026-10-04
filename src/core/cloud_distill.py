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

"""Higher-quality QA training data via the cloud AI provider, instead of
ingestion/pipeline.py's build_qa_dataset() heuristic (first sentence of a
chunk as the question, the rest as the answer -- cheap, but frequently
produces a nonsensical question or an answer that doesn't actually answer
it). This asks a real model to read each chunk and write several genuine,
grounded, *differently-phrased* question/answer pairs about it, in the same
language as the source text (see summarize.py's _detect_language -- same
reasoning, reused here rather than duplicated).

Multiple differently-worded pairs per chunk, not just one: live-tested with
one pair per chunk first, and a model fine-tuned on that dataset turned out
to be badly overfit to the *exact phrasing* of its training questions --
excellent on inputs resembling the training set, but producing outright
garbled/degenerate output on a real user's differently-worded phrasing of
the very same question. Generating several phrasings per fact is the direct
fix: the same real API call now returns 2-3 pairs instead of 1, so this
doesn't cost extra API calls, just extracts more training signal from each
one.

Opt-in and costs one real API call per chunk -- this project's existing
`--max-rows` sampling cap (applied by the caller before this runs) is what
keeps that bounded, not anything in here.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

from .cloud_chat import cloud_generate
from .cloud_settings import get_cloud_settings
from .state import EngineState
from .summarize import _detect_language

DEFAULT_PAIRS_PER_CHUNK = 3
# Asking for several full QA pairs per call needs real room -- live-tested
# at the chat-settings default (1024) and it wasn't enough: responses got
# cut off mid-answer on a real, reproducible fraction of chunks, silently
# writing truncated half-sentences into the training set (confirmed by
# inspecting the actual generated dataset, not assumed). This is
# independent of whatever max_tokens the user has configured for
# interactive chat -- that's a different use case with different needs.
QA_GENERATION_MAX_TOKENS = 2048

# non-greedy answer capture stops at the next "Question:" marker or the end
# of the response, so this correctly splits multiple pairs out of one reply
# instead of only ever matching the first "Question:...Answer:..." span.
_QA_PAIR_RE = re.compile(
    r"question\s*:\s*(.+?)\s*answer\s*:\s*(.+?)(?=\s*question\s*:|\s*\Z)",
    re.IGNORECASE | re.DOTALL,
)
_SENTENCE_END_CHARS = ".!?\"'»”)"


def _build_qa_prompt(chunk_text: str, language: str, n_pairs: int = DEFAULT_PAIRS_PER_CHUNK) -> str:
    return (
        f"Read the following passage and write {n_pairs} DIFFERENT questions "
        "a reader might genuinely ask about it. Vary the phrasing, wording, "
        "tone, and structure between them (e.g. one formal, one casual, one "
        "phrased as a short direct question) so they don't all read like "
        "templates of each other -- a real person would never ask about the "
        "same fact the same way twice. For each question, write a complete, "
        "accurate answer using only information found in the passage -- do "
        f"not add anything the passage doesn't say. Write everything in "
        f"{language}. Respond in exactly this format and nothing else:\n"
        "Question: <question 1>\nAnswer: <answer 1>\n"
        "Question: <question 2>\nAnswer: <answer 2>\n"
        "Question: <question 3>\nAnswer: <answer 3>\n\n"
        f"Passage:\n{chunk_text}"
    )


def _looks_truncated(answer: str) -> bool:
    stripped = answer.rstrip()
    return not stripped or stripped[-1] not in _SENTENCE_END_CHARS


def _parse_qa_pairs(text: str) -> list[tuple[str, str]]:
    pairs = []
    for m in _QA_PAIR_RE.finditer(text):
        q, a = m.group(1).strip(), m.group(2).strip()
        # a response cut off by the token budget mid-way through a *later*
        # pair leaves a dangling, incomplete "Question" label stuck onto the
        # end of the previous answer instead of a clean boundary (confirmed
        # in a real generated dataset) -- strip that artifact specifically.
        a = re.sub(r"\n+\s*question\s*$", "", a, flags=re.IGNORECASE).strip()
        if q and a:
            pairs.append((q, a))
    # Only the *last* pair in a response can have been cut off mid-answer by
    # the token budget -- every earlier pair was already followed by a real
    # "Question:" marker, proving it completed. Drop the last one if it
    # doesn't end like a finished sentence rather than risk training on a
    # truncated answer (confirmed live: raw answers like "...ella res" and
    # "...descienden y luch" made it into a real dataset before this check).
    if pairs and _looks_truncated(pairs[-1][1]):
        pairs = pairs[:-1]
    return pairs


def ensure_cloud_ready(state: EngineState) -> dict[str, Any]:
    """Fail fast, before any extraction/chunking work happens, if cloud
    settings aren't actually usable -- raises ValueError with a message
    suitable for surfacing directly as an HTTP 400."""
    settings = get_cloud_settings(state)
    if not settings.get("enabled"):
        raise ValueError("cloud AI provider is not enabled -- configure it on the Settings page first")
    if not settings.get("api_key"):
        raise ValueError(f"no API key configured for cloud provider '{settings.get('provider')}'")
    if not settings.get("model"):
        raise ValueError("no model configured for the cloud provider")
    return settings


def build_cloud_qa_dataset(
    state: EngineState,
    chunks: list[dict[str, Any]],
    on_progress: Optional[Callable[[int, int, int], None]] = None,
    n_pairs: int = DEFAULT_PAIRS_PER_CHUNK,
) -> list[dict[str, Any]]:
    """One cloud call per chunk, each call producing up to `n_pairs`
    differently-phrased QA rows (see the module docstring for why phrasing
    diversity matters, not just answer quality). Best-effort: a chunk that
    fails (network error, unparseable response) is skipped, not fatal --
    with potentially hundreds of real network calls, some transient
    failures are expected, and losing the whole job over one bad chunk would
    be worse than a slightly smaller dataset. `on_progress(done, total,
    failed)` is called after every chunk, mainly so a long-running job's log
    doesn't go silent for however long hundreds of sequential API calls
    take."""
    if not chunks:
        return []
    settings = ensure_cloud_ready(state)

    sample_text = " ".join(c["text"] for c in chunks[:3])[:3000]
    language = _detect_language(sample_text)

    rows: list[dict[str, Any]] = []
    failed = 0
    for i, c in enumerate(chunks, 1):
        try:
            raw = cloud_generate(
                settings, _build_qa_prompt(c["text"], language, n_pairs),
                max_tokens=QA_GENERATION_MAX_TOKENS,
            )
            pairs = _parse_qa_pairs(raw)
            if pairs:
                for j, (q, a) in enumerate(pairs):
                    rows.append({"question": q, "answer": a, "id": f"{c['id']}-{j}"})
            else:
                failed += 1
        except (ValueError, RuntimeError):
            failed += 1
        if on_progress:
            on_progress(i, len(chunks), failed)
    return rows
