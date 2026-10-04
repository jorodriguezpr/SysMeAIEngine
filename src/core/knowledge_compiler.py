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

"""Knowledge Compiler -- precise offline answering without generating text
at chat time.

Problem it solves: a small local model *writing* answers from retrieved
passages is the weak link offline (measured on aprocrifos-v2: 0.5B 4%,
1.5B 28% accuracy). Retrieval itself is fine (~0.9 recall). So move the
reading/writing work to BUILD time, where a strong generator can be used
without time pressure, and at chat time only *look up* verified answers.

Build time (``compile_corpus``), once per corpus, resumable:
  * for every indexed chunk, a generator (cloud model, or a local model via
    the engine's providers -- e.g. Ollama qwen2.5:7b for a fully offline
    build) writes several question / short answer / EXACT QUOTE triples in
    the chunk's own language;
  * each triple is VERIFIED against the chunk text: the quote must really
    occur in the chunk (whitespace/accent/case-insensitive, fuzzy >= 0.9)
    and the answer's content words must be supported by quote+chunk.
    Unverifiable triples are dropped -- the knowledge base can't contain
    facts that aren't in the documents;
  * stored at data/compiled/<corpus>/items.json + question embeddings.

Chat time (``answer``), 100% local:
  1. embed the question (local multilingual embedder), take the nearest
     compiled questions, rerank them with the local cross-encoder;
  2. confident match -> return the stored answer + its quote + source,
     instantly, no LLM involved;
  3. otherwise fall back to a local reader over normal retrieval with the
     "quote" prompt, then check the reader's quote actually appears in the
     retrieved context; refuse when it doesn't (``verify="strict"``).
"""
from __future__ import annotations

import difflib
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .state import EngineState

DEFAULT_PER_CHUNK = 8
# rerank score (cross-encoder logit) above which a compiled answer is
# returned without running the reader; calibrated by scripts/eval_grounded.py
# runs -- see docs/GROUNDING.md "Knowledge compiler".
DEFAULT_MATCH_THRESHOLD = 3.0

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.S)
_WORD_RE = re.compile(r"[^\W\d_]{4,}")


def _dir(state: EngineState, corpus: str) -> Path:
    return state.config.data_dir / "compiled" / corpus


# ── text normalization / verification ──────────────────────────────────────

def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[«»“”\"'`´]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def quote_in_text(quote: str, text: str, min_ratio: float = 0.9) -> bool:
    """Is `quote` (near-)verbatim in `text`? Exact normalized substring, else
    best fuzzy match over same-length windows (tolerates OCR/typo noise and
    small copying slips, not paraphrase)."""
    q, t = _norm(quote), _norm(text)
    if len(q) < 12:
        return False
    if q in t:
        return True
    n = len(q)
    step = max(1, n // 8)
    sm = difflib.SequenceMatcher(None, q, "", autojunk=False)
    best = 0.0
    for i in range(0, max(1, len(t) - n + 1), step):
        sm.set_seq2(t[i:i + n])
        if sm.quick_ratio() < min_ratio:
            continue
        best = max(best, sm.ratio())
        if best >= min_ratio:
            return True
    return False


def answer_supported(answer: str, evidence: str, min_frac: float = 0.6) -> bool:
    """Share of the answer's content words (4+ letters) found in the
    evidence. Catches answers that add facts the quote doesn't contain."""
    words = {_norm(w) for w in _WORD_RE.findall(answer)}
    if not words:
        return False
    ev = _norm(evidence)
    return sum(1 for w in words if w in ev) / len(words) >= min_frac


# ── build time ──────────────────────────────────────────────────────────────

def _compile_prompt(passage: str, n: int) -> str:
    return (
        "You are compiling a question-answer knowledge base from a document.\n"
        f"From the passage below, write {n} different questions a reader could ask "
        "that this passage answers -- cover different facts (who, what, where, when, "
        "why, how), and phrase them the way a person would ask, NOT by copying the "
        "passage's wording. For each question give:\n"
        '  "answer": a short, precise answer (one sentence),\n'
        '  "quote": the exact sentence or phrase from the passage that proves the '
        "answer, copied WORD FOR WORD.\n"
        "Write everything in the same language as the passage. Only facts stated "
        "in the passage.\n"
        'Respond ONLY with a JSON array: [{"question": "...", "answer": "...", "quote": "..."}, ...]\n\n'
        f"Passage:\n{passage}"
    )


def _generate(state: EngineState, generator: str, prompt: str) -> str:
    if generator == "cloud":
        from .cloud_chat import cloud_generate
        from .cloud_settings import get_cloud_settings

        return cloud_generate(get_cloud_settings(state), prompt)
    if generator.startswith("local:"):
        from .model_loader import chat as core_chat

        return core_chat(state, prompt, model_name=generator.split(":", 1)[1],
                         max_new_tokens=1500, temperature=0.2)["text"]
    raise ValueError("generator must be 'cloud' or 'local:<registered model>'")


def _parse_items(raw: str) -> list[dict[str, str]]:
    m = _JSON_ARRAY_RE.search(raw or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for d in data if isinstance(data, list) else []:
        if isinstance(d, dict) and all(isinstance(d.get(k), str) and d[k].strip() for k in ("question", "answer", "quote")):
            out.append({k: d[k].strip() for k in ("question", "answer", "quote")})
    return out


def compile_corpus(
    state: EngineState, corpus: str, generator: str = "cloud",
    per_chunk: int = DEFAULT_PER_CHUNK, exclude_chunk_ids: Optional[set[str]] = None,
    workers: int = 3, out_name: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Build (or resume) the compiled knowledge base for `corpus`.
    exclude_chunk_ids: chunks to leave out (used to test generalization)."""
    from .rag import _embed_texts, _paths

    chunks = json.loads(_paths(state, "semantic", corpus)["meta"].read_text(encoding="utf-8"))
    exclude = exclude_chunk_ids or set()
    out_dir = state.config.data_dir / "compiled" / (out_name or corpus)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.json"
    items: list[dict[str, Any]] = json.loads(items_path.read_text(encoding="utf-8")) if items_path.exists() else []
    done_chunks = {i["chunk_id"] for i in items} | set(json.loads((out_dir / "done_chunks.json").read_text()) if (out_dir / "done_chunks.json").exists() else [])
    todo = [c for c in chunks if c["id"] not in exclude and c["id"] not in done_chunks and len(c["text"]) >= 150]
    log(f"compile {corpus}: {len(todo)} chunks to process ({len(done_chunks)} already done, {len(exclude)} excluded)")

    stats = {"generated": 0, "kept": 0, "bad_quote": 0, "unsupported": 0, "failed_chunks": 0}

    def work(c: dict) -> tuple[str, list[dict], dict]:
        st = {"generated": 0, "kept": 0, "bad_quote": 0, "unsupported": 0}
        try:
            cand = _parse_items(_generate(state, generator, _compile_prompt(c["text"], per_chunk)))
        except Exception:
            return c["id"], [], {"failed": 1}
        kept = []
        for it in cand:
            st["generated"] += 1
            if not quote_in_text(it["quote"], c["text"]):
                st["bad_quote"] += 1
                continue
            if not answer_supported(it["answer"], it["quote"] + " " + c["text"]):
                st["unsupported"] += 1
                continue
            kept.append({**it, "chunk_id": c["id"], "source": c.get("source", ""), "corpus": corpus})
            st["kept"] += 1
        return c["id"], kept, st

    done_list = sorted(done_chunks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, (cid, kept, st) in enumerate(ex.map(work, todo), 1):
            if st.get("failed"):
                stats["failed_chunks"] += 1
                continue
            items.extend(kept)
            done_list.append(cid)
            for k in ("generated", "kept", "bad_quote", "unsupported"):
                stats[k] += st[k]
            if n % 10 == 0 or n == len(todo):
                items_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
                (out_dir / "done_chunks.json").write_text(json.dumps(done_list))
                log(f"  {n}/{len(todo)} chunks, {len(items)} verified items")

    items_path.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "done_chunks.json").write_text(json.dumps(done_list))
    if items:
        vecs = _embed_texts([i["question"] for i in items])
        np.save(out_dir / "q_emb.npy", vecs.astype(np.float32))
    meta = {"corpus": corpus, "generator": generator, "per_chunk": per_chunk,
            "items": len(items), "excluded_chunks": len(exclude), "built": time.strftime("%Y-%m-%dT%H:%M:%S"), **stats}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    _CACHE.pop(str(out_dir), None)
    log(f"compiled: {json.dumps(meta)}")
    return meta


# ── chat time ───────────────────────────────────────────────────────────────

_CACHE: dict[str, tuple[list[dict], np.ndarray]] = {}


def _load(state: EngineState, kb: str) -> Optional[tuple[list[dict], np.ndarray]]:
    d = state.config.data_dir / "compiled" / kb
    key = str(d)
    if key not in _CACHE:
        if not (d / "items.json").exists() or not (d / "q_emb.npy").exists():
            return None
        _CACHE[key] = (json.loads((d / "items.json").read_text(encoding="utf-8")), np.load(d / "q_emb.npy"))
    return _CACHE[key]


def lookup(state: EngineState, kb: str, question: str, top_n: int = 8) -> list[dict[str, Any]]:
    """Nearest compiled items for `question`, reranked by the local
    cross-encoder on (question, stored question + answer)."""
    from .rag import _embed_texts, _reranker

    loaded = _load(state, kb)
    if not loaded:
        return []
    items, emb = loaded
    q = _embed_texts([question])[0]
    sims = emb @ q
    idx = np.argsort(-sims)[:top_n]
    cands = [{**items[i], "cosine": float(sims[i])} for i in idx]
    scores = _reranker().predict([(question, c["question"] + " " + c["answer"]) for c in cands])
    for c, s in zip(cands, scores):
        c["score"] = float(s)
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands


def _fact_hit(c: dict[str, Any]) -> dict[str, Any]:
    """A compiled item rendered as a context row for rag.build_prompt."""
    return {"corpus": c.get("corpus", ""), "source": c.get("source", ""), "id": c.get("chunk_id"),
            "score": c.get("score"), "text": f"{c['question']} {c['answer']} («{c['quote']}»)"}


def _answer_from_facts(
    state: EngineState, question: str, corpus: str, reader_model: Optional[str],
    cands: list[dict[str, Any]], strategy: str, direct_threshold: float, n_facts: int,
    top_k: int, verify: str, max_new_tokens: int, t0: float,
) -> dict[str, Any]:
    from . import rag
    from .model_loader import chat as core_chat

    if cands and cands[0]["score"] >= direct_threshold:
        best = cands[0]
        return {"mode": "compiled", "text": best["answer"], "quote": best["quote"],
                "source": best["source"], "chunk_id": best["chunk_id"], "score": best["score"],
                "matched_question": best["question"], "elapsed_s": round(time.perf_counter() - t0, 2)}
    hits = [_fact_hit(c) for c in cands[:n_facts]]
    if strategy == "facts+raw":
        raw_hits, _ = rag.retrieve_grounded(state, question, top_k=top_k, corpus=corpus, multi_query=False,
                                            parent_child=False, kgraph_enabled=False, mmr=True, allow_cloud=False)
        hits += raw_hits
    lang = rag.question_language(question, hits)
    refusal = rag.REFUSALS.get(lang, rag.REFUSALS["English"])
    base = {"score": cands[0]["score"] if cands else None, "sources": sorted({h.get("source") for h in hits})}
    if not hits or not reader_model:
        return {**base, "mode": "refusal", "text": refusal, "elapsed_s": round(time.perf_counter() - t0, 2)}
    raw = rag.local_answer(state, rag.build_prompt(question, hits, "native"), reader_model, lang, max_new_tokens)
    evidence = "\n".join(h["text"] for h in hits)
    verified = answer_supported(raw, evidence, min_frac=0.5)
    out = {**base, "mode": "reader", "text": raw, "verified": verified,
           "elapsed_s": round(time.perf_counter() - t0, 2)}
    if rag.is_refusal(raw):
        return {**out, "mode": "refusal", "text": refusal}
    if verify == "strict" and not verified:
        return {**out, "mode": "refusal", "text": refusal, "unverified_answer": raw}
    return out


_QUOTE_RE = re.compile(r"(?is)(?:cita|quote)\s*:\s*(.+?)\s*(?:\n\s*(?:respuesta|answer)\s*:\s*(.+))?$")


def parse_quoted_answer(text: str) -> tuple[Optional[str], str]:
    """Split a "quote" style reply into (quote, answer). The prompt ends
    with "Cita:"/"Quote:", so the reply usually starts with the quote."""
    t = text.strip()
    m = re.search(r"(?is)^(.*?)\n\s*(?:respuesta|answer)\s*:\s*(.+)$", t)
    if m:
        quote = re.sub(r"(?i)^\s*(cita|quote)\s*:\s*", "", m.group(1)).strip().strip("«»\"“” ")
        return (quote or None), m.group(2).strip()
    return None, t


def answer(
    state: EngineState, question: str, corpus: str, reader_model: Optional[str],
    kb: Optional[str] = None, threshold: float = DEFAULT_MATCH_THRESHOLD,
    top_k: int = 3, verify: str = "strict", max_new_tokens: int = 250,
    reader_style: str = "native", strategy: str = "lookup",
    direct_threshold: float = 8.0, n_facts: int = 8,
) -> dict[str, Any]:
    """Local precise answer. verify: "strict" (refuse when the reader's
    answer can't be traced to the retrieved context) | "lenient" (answer
    anyway, flagged) | "off".

    reader_style: "native" (default) or "quote". Measured on aprocrifos-v2
    with local qwen2.5-3b: the quote-first prompt made it wrongly refuse 52%
    of answerable questions (it can't hold the format), native only 12% --
    so native is the default; its answers are verified by content-word
    support against the context instead of by an exact quote.

    strategy:
      lookup    -- return the stored answer when the best compiled match
                   scores >= threshold, else the reader over raw passages.
                   Measured weak: paraphrased questions often match a
                   different fact of the same (or a wrong) passage.
      facts     -- the reader answers from the top `n_facts` compiled facts
                   (question + answer + exact quote) instead of raw
                   passages; only a near-duplicate match (>= direct_threshold)
                   is answered directly without the reader.
      facts+raw -- facts plus the top raw passages (covers questions the
                   compiled facts don't).
    """
    from . import rag
    from .model_loader import chat as core_chat

    t0 = time.perf_counter()
    cands = lookup(state, kb or corpus, question)
    if strategy in ("facts", "facts+raw"):
        return _answer_from_facts(state, question, corpus, reader_model, cands, strategy,
                                  direct_threshold, n_facts, top_k, verify, max_new_tokens, t0)
    if cands and cands[0]["score"] >= threshold:
        best = cands[0]
        return {"mode": "compiled", "text": best["answer"], "quote": best["quote"],
                "source": best["source"], "chunk_id": best["chunk_id"], "score": best["score"],
                "matched_question": best["question"], "elapsed_s": round(time.perf_counter() - t0, 2)}

    hits, _ = rag.retrieve_grounded(state, question, top_k=top_k, corpus=corpus, multi_query=False,
                                    parent_child=False, kgraph_enabled=False, mmr=True, allow_cloud=False)
    lang = rag.question_language(question, hits)
    refusal = rag.REFUSALS.get(lang, rag.REFUSALS["English"])
    base = {"score": cands[0]["score"] if cands else None, "elapsed_s": None}
    if not hits or not reader_model:
        return {**base, "mode": "refusal", "text": refusal, "elapsed_s": round(time.perf_counter() - t0, 2)}
    raw = rag.local_answer(state, rag.build_prompt(question, hits, reader_style), reader_model, lang, max_new_tokens)
    context = "\n".join(h["text"] for h in hits)
    if reader_style == "quote":
        quote, ans = parse_quoted_answer(raw)
        verified = bool(quote) and quote_in_text(quote, context, min_ratio=0.85)
    else:
        quote, ans = None, raw.strip()
        verified = answer_supported(ans, context, min_frac=0.5)
    out = {**base, "mode": "reader", "text": ans, "quote": quote, "verified": verified,
           "sources": [h.get("source") for h in hits], "elapsed_s": round(time.perf_counter() - t0, 2)}
    if rag.is_refusal(ans) or rag.is_refusal(raw):
        return {**out, "mode": "refusal", "text": refusal}
    if verify == "strict" and not verified:
        return {**out, "mode": "refusal", "text": refusal, "unverified_answer": ans}
    return out
