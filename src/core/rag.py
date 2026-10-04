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

"""Retrieval-augmented generation over ingested documents.

Three retrieval methods, all indexing the same chunked source text (kept
separate from the fine-tuning dataset so retrieval always reflects the raw
book text, not the duplicated/overlapping rows used for training):

  * "tfidf"    — lexical keyword matching (scikit-learn). Fast, no model
                 download, but only finds passages sharing vocabulary with
                 the query. Stopword list is combined English+Spanish (see
                 _STOP_WORDS) since this project's corpora are routinely
                 Spanish-language.
  * "semantic" — dense embeddings from a real sentence-embedding model
                 (a multilingual MiniLM via the `sentence_embed` provider —
                 see that module's own docstring for why it's multilingual,
                 not the more common English-only all-MiniLM-L6-v2).
                 Finds conceptually related passages even with no shared
                 words.
  * "hybrid"   — the default. Runs both of the above, fuses the two ranked
                 lists with Reciprocal Rank Fusion (robust because it works
                 on rank position, not raw scores, which aren't comparable
                 across a lexical and a dense retriever), then reranks the
                 fused candidates with a cross-encoder (a multilingual
                 mMARCO model, same Spanish-support reasoning as the
                 embedder above) that scores the query and passage together
                 — more accurate than scoring them independently and
                 comparing vectors, since it can model interaction between
                 the two. Falls back to semantic-only + rerank if a corpus
                 has no TF-IDF index.

The semantic embedder and reranker are each loaded once into their own
module-level cache, independent of `state.loaded` — building/querying an
index must not evict whatever chat model the live server currently has
loaded.
"""
from __future__ import annotations

import json
import re
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ..ingestion.pipeline import chunk_text_paragraphs
from .providers.sentence_embed import DEFAULT_MODEL_ID, SentenceEmbedProvider
from .state import EngineState

DEFAULT_METHOD = "hybrid"
# mMARCO is MS MARCO machine-translated into 14 languages -- a plain
# ms-marco-MiniLM reranker was trained English-only, so scoring Spanish
# query/passage pairs with it is out-of-distribution for the model, not
# just lower-accuracy garbage-in-garbage-out.
DEFAULT_RERANKER_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# sklearn's TfidfVectorizer ships a built-in English stopword list but no
# Spanish one -- passing stop_words="english" on Spanish text leaves every
# Spanish function word (de, la, que, en, ...) in the vocabulary, where
# their sheer frequency drowns out genuinely distinctive terms in the TF-IDF
# weighting. This is a solid, practical stopword set, not the exhaustive
# canonical list (verb conjugations etc. are diminishing returns for this).
_SPANISH_STOP_WORDS = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero",
    "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes",
    "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto",
    "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco",
    "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis",
    "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros",
    "vosotras", "os", "esos", "esas", "soy", "eres", "es", "somos", "sois",
    "son", "sea", "sean", "era", "eran", "fui", "fue", "fueron", "siendo",
    "sido", "tengo", "tiene", "tienen", "tenía", "tuvo", "tuvieron",
    "teniendo", "tenido", "cada", "cómo", "así", "aunque", "mientras",
    "aquí", "allí", "ahí", "dentro", "fuera", "ademas", "además", "vez",
    "haber", "hacer", "hace", "hacia", "sólo", "solo", "según", "sino",
]
_STOP_WORDS: list[str] | None = None


def _stop_words() -> list[str]:
    global _STOP_WORDS
    if _STOP_WORDS is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        _STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | set(_SPANISH_STOP_WORDS))
    return _STOP_WORDS

_CACHE: dict[str, tuple[Any, ...]] = {}
_EMBEDDER_CACHE: dict[str, Any] = {}
_RERANKER_CACHE: dict[str, Any] = {}


def _reranker(model_id: str = DEFAULT_RERANKER_ID):
    if model_id not in _RERANKER_CACHE:
        from sentence_transformers import CrossEncoder

        _RERANKER_CACHE[model_id] = CrossEncoder(model_id)
    return _RERANKER_CACHE[model_id]


def _embedder(model_id: str = DEFAULT_MODEL_ID):
    if model_id not in _EMBEDDER_CACHE:
        provider = SentenceEmbedProvider()
        model, _ = provider.load(model_id, Path("."), path=model_id)
        _EMBEDDER_CACHE[model_id] = (provider, model)
    return _EMBEDDER_CACHE[model_id]


def _embed_texts(texts: list[str], model_id: str = DEFAULT_MODEL_ID, batch_size: int = 64) -> np.ndarray:
    provider, model = _embedder(model_id)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        out.extend(provider.embed(model, None, texts[i : i + batch_size]))
    return np.array(out, dtype=np.float32)


def _chunk_source_dir(source_dir: Path) -> list[dict[str, Any]]:
    """Paragraph-aware, not the plain fixed-size sliding window -- this used
    to always cut chunks at a raw character count regardless of sentence/
    paragraph boundaries, which meant retrieval could easily hand a
    mid-sentence fragment to the model as "context" (confirmed by
    inspecting a real chunk that ended "...pero las s"). Packing whole
    paragraphs avoids that, without changing anything about how the
    training dataset itself gets chunked (train_from_folder.py's own
    --chunking flag is separate and unaffected)."""
    docs: list[dict[str, Any]] = []
    for txt_path in sorted(source_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        for c in chunk_text_paragraphs(text):
            docs.append({"id": c["id"], "source": txt_path.stem, "text": c["text"]})
    if not docs:
        raise ValueError(f"no .txt files found under {source_dir}")
    return docs


DEFAULT_CORPUS = "default"


def list_corpora(state: EngineState, method: str = DEFAULT_METHOD) -> list[str]:
    """Every corpus that actually has a built index for `method`.

    "hybrid" isn't its own stored index (see build_index()) -- it fuses the
    "semantic" and "tfidf" indexes at query time, so listing for it means
    listing corpora that have a "semantic" index on disk, same translation
    retrieve() already does for its own listing_method.
    """
    root = state.config.data_dir / "rag_index"
    if not root.exists():
        return []
    dir_name = "tfidf" if method == "tfidf" else "semantic"
    marker = "matrix.pkl" if method == "tfidf" else "embeddings.npy"
    return sorted(
        d.parent.parent.name for d in root.glob(f"*/{dir_name}/{marker}")
    )


def _paths(state: EngineState, method: str, corpus: str = DEFAULT_CORPUS) -> dict[str, Path]:
    d = state.config.data_dir / "rag_index" / corpus / method
    if method == "tfidf":
        return {
            "dir": d,
            "vectorizer": d / "vectorizer.pkl",
            "matrix": d / "matrix.pkl",
            "meta": d / "chunks.json",
        }
    return {"dir": d, "embeddings": d / "embeddings.npy", "meta": d / "chunks.json"}


# ── build ────────────────────────────────────────────────────────────────────
def build_index(
    state: EngineState, source_dir: Path, method: str = DEFAULT_METHOD, corpus: str = DEFAULT_CORPUS
) -> dict[str, Any]:
    if method == "hybrid":
        # "hybrid" isn't its own stored index — it's semantic + tfidf fused
        # at query time — so building it means building both.
        build_index(state, source_dir, method="tfidf", corpus=corpus)
        result = build_index(state, source_dir, method="semantic", corpus=corpus)
        return {**result, "method": "hybrid"}

    docs = _chunk_source_dir(source_dir)
    p = _paths(state, method, corpus)
    p["dir"].mkdir(parents=True, exist_ok=True)

    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=50000, stop_words=_stop_words(), ngram_range=(1, 2))
        matrix = vectorizer.fit_transform([d["text"] for d in docs])
        with p["vectorizer"].open("wb") as f:
            pickle.dump(vectorizer, f)
        with p["matrix"].open("wb") as f:
            pickle.dump(matrix, f)
    elif method == "semantic":
        vecs, n_embedded, n_reused = _embed_incremental(p, docs)
        np.save(p["embeddings"], vecs)
    else:
        raise ValueError(f"unknown RAG method '{method}' (expected 'tfidf' or 'semantic')")

    p["meta"].write_text(json.dumps(docs), encoding="utf-8")
    _CACHE.pop(f"{corpus}:{method}:{p['dir']}", None)
    result = {
        "method": method, "corpus": corpus, "chunks": len(docs),
        "sources": sorted({d["source"] for d in docs}),
    }
    if method == "semantic":
        result["embedded"] = n_embedded
        result["reused"] = n_reused
    return result


def _embed_incremental(p: dict[str, Path], docs: list[dict[str, Any]]) -> tuple[np.ndarray, int, int]:
    """Embedding is the most CPU-expensive ingest step (a neural network
    forward pass per chunk) -- re-embedding an entire corpus for one added
    file wastes most of that work. The chunker's ids are content-addressed
    (sha256 of the chunk's own text), so "same id" always means "identical
    text" by construction -- reusing an old vector for a matching id can
    never silently serve a stale embedding for changed content, since
    changed content necessarily gets a new id. Returns (vectors, n_newly_
    embedded, n_reused)."""
    old_vecs_by_id: dict[str, np.ndarray] = {}
    if p["embeddings"].exists() and p["meta"].exists():
        try:
            old_meta = json.loads(p["meta"].read_text(encoding="utf-8"))
            old_vecs = np.load(p["embeddings"])
            old_vecs_by_id = {m["id"]: old_vecs[i] for i, m in enumerate(old_meta) if i < len(old_vecs)}
        except Exception:
            old_vecs_by_id = {}  # corrupt/unreadable old index -- fall back to a full rebuild

    vecs: list[Any] = [None] * len(docs)
    new_texts: list[str] = []
    new_indices: list[int] = []
    for i, d in enumerate(docs):
        cached = old_vecs_by_id.get(d["id"])
        if cached is not None:
            vecs[i] = cached
        else:
            new_texts.append(d["text"])
            new_indices.append(i)

    if new_texts:
        fresh = _embed_texts(new_texts)
        for idx, vec in zip(new_indices, fresh):
            vecs[idx] = vec

    return np.array(vecs, dtype=np.float32), len(new_texts), len(docs) - len(new_texts)


# ── load / retrieve ──────────────────────────────────────────────────────────
def _load(state: EngineState, method: str, corpus: str = DEFAULT_CORPUS) -> tuple[Any, ...]:
    p = _paths(state, method, corpus)
    key = f"{corpus}:{method}:{p['dir']}"
    if key in _CACHE:
        return _CACHE[key]

    if method == "tfidf":
        if not p["vectorizer"].exists():
            raise FileNotFoundError(
                f"TF-IDF RAG index for corpus '{corpus}' not built yet — "
                f"run `rag build --method tfidf --corpus {corpus}` first"
            )
        with p["vectorizer"].open("rb") as f:
            vectorizer = pickle.load(f)
        with p["matrix"].open("rb") as f:
            matrix = pickle.load(f)
        meta = json.loads(p["meta"].read_text(encoding="utf-8"))
        _CACHE[key] = (vectorizer, matrix, meta)
    elif method == "semantic":
        if not p["embeddings"].exists():
            raise FileNotFoundError(
                f"Semantic RAG index for corpus '{corpus}' not built yet — "
                f"run `rag build --method semantic --corpus {corpus}` first"
            )
        vecs = np.load(p["embeddings"])
        meta = json.loads(p["meta"].read_text(encoding="utf-8"))
        _CACHE[key] = (vecs, meta)
    else:
        raise ValueError(f"unknown RAG method '{method}' (expected 'tfidf' or 'semantic')")
    return _CACHE[key]


def _retrieve_one(
    state: EngineState, query: str, top_k: int, method: str, corpus: str
) -> list[dict[str, Any]]:
    if method == "tfidf":
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer, matrix, meta = _load(state, method, corpus)
        qvec = vectorizer.transform([query])
        sims = cosine_similarity(qvec, matrix)[0]
    elif method == "semantic":
        vecs, meta = _load(state, method, corpus)
        qvec = _embed_texts([query])[0]  # already L2-normalized
        sims = vecs @ qvec
    else:
        raise ValueError(f"unknown RAG method '{method}' (expected 'tfidf' or 'semantic')")

    top_idx = sims.argsort()[::-1][:top_k]
    return [
        {
            "score": float(sims[i]), "source": meta[i]["source"], "text": meta[i]["text"],
            "corpus": corpus, "id": meta[i]["id"],
        }
        for i in top_idx
        if sims[i] > 0
    ]


def _rrf_fuse(rankings: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion: combine ranked lists from different retrieval
    methods using rank position (1/(k+rank)) rather than raw scores, which
    aren't comparable between e.g. TF-IDF cosine similarity and embedding
    cosine similarity."""
    rrf_scores: dict[tuple[str, str], float] = {}
    items: dict[tuple[str, str], dict[str, Any]] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit["corpus"], hit["id"])
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            items.setdefault(key, hit)
    ordered_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)
    return [items[key] for key in ordered_keys]


def _rerank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not hits:
        return hits
    reranker = _reranker()
    scores = reranker.predict([(query, h["text"]) for h in hits])
    for h, s in zip(hits, scores):
        h["score"] = float(s)
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


def _hybrid_candidates_one(
    state: EngineState, query: str, candidate_k: int, corpus: str
) -> list[dict[str, Any]]:
    semantic_hits = _retrieve_one(state, query, candidate_k, "semantic", corpus)
    try:
        tfidf_hits = _retrieve_one(state, query, candidate_k, "tfidf", corpus)
    except FileNotFoundError:
        tfidf_hits = []  # no TF-IDF index for this corpus — degrade to semantic-only
    return _rrf_fuse([semantic_hits, tfidf_hits])


def retrieve(
    state: EngineState,
    query: str,
    top_k: int = 5,
    method: str = DEFAULT_METHOD,
    corpus: str | list[str] = DEFAULT_CORPUS,
) -> list[dict[str, Any]]:
    """`corpus` can be one name, a list of names, or "all" to search every
    corpus that has a built index.

    method="hybrid" (default): fuse semantic + TF-IDF rankings per corpus via
    RRF, then rerank the merged candidate pool with a cross-encoder — the
    final scores are cross-encoder relevance scores, comparable across
    corpora since one model scores everything.

    method="semantic"/"tfidf": raw cosine similarities from that one method,
    comparable across corpora since the same embedding model is used
    throughout — merging and re-sorting is valid, not just concatenation.
    """
    listing_method = "tfidf" if method == "tfidf" else "semantic"
    if corpus == "all":
        names = list_corpora(state, listing_method)
        if not names:
            raise FileNotFoundError(f"no RAG indexes built yet for method '{listing_method}'")
    elif isinstance(corpus, str):
        names = [corpus]
    else:
        names = list(corpus)

    if method == "hybrid":
        candidate_k = max(top_k * 4, 20)
        candidates: list[dict[str, Any]] = []
        for name in names:
            candidates.extend(_hybrid_candidates_one(state, query, candidate_k, name))
        reranked = _rerank(query, candidates)
        return reranked[:top_k]

    all_hits: list[dict[str, Any]] = []
    for name in names:
        all_hits.extend(_retrieve_one(state, query, top_k, method, name))
    all_hits.sort(key=lambda h: h["score"], reverse=True)
    return all_hits[:top_k]


QUESTION_MODES = ("first_sentence", "keywords")

# Letters only (any script, incl. accented Spanish -- the previous
# [a-zA-Z]{4,} split "preparación" into "preparaci"), no digits/underscore.
_EVAL_WORD_RE = re.compile(r"[^\W\d_]{4,}")
# 5+ letters for keyword queries: short enough to keep real content words,
# long enough to skip most function words in English and Spanish alike
# ("that", "with", "para", "como", "este"...) without a stopword list.
_KEYWORD_RE = re.compile(r"[^\W\d_]{5,}")


def _eval_sample(
    state: EngineState, corpus: str, method: str, held_out_fraction: float, seed: int
) -> list[dict[str, Any]]:
    import random

    meta_method = "tfidf" if method == "tfidf" else "semantic"
    p = _paths(state, meta_method, corpus)
    if not p["meta"].exists():
        raise FileNotFoundError(f"no RAG index built for corpus '{corpus}' -- build one first")
    all_chunks = json.loads(p["meta"].read_text(encoding="utf-8"))
    if not all_chunks:
        raise ValueError(f"corpus '{corpus}' has no chunks")
    rng = random.Random(seed)
    n_held_out = max(1, int(len(all_chunks) * held_out_fraction))
    return rng.sample(all_chunks, min(n_held_out, len(all_chunks)))


def _keyword_question(text: str, seed_key: str, n_words: int = 6) -> str:
    """A sparse query of `n_words` distinct content words drawn from the
    chunk (kept in their original order), deterministic per chunk. Never a
    verbatim sentence, but still the chunk's own exact words -- so this is
    a *lexical* test: measured on the real aprocrifos-v2 corpus it's near-
    trivial for tfidf (hit@1 1.00) and hard for semantic (hit@1 0.05, vs
    0.30 with first_sentence). Useful for comparing retrieval methods, not
    as a paraphrase-robustness test (that would need real human questions)."""
    import random

    seen: dict[str, None] = {}
    for w in _KEYWORD_RE.findall(text):
        seen.setdefault(w.lower(), None)
    words = list(seen)
    if len(words) > n_words:
        keep = set(random.Random(seed_key).sample(range(len(words)), n_words))
        words = [w for i, w in enumerate(words) if i in keep]
    return " ".join(words) if words else text[:120]


def eval_questions(chunks: list[dict[str, Any]], question_mode: str) -> list[dict[str, Any]]:
    """[{question, answer, id}] per chunk. first_sentence reuses
    build_qa_dataset()'s heuristic (the original behavior, unchanged);
    keywords builds a sparse exact-word query instead -- see _keyword_question."""
    from ..ingestion.pipeline import build_qa_dataset

    if question_mode not in QUESTION_MODES:
        raise ValueError(f"question_mode must be one of {QUESTION_MODES}")
    qa = build_qa_dataset(chunks)
    if question_mode == "keywords":
        for c, row in zip(chunks, qa):
            row["question"] = _keyword_question(c["text"], c["id"])
    return qa


def evaluate_retrieval(
    state: EngineState,
    corpus: str,
    method: str = DEFAULT_METHOD,
    held_out_fraction: float = 0.1,
    top_k: int = 5,
    seed: int = 42,
    question_mode: str = "first_sentence",
) -> dict[str, Any]:
    """Measures whether the RAG index actually retrieves the right passage
    for a question generated from that passage -- model-independent (tests
    the index/retrieval quality itself, not any particular chat model's
    answers), and needs no separate labeled eval set: generates eval
    questions from a held-out slice of the corpus's own already-indexed
    chunks (see eval_questions() for the two question modes).

    recall_at_k: fraction of held-out questions where the source chunk's
    own id appears somewhere in the top-k results for its own question.
    hit_at_1: ...appears as the very first result.
    mrr: mean reciprocal rank of the source chunk (0 when not in top-k) --
    distinguishes "found at #1" from "found at #5", which recall@k can't;
    first_sentence mode commonly saturates recall@k at 1.0.
    """
    held_out = _eval_sample(state, corpus, method, held_out_fraction, seed)
    qa_pairs = eval_questions(held_out, question_mode)

    details = []
    hits = hits_at_1 = 0
    rr_sum = 0.0
    for chunk, qa in zip(held_out, qa_pairs):
        results = retrieve(state, qa["question"], top_k=top_k, method=method, corpus=corpus)
        ids = [r["id"] for r in results]
        rank = ids.index(chunk["id"]) + 1 if chunk["id"] in ids else None
        hits += rank is not None
        hits_at_1 += rank == 1
        rr_sum += 1.0 / rank if rank else 0.0
        details.append({
            "question": qa["question"], "expected_id": chunk["id"],
            "found_in_top_k": rank is not None, "rank": rank,
        })

    n = len(held_out)
    return {
        "corpus": corpus,
        "method": method,
        "question_mode": question_mode,
        "top_k": top_k,
        "n_evaluated": n,
        "recall_at_k": hits / n if n else 0.0,
        "hit_at_1": hits_at_1 / n if n else 0.0,
        "mrr": rr_sum / n if n else 0.0,
        "details": details,
    }


def evaluate_answers(
    state: EngineState,
    corpus: str,
    model_name: str,
    method: str = DEFAULT_METHOD,
    held_out_fraction: float = 0.1,
    top_k: int = 3,
    max_new_tokens: int = 128,
    seed: int = 42,
    question_mode: str = "first_sentence",
) -> dict[str, Any]:
    """Rough answer-quality proxy: generates an actual RAG-grounded answer
    for each held-out question and measures keyword overlap against the
    ground-truth answer text (fraction of the expected answer's significant
    -- 4+ letter -- words that appear as words in the generated answer).
    Crude compared to a real BLEU/ROUGE score, but needs no extra
    dependency and gives a directional "did this help" signal, which is
    what this is actually for -- not a precise quality score.

    refusal_rate: fraction of answers that are the grounding prompt's
    "context does not contain" refusal -- a model that refuses everything
    scores ~0 overlap for a different reason than one that hallucinates.
    """
    from .model_loader import chat as core_chat

    held_out = _eval_sample(state, corpus, method, held_out_fraction, seed)
    qa_pairs = eval_questions(held_out, question_mode)

    details = []
    overlaps = []
    refusals = 0
    for qa in qa_pairs:
        results = retrieve(state, qa["question"], top_k=top_k, method=method, corpus=corpus)
        prompt = build_prompt(qa["question"], results)
        out = core_chat(state, prompt, model_name=model_name, max_new_tokens=max_new_tokens, temperature=0.0)
        generated = out["text"].lower()
        refused = is_refusal(generated)
        refusals += refused
        gen_words = set(_EVAL_WORD_RE.findall(generated))
        expected_words = {w.lower() for w in _EVAL_WORD_RE.findall(qa["answer"])}
        overlap = (
            sum(1 for w in expected_words if w in gen_words) / len(expected_words)
            if expected_words else 0.0
        )
        overlaps.append(overlap)
        details.append({
            "question": qa["question"], "expected_answer": qa["answer"],
            "generated": out["text"], "keyword_overlap": overlap, "refused": refused,
        })

    n = len(held_out)
    return {
        "corpus": corpus, "model": model_name, "method": method,
        "question_mode": question_mode,
        "n_evaluated": n,
        "avg_keyword_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "refusal_rate": refusals / n if n else 0.0,
        "details": details,
    }


REFUSALS = {
    "English": "The provided context does not contain this information.",
    "Spanish": "El contexto proporcionado no contiene esta información.",
}
PROMPT_STYLES = ("default", "native", "complete", "quote")
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def has_script_drift(answer: str, lang: str) -> bool:
    """Qwen-family local models sometimes switch to Chinese mid-answer
    (seen live: "Salomé reaccionó con desconfianza y pidió验证我的理解..." for
    a Spanish question). True when a Spanish/English answer contains CJK."""
    return lang in ("Spanish", "English") and bool(_CJK_RE.search(answer or ""))


def language_reminder(lang: str) -> str:
    return ("\n\nIMPORTANTE: responde únicamente en español." if lang == "Spanish"
            else "\n\nIMPORTANT: answer only in English.")


def local_answer(state: EngineState, prompt: str, model_name: str, lang: str,
                 max_new_tokens: int = 250, temperature: float = 0.0) -> str:
    """Generate a grounded answer with a LOCAL model, guarding against
    script drift: regenerate once with an explicit language reminder; if
    it still drifts, cut the answer before the first foreign character."""
    from .model_loader import chat as core_chat

    text = core_chat(state, prompt, model_name=model_name, max_new_tokens=max_new_tokens,
                     temperature=temperature)["text"].strip()
    if has_script_drift(text, lang):
        text = core_chat(state, prompt + language_reminder(lang), model_name=model_name,
                         max_new_tokens=max_new_tokens, temperature=temperature)["text"].strip()
    if has_script_drift(text, lang):
        text = text[: _CJK_RE.search(text).start()].rstrip(" ,;:")
    return text


def is_refusal(text: str) -> bool:
    """True if `text` is (or contains) a grounding refusal in any supported
    language."""
    t = (text or "").lower()
    return any(r.lower().rstrip(".") in t for r in REFUSALS.values())


def question_language(question: str, hits: list[dict[str, Any]] | None = None) -> str:
    """Language to answer in. The question alone is often too short for the
    marker-based detector, so the retrieved text breaks ties."""
    from .summarize import _detect_language

    sample = question + " " + " ".join(h.get("text", "")[:400] for h in (hits or [])[:2])
    q_lang = _detect_language(question)
    if "¿" in question or "ñ" in question.lower():
        return "Spanish"
    return q_lang if q_lang == _detect_language(sample) else _detect_language(sample)


def _context_block(hits: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{h['corpus']} / {h['source']}]\n{h['text']}" for h in hits)


def build_prompt(question: str, hits: list[dict[str, Any]], style: str = "default") -> str:
    """Grounded-answer prompt.

    style:
      default -- English instructions, English refusal (original; used for
                 the cloud answerer).
      native  -- the same rules written in the question's language, refusal
                 in that language. For small local models, instructions in
                 the conversation's language are easier to follow.
      quote   -- native + the model must first copy the exact supporting
                 sentence(s) from the context, then answer from them only:
                 a standard faithfulness technique for small readers, and
                 the quote doubles as a visible citation.
    """
    if not hits:
        return question
    if style not in PROMPT_STYLES:
        raise ValueError(f"style must be one of {PROMPT_STYLES}")
    context = _context_block(hits)
    if style == "default":
        return (
            "Answer strictly from the context passages below. Do not use any outside "
            "knowledge, prior training, or assumptions beyond what is written here. "
            "Ground your answer in the specific passage(s) that support it — refer to "
            "which one you used. If the context does not contain the answer, respond "
            f'exactly: "{REFUSALS["English"]}"\n\n'
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer (using only the context above):"
        )

    lang = question_language(question, hits)
    refusal = REFUSALS.get(lang, REFUSALS["English"])
    if lang == "Spanish":
        rules = (
            "Responde la pregunta usando ÚNICAMENTE la información de los pasajes de "
            "contexto de abajo. No uses conocimientos propios ni supongas nada que no "
            "esté escrito en ellos. Responde en español, "
            + ("de forma precisa y COMPLETA: incluye todos los detalles de los pasajes "
               "que respondan a cada parte de la pregunta, en pocas oraciones. "
               if style == "complete" else "de forma breve y precisa, ")
            + "usando los nombres tal como aparecen en el texto. Si los pasajes no "
            f'contienen la respuesta, responde exactamente: "{refusal}"'
        )
        if style == "quote":
            rules += (
                "\n\nFormato obligatorio:\nCita: «copia aquí, palabra por palabra, la frase "
                "del contexto que responde la pregunta»\nRespuesta: una o dos oraciones "
                "basadas solo en esa cita."
                f'\nSi no existe tal frase en el contexto, escribe solo: "{refusal}"'
            )
        return f"{rules}\n\nContexto:\n{context}\n\nPregunta: {question}\n" + (
            "Cita:" if style == "quote" else "Respuesta:"
        )
    rules = (
        "Answer the question using ONLY the information in the context passages "
        "below. Do not use your own knowledge or assume anything not written there. "
        + ("Answer precisely and COMPLETELY: include every detail from the passages that "
           "answers each part of the question, in a few sentences. "
           if style == "complete" else "Answer briefly and precisely, ")
        + "using names exactly as they appear in the text. "
        f'If the passages do not contain the answer, respond exactly: "{refusal}"'
    )
    if style == "quote":
        rules += (
            "\n\nRequired format:\nQuote: \"copy here, word for word, the sentence from "
            "the context that answers the question\"\nAnswer: one or two sentences "
            f'based only on that quote.\nIf no such sentence exists, write only: "{refusal}"'
        )
    return f"{rules}\n\nContext:\n{context}\n\nQuestion: {question}\n" + (
        "Quote:" if style == "quote" else "Answer:"
    )


# ── P1 retrieval upgrades: multi-query, parent-child, MMR ──────────────────

GROUNDING_REFUSAL = REFUSALS["English"]  # back-compat name


_LIST_MARKER_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s*")


def expand_query(
    state: EngineState, question: str, n_variants: int = 2
) -> list[str]:
    """Rewrite the user's question into 1..n differently-phrased variants
    with the cloud provider, so retrieval can match passages that don't
    share the question's exact wording (the phrasing gap that live-tested
    as the single biggest retrieval failure on the aprocrifos corpus: a
    user asks "quien bautizo a jesus" while the book writes "el bautismo
    del Senor fue administrado por...").

    Best-effort and strictly optional: any failure (provider disabled, no
    key, network error) returns [question] unchanged -- the plain question
    is always searched too, so a failure can never make retrieval worse
    than not calling this at all. Deterministic ordering: the original
    question is always first.
    """
    from .cloud_chat import cloud_generate
    from .cloud_settings import get_cloud_settings

    settings = get_cloud_settings(state)
    if not (settings.get("enabled") and settings.get("api_key") and settings.get("model")):
        return [question]
    prompt = (
        f"Rewrite the following question in {n_variants} DIFFERENT ways that a "
        "book or document might phrase the same topic. Use synonyms and "
        "different sentence structure. Answer in the same language as the "
        f"question. Respond with {n_variants} lines, nothing else.\n\n"
        f"Question: {question}"
    )
    try:
        # No small explicit cap: reasoning models (glm-5.3-flash, the configured provider)
        # spend the budget in a separate "reasoning" field first -- measured
        # live, max_tokens=200 returned EMPTY content 2/2 times (zero
        # variants, so multi-query silently did nothing); 600/1024/2048 all
        # returned 2 good rewrites at the same ~1.5-2s latency.
        raw = cloud_generate(settings, prompt)  # default budget (cloud_settings.DEFAULT_MAX_TOKENS)
    except (ValueError, RuntimeError):
        return [question]
    variants = [_LIST_MARKER_RE.sub("", line).strip() for line in raw.splitlines() if line.strip()]
    variants = [v for v in variants if v and v.lower() != question.lower()][:n_variants]
    return [question] + variants


def expand_context(
    state: EngineState, hits: list[dict[str, Any]], parent_chars: int = 1600,
    window_sentences: int = 1,
) -> list[dict[str, Any]]:
    """Parent-child context expansion: each hit is a small chunk that was
    retrieved for precision, but the answer often continues in the
    surrounding text. Returns hits whose `text` is widened to include
    neighbors from the SAME source document (parent passage), so the
    generator sees the full passage, not a fragment.

    Neighbor text is pulled from the corpus chunk metadata (chunks.json),
    ordered by chunk_index within the same source, so this is a pure
    read-only widening of already-indexed content -- no re-extraction.
    """
    if not hits:
        return hits
    seen_meta: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    # (corpus:source, chunk index) already present in the context. Without
    # this, two hits from neighboring chunks of one document each pulled in
    # the other as "context" -- measured live: up to 10 of 53 paragraphs in
    # a grounded prompt were verbatim duplicates.
    used: set[tuple[str, int]] = set()
    out: list[dict[str, Any]] = []
    for h in hits:
        corpus, source = h.get("corpus"), h.get("source")
        key = f"{corpus}:{source}"
        if key not in seen_meta:
            chunks = _source_chunks(state, corpus, source)
            seen_meta[key] = (source, chunks)
        _, chunks = seen_meta[key]
        idx = next((i for i, c in enumerate(chunks) if c["id"] == h["id"]), None)
        if idx is None:
            out.append(h)
            continue
        if (key, idx) in used:
            continue  # already in the context as part of an earlier hit's window
        used.add((key, idx))
        texts = [chunks[idx]["text"]]
        # walk backwards/forwards in the same document, accumulating up to
        # parent_chars of surrounding parent text, window_sentences on each
        # side minimum
        lo, hi = max(0, idx - window_sentences), min(len(chunks) - 1, idx + window_sentences)
        n_chars = len(chunks[idx]["text"])
        # seed with the minimum window first so the ±1 context always lands
        # in the output even when parent_chars is already satisfied --
        # nearest-first on each side, preceding chunks inserted at the front
        # so the widened text stays in document order (it used to append the
        # preceding chunk AFTER the hit itself)
        for n in [*range(idx - 1, lo - 1, -1), *range(idx + 1, hi + 1)]:
            if (key, n) in used:
                continue
            used.add((key, n))
            if n < idx:
                texts.insert(0, chunks[n]["text"])
            else:
                texts.append(chunks[n]["text"])
            n_chars += len(chunks[n]["text"])
        # then keep widening toward parent_chars, alternating outward
        while n_chars < parent_chars and (lo > 0 or hi < len(chunks) - 1):
            if lo > 0:
                lo -= 1
                if (key, lo) not in used:
                    used.add((key, lo))
                    texts.insert(0, chunks[lo]["text"])
                    n_chars += len(chunks[lo]["text"])
                if n_chars >= parent_chars:
                    break
            if hi < len(chunks) - 1:
                hi += 1
                if (key, hi) not in used:
                    used.add((key, hi))
                    texts.append(chunks[hi]["text"])
                    n_chars += len(chunks[hi]["text"])
        text = "\n".join(texts)
        if len(text) > len(h["text"]):
            h = {**h, "text": text, "parent_expanded": True}
        out.append(h)
    return out


def _source_chunks(state: EngineState, corpus: str, source: str) -> list[dict[str, Any]]:
    """All indexed chunks for one source doc, in document order, from the
    semantic index's chunks.json (tfidf's has the same content)."""
    from .state import EngineState as _E  # noqa: F401  (type only)
    p = _paths(state, "semantic", corpus)
    if not p["meta"].exists():
        p = _paths(state, "tfidf", corpus)
    if not p["meta"].exists():
        return []
    try:
        meta = json.loads(p["meta"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [c for c in meta if c.get("source") == source]


NEAR_DUPLICATE_SIMILARITY = 0.9


def _mmr_diversify(hits: list[dict[str, Any]], top_k: int, lambda_: float = 0.72) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance: pick top_k hits balancing relevance and
    diversity, so the context isn't 6 copies of one paragraph. Char-3gram
    Jaccard as a cheap similarity proxy -- no model call needed.

    Measured on the real aprocrifos corpora:
      * corpus="all" (the grounded-chat default whenever the model has no
        rag_corpus) put the SAME passage 3-4 times into the top 6 -- the 4
        aprocrifos corpora were built from the same PDFs. Near-identical
        candidates (similarity >= NEAR_DUPLICATE_SIMILARITY) are therefore
        skipped outright while anything else remains.
      * within one corpus the plain top 6 is already non-redundant (mean max
        pair similarity 0.278, about the baseline for unrelated Spanish
        paragraphs); scaled MMR at lambda 0.72 kept gold-passage recall at
        37/40 (same as no MMR), lambda 0.5 dropped it to 36/40.
    """
    if len(hits) <= top_k:
        return hits

    def _shingles(t: str) -> set[str]:
        t = "".join(t.lower().split())
        return {t[i : i + 3] for i in range(max(0, len(t) - 2))}

    def _sim(i: int, j: int) -> float:
        return len(shingles[i] & shingles[j]) / max(1, len(shingles[i] | shingles[j]))

    shingles = [_shingles(h["text"]) for h in hits]
    # Relevance min-max scaled to 0..1 first. Raw cross-encoder scores span
    # roughly -5..+8 while the diversity penalty is at most (1-lambda)=0.28,
    # so unscaled MMR never changed a single selection (measured live on 3
    # real questions: MMR picks == plain top-k every time). Knowledge-graph
    # rows (fixed score 1.0, exact entity matches by construction) are
    # scaled as top relevance instead of being mixed into the logit range.
    raw = [h["score"] for h in hits if not h.get("kgraph")]
    lo, hi = (min(raw), max(raw)) if raw else (0.0, 1.0)
    rels = [
        1.0 if h.get("kgraph") else ((h["score"] - lo) / (hi - lo) if hi > lo else 1.0)
        for h in hits
    ]
    selected = [0]
    remaining = [i for i in range(1, len(hits))]
    while len(selected) < top_k and remaining:
        scored = []
        for i in remaining:
            div = max(_sim(i, j) for j in selected)
            scored.append((div >= NEAR_DUPLICATE_SIMILARITY, -(lambda_ * rels[i] - (1 - lambda_) * div), i))
        # non-duplicates first, then best MMR score; ties keep relevance order
        scored.sort()
        best = scored[0][2]
        if scored[0][0]:
            # only near-duplicates are left: stop rather than pad the
            # context with copies of what's already selected
            break
        selected.append(best)
        remaining.remove(best)
    return [hits[i] for i in selected]


GROUNDED_POOL_SIZE = 30


def _grounded_candidates(
    state: EngineState, queries: list[str], corpus: str | list[str],
    per_list_k: int = 20, pool_size: int = GROUNDED_POOL_SIZE,
) -> list[dict[str, Any]]:
    """Cheap first stage for grounded retrieval: semantic + TF-IDF top
    `per_list_k` for every (query variant x corpus), all fused in ONE RRF,
    exact copies dropped by content id, cut to `pool_size` -- so the
    expensive cross-encoder runs once over ~30 passages.

    It used to call retrieve(hybrid) per variant, which reranks its own
    ~100-passage pool per corpus, then rerank everything again: ~70s per
    grounded turn on corpus="all" (3 variants x 4 corpora). Measured on
    held-out aprocrifos questions: identical gold-passage-in-top-6 recall
    (11/12 first_sentence, 12/12 keywords, both single-corpus and "all"),
    2.4s vs 10.3s per query single-corpus and 2.4s vs 34.6s on "all".

    Chunk ids are sha256 of the chunk text, so the same passage indexed in
    several corpora (the aprocrifos corpora were built from the same PDFs)
    has the same id -- deduping by id keeps its best-ranked copy only.
    """
    if corpus == "all":
        names = list_corpora(state, "semantic")
        if not names:
            raise FileNotFoundError("no RAG indexes built yet")
    elif isinstance(corpus, str):
        names = [corpus]
    else:
        names = list(corpus)

    rankings: list[list[dict[str, Any]]] = []
    for q in queries:
        for name in names:
            for method in ("semantic", "tfidf"):
                try:
                    rankings.append(_retrieve_one(state, q, per_list_k, method, name))
                except FileNotFoundError:
                    continue  # this corpus lacks this index type -- use what exists
    if not rankings:
        raise FileNotFoundError(f"no RAG index found for corpus {corpus!r}")
    seen: set[str] = set()
    pool: list[dict[str, Any]] = []
    for h in _rrf_fuse(rankings):
        if h["id"] in seen:
            continue
        seen.add(h["id"])
        pool.append(dict(h))  # _rerank mutates scores; don't touch cached hits
        if len(pool) >= pool_size:
            break
    return pool


def retrieve_grounded(
    state: EngineState,
    question: str,
    top_k: int = 6,
    corpus: str | list[str] = "all",
    multi_query: bool = True,
    parent_child: bool = True,
    kgraph_enabled: bool = False,
    allow_cloud: bool = True,
    mmr: bool = True,
    graph_slot: bool = False,
    graph_slot_below: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One-call retrieval for grounded chat: multi-query expansion (P1) →
    cheap per-variant/per-corpus candidates fused by RRF (see
    _grounded_candidates) → ONE cross-encoder rerank → MMR
    diversity → parent-child widening (P1). Optionally merges in knowledge-
    graph passages (see kgraph.py) as additional context rows.

    Returns (hits, debug) where debug reports which steps actually ran
    (variants used, expansion applied, kgraph hits merged) -- surfaced in
    the API response so the behavior is observable, not assumed.

    allow_cloud=False skips multi-query expansion entirely (it's the only
    step that calls the cloud provider): a caller that asked for a
    local-only answer must not have its question sent out anyway.
    """
    debug: dict[str, Any] = {"variants": 1, "multi_query": False, "parent_expanded": 0, "kgraph_hits": 0}

    queries = [question]
    if multi_query and allow_cloud:
        queries = expand_query(state, question)
        debug["variants"] = len(queries)
        debug["multi_query"] = len(queries) > 1

    try:
        pool = _grounded_candidates(state, queries, corpus, pool_size=max(GROUNDED_POOL_SIZE, top_k * 3))
    except FileNotFoundError:
        return [], debug  # no indexes built at all
    if not pool:
        return [], debug

    # Graph-walk candidates (src/core/graph_retrieval.py): Personalized
    # PageRank over passages + knowledge-graph entities, seeded by matching
    # the question to entity fact-descriptions with the cross-encoder. They
    # join the reranker's pool, and ONE final slot is reserved for the best
    # graph passage the reranker didn't already choose. Measured on the hard
    # eval set (paraphrased / multi-step questions): gold passage in top-3
    # 0.42 -> 0.60 (paraphrase 0.25 -> 0.50, chained 0.44 -> 0.62), no loss
    # on the original set (0.80 -> 0.80). Plain rank fusion of all rankers
    # was measured WORSE (0.24) -- the graph is a candidate source, not a judge.
    #
    # ADAPTIVE: only when retrieval is unsure (best reranked passage scores
    # below graph_slot_below). Answer-level eval showed an always-on slot
    # helps hard questions (0.32 -> 0.44 accuracy, 0 wrong answers) but
    # hurts easy ones (0.67 -> 0.52: the slot displaced a useful 3rd passage
    # with a distractor). A top-1 threshold of 0 opens the slot for 6/25
    # easy questions but 13/25 hard ones, covering 8 of 11 hard misses.
    reranked = _rerank(question, pool)
    graph_hits: list[dict[str, Any]] = []
    single_corpus = corpus if isinstance(corpus, str) and corpus != "all" else None
    top1 = reranked[0]["score"] if reranked else float("-inf")
    if graph_slot and single_corpus and top1 < graph_slot_below:
        try:
            from . import graph_retrieval

            if graph_retrieval.available(state, single_corpus):
                graph_hits = graph_retrieval.rank(state, single_corpus, question, top_n=10, seeding="cross")
                ids = {h["id"] for h in pool}
                extra = [{"id": g["id"], "text": g["text"], "source": g["source"],
                          "corpus": single_corpus, "score": 0.0}
                         for g in graph_hits if g["id"] not in ids]
                if extra:
                    reranked = _rerank(question, reranked + extra)
                debug["graph_candidates"] = len(graph_hits)
        except Exception as e:  # optional layer: never break retrieval over it
            debug["graph_error"] = f"{type(e).__name__}: {e}"
            graph_hits = []
    elif graph_slot:
        debug["graph_slot_skipped"] = f"confident retrieval (top1={top1:.2f} >= {graph_slot_below})"

    # kgraph passages: entity/path context from the knowledge graph (if a
    # graph has been built for any corpus being searched)
    if kgraph_enabled:
        try:
            from .kgraph import retrieve_graph_context

            kq = [question] + queries[1:]
            kg_hits = retrieve_graph_context(state, kq, corpus=corpus)
            if kg_hits:
                debug["kgraph_hits"] = len(kg_hits)
                reranked = reranked + kg_hits
        except Exception:
            pass  # graph is optional; never break retrieval over it

    # mmr=False keeps plain rerank order -- exists for ablation (see
    # src/core/grounded_eval.py), not as a user-facing setting
    # (stable sort: knowledge-graph rows first, passages keep rerank order --
    # otherwise the appended graph rows would always fall past top_k)
    diversified = (
        _mmr_diversify(reranked, top_k) if mmr
        else sorted(reranked, key=lambda h: not h.get("kgraph"))[:top_k]
    )
    if graph_hits and top_k >= 2:
        chosen = {h["id"] for h in diversified[: top_k - 1]}
        best_graph = next((g for g in graph_hits if g["id"] not in chosen), None)
        if best_graph is not None:
            by_id = {h["id"]: h for h in reranked}
            slot = by_id.get(best_graph["id"]) or {**best_graph, "score": 0.0}
            diversified = diversified[: top_k - 1] + [{**slot, "graph_slot": True}]
            debug["graph_slot"] = best_graph["id"]
    if parent_child:
        diversified = expand_context(state, diversified)
        debug["parent_expanded"] = sum(1 for h in diversified if h.get("parent_expanded"))
    return diversified[:top_k], debug
