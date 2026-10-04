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

"""End-to-end answer-quality evaluation for grounded chat (P5).

The older rag.evaluate_retrieval()/evaluate_answers() build questions by
copying a chunk's own sentence/words -- fine for comparing retrievers, but
it can't tell whether a *user-style* question gets a *correct answer*. This
module measures the whole grounded path instead:

  1. Eval set (built once, saved as JSON, reused so runs are comparable):
       * single -- a natural question answerable from one chunk, written by
         the cloud model in the corpus language, told NOT to reuse the
         passage's wording (so retrieval has to bridge phrasing).
       * multi  -- a question needing two chunks, paired through an entity
         the knowledge graph found in both (falls back to two chunks of the
         same document when no graph exists). What a one-pass retriever is
         most likely to miss -- the P3 (agentic loop) decision input.
       * ooc    -- out-of-corpus questions; the right answer is a refusal.
  2. Per configuration (retrieval feature flags) and answerer (cloud or a
     local model): retrieve with rag.retrieve_grounded(), answer from
     rag.build_prompt(), then have the cloud model judge the answer against
     the reference (0 wrong / 1 partial / 2 correct, and whether it refused).

Metrics: context recall (gold passage text present in what the answerer
saw), mean correctness, strict accuracy, false-refusal rate on answerable
questions, refusal accuracy on out-of-corpus ones, latency, context size.

Caveat, stated in every report: the judge is the same cloud model that
writes cloud answers, so absolute scores may be flattering; differences
between configurations on the same set are the signal.
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from .state import EngineState

# Feature ablation: every config is the baseline plus ONE feature, plus the
# full current default -- so each feature's effect is isolated.
CONFIGS: dict[str, dict[str, bool]] = {
    "baseline": {"multi_query": False, "parent_child": False, "kgraph": False, "mmr": False},
    "+multi_query": {"multi_query": True, "parent_child": False, "kgraph": False, "mmr": False},
    "+parent_child": {"multi_query": False, "parent_child": True, "kgraph": False, "mmr": False},
    "+kgraph": {"multi_query": False, "parent_child": False, "kgraph": True, "mmr": False},
    "+mmr": {"multi_query": False, "parent_child": False, "kgraph": False, "mmr": True},
    "all": {"multi_query": True, "parent_child": True, "kgraph": True, "mmr": True},
    # graph-walk candidate slot (src/core/graph_retrieval.py) on top of the
    # offline default (MMR on, no widening)
    "+graph_slot": {"multi_query": False, "parent_child": False, "kgraph": False, "mmr": True,
                    "graph_slot": True},
    "mmr_only": {"multi_query": False, "parent_child": False, "kgraph": False, "mmr": True},
}

OOC_QUESTIONS = [
    # clearly unrelated
    "¿Cuál es la capital de Australia?",
    "¿Quién escribió Don Quijote de la Mancha?",
    # near-domain: plausible for this corpus, but not in these seven books
    "¿Qué dice el Evangelio de Judas sobre la traición a Jesús?",
    "¿Qué enseña el Evangelio de Tomás sobre el reino de los cielos?",
    "¿Qué papel tiene la Virgen María en el Corán?",
]

_JSON_RE = re.compile(r"\{.*\}", re.S)
_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*([012])')
_REFUSED_RE = re.compile(r'"?refused"?\s*[:=]\s*(true|false)', re.I)
ANSWER_ATTEMPTS = 3


def _eval_path(state: EngineState, corpus: str) -> Path:
    return state.config.data_dir / "eval" / f"{corpus}-grounded.json"


CLOUD_RETRY_WAITS = (5, 15, 45)


def _cloud(state: EngineState, prompt: str) -> str:
    """cloud_generate with backoff on transient failures. The second real
    ablation run died ~20 min in on one "Remote end closed connection
    without response" from the provider, losing the whole run."""
    from .cloud_chat import cloud_generate
    from .cloud_settings import get_cloud_settings

    for wait in (*CLOUD_RETRY_WAITS, None):
        try:
            return cloud_generate(get_cloud_settings(state), prompt)
        except RuntimeError:
            if wait is None:
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")


def _parse_json(raw: str) -> Optional[dict[str, Any]]:
    m = _JSON_RE.search(raw or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ── eval set ────────────────────────────────────────────────────────────────

def _single_prompt(passage: str) -> str:
    return (
        "You are building a reading-comprehension test for a document collection.\n"
        "Write ONE question that a curious reader might ask, which this passage "
        "answers. Rules: write it in the same language as the passage; phrase it "
        "naturally, the way a person would ask -- do NOT copy distinctive phrases "
        "from the passage; it must be answerable from this passage alone; avoid "
        "yes/no questions. Then give the correct answer in 1-2 sentences, "
        "strictly based on the passage.\n"
        'Respond ONLY with JSON: {"question": "...", "answer": "..."}\n\n'
        f"Passage:\n{passage}"
    )


def _multi_prompt(entity: str, a: str, b: str) -> str:
    return (
        "You are building a reading-comprehension test that requires combining "
        "two passages.\n"
        f"Both passages mention: {entity}.\n"
        "Write ONE question whose complete answer needs information from BOTH "
        "passages (not answerable from either one alone). Same language as the "
        "passages, natural wording, no copied phrases, not yes/no. Then give the "
        "correct answer in 1-3 sentences, based only on the two passages. If no "
        'sensible such question exists, respond {"question": null}.\n'
        'Respond ONLY with JSON: {"question": "...", "answer": "..."}\n\n'
        f"Passage A:\n{a}\n\nPassage B:\n{b}"
    )


def _multi_pairs(state: EngineState, corpus: str, chunks: list[dict[str, Any]],
                 rng: random.Random, n: int, exclude: set[str]) -> list[tuple[str, dict, dict]]:
    """(entity, chunk_a, chunk_b) pairs sharing a knowledge-graph entity,
    chunks non-adjacent (a neighbor pair is too easy with parent-child)."""
    by_id = {c["id"]: c for c in chunks}
    order = {c["id"]: i for i, c in enumerate(chunks)}
    db = state.config.data_dir / "kgraph" / f"{corpus}.db"
    pairs: list[tuple[str, dict, dict]] = []
    if db.exists():
        con = sqlite3.connect(str(db))
        rows = con.execute(
            """SELECT e.name, GROUP_CONCAT(DISTINCT t.chunk_id) FROM triples t
               JOIN entities e ON e.id IN (t.subject_id, t.object_id)
               GROUP BY e.id HAVING COUNT(DISTINCT t.chunk_id) BETWEEN 2 AND 6"""
        ).fetchall()
        con.close()
        rng.shuffle(rows)
        used: set[str] = set()
        for name, ids in rows:
            cand = [i for i in ids.split(",") if i in by_id and i not in exclude and i not in used]
            cand.sort(key=lambda i: order[i])
            for x in range(len(cand)):
                for y in range(x + 1, len(cand)):
                    if abs(order[cand[x]] - order[cand[y]]) >= 3:
                        pairs.append((name, by_id[cand[x]], by_id[cand[y]]))
                        used.update((cand[x], cand[y]))
                        break
                else:
                    continue
                break
            if len(pairs) >= n * 2:  # extra, some get rejected as "no question"
                break
    if not pairs:  # no graph: two far-apart chunks of the same document
        by_src: dict[str, list[dict]] = {}
        for c in chunks:
            if c["id"] not in exclude:
                by_src.setdefault(c["source"], []).append(c)
        for src, cs in by_src.items():
            if len(cs) >= 6:
                pairs.append((src, cs[len(cs) // 4], cs[3 * len(cs) // 4]))
    return pairs


def build_eval_set(
    state: EngineState, corpus: str, n_single: int = 20, n_multi: int = 5,
    seed: int = 7, log: Callable[[str], None] = print,
) -> dict[str, Any]:
    from .rag import _paths

    meta = _paths(state, "semantic", corpus)["meta"]
    if not meta.exists():
        raise FileNotFoundError(f"no RAG index for corpus '{corpus}'")
    chunks = json.loads(meta.read_text(encoding="utf-8"))
    rng = random.Random(seed)

    items: list[dict[str, Any]] = []
    picked = rng.sample(chunks, min(len(chunks), n_single * 2))
    for c in picked:
        if sum(1 for i in items if i["type"] == "single") >= n_single:
            break
        if len(c["text"]) < 300:
            continue
        qa = _parse_json(_cloud(state, _single_prompt(c["text"])))
        if not qa or not qa.get("question") or not qa.get("answer"):
            log(f"  skip chunk {c['id']}: no usable question")
            continue
        items.append({"id": f"s{len(items)}", "type": "single", "question": qa["question"],
                      "reference": qa["answer"], "gold": [c["id"]], "source": c["source"]})
        log(f"  single {len(items)}: {qa['question'][:90]}")

    used = {g for i in items for g in i["gold"]}
    for entity, a, b in _multi_pairs(state, corpus, chunks, rng, n_multi, used):
        if sum(1 for i in items if i["type"] == "multi") >= n_multi:
            break
        qa = _parse_json(_cloud(state, _multi_prompt(entity, a["text"], b["text"])))
        if not qa or not qa.get("question") or not qa.get("answer"):
            continue
        items.append({"id": f"m{len(items)}", "type": "multi", "question": qa["question"],
                      "reference": qa["answer"], "gold": [a["id"], b["id"]], "entity": entity})
        log(f"  multi: {qa['question'][:90]}")

    for q in OOC_QUESTIONS:
        items.append({"id": f"o{len(items)}", "type": "ooc", "question": q,
                      "reference": None, "gold": []})

    ev = {"corpus": corpus, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed, "items": items}
    path = _eval_path(state, corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {path} ({len(items)} items)")
    return ev


def load_eval_set(state: EngineState, corpus: str, name: str = "grounded") -> dict[str, Any]:
    path = state.config.data_dir / "eval" / f"{corpus}-{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ── harder set (v2): paraphrase / chained multi-step / why-how / near-OOC ──

HARD_OOC_QUESTIONS = [
    "¿Qué dice el Evangelio de Felipe sobre la cámara nupcial?",
    "¿Cuántos hijos tuvieron José y María según el Evangelio de Nicodemo?",
    "¿Qué castigo recibe Judas Iscariote en el Apocalipsis de Pedro?",
    "¿En qué año fue escrito el Protoevangelio de Santiago según el propio texto?",
    "¿Qué milagro hizo Jesús en las bodas de Caná según el Evangelio de María?",
]


def _paraphrase_prompt(passage: str) -> str:
    return (
        "You are writing a HARD reading-comprehension question about a passage.\n"
        "Write ONE question answerable from the passage, but phrased the way a person "
        "who has NOT read the text would ask it: use synonyms and everyday wording, and "
        "do NOT reuse the passage's distinctive words (names are allowed only if "
        "unavoidable; prefer describing people by their role). Not yes/no. Same language "
        "as the passage. Then the correct answer in 1-2 sentences from the passage.\n"
        'Respond ONLY with JSON: {"question": "...", "answer": "..."}\n\n'
        f"Passage:\n{passage}"
    )


def _why_prompt(passage: str) -> str:
    return (
        "Write ONE 'why' or 'how' question about the passage whose answer requires "
        "understanding the passage (a reason, cause, purpose or process stated in it), "
        "not just copying a name or number. Same language as the passage, natural "
        "wording. Then the correct answer in 1-2 sentences, based only on the passage.\n"
        'Respond ONLY with JSON: {"question": "...", "answer": "..."}\n\n'
        f"Passage:\n{passage}"
    )


def _chained_prompt(entity: str, a: str, b: str) -> str:
    return (
        "Write ONE multi-step question that requires chaining two passages.\n"
        f"Both passages involve: {entity}.\n"
        "Rules: the question must NOT name this entity directly; instead identify it "
        "with a clue that only Passage A provides (something it did, said or was), and "
        "then ask for a fact that only Passage B provides about it. So the reader must "
        "use A to know who/what is meant, and B to answer. Same language as the "
        "passages, natural wording, not yes/no. Then the correct answer in 1-2 "
        'sentences. If impossible, respond {"question": null}.\n'
        'Respond ONLY with JSON: {"question": "...", "answer": "..."}\n\n'
        f"Passage A:\n{a}\n\nPassage B:\n{b}"
    )


def build_hard_eval_set(
    state: EngineState, corpus: str, n_paraphrase: int = 12, n_chained: int = 8,
    n_why: int = 5, seed: int = 21, log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Harder companion set (saved as <corpus>-hard.json), built from chunks
    NOT used as gold passages by the first set."""
    from .rag import _paths

    chunks = json.loads(_paths(state, "semantic", corpus)["meta"].read_text(encoding="utf-8"))
    try:
        used = {g for it in load_eval_set(state, corpus)["items"] for g in it["gold"]}
    except FileNotFoundError:
        used = set()
    pool = [c for c in chunks if c["id"] not in used and len(c["text"]) >= 400]
    rng = random.Random(seed)
    rng.shuffle(pool)
    items: list[dict[str, Any]] = []

    def add(kind: str, prompt: str, gold: list[str], extra: Optional[dict] = None) -> bool:
        qa = _parse_json(_cloud(state, prompt))
        if not qa or not qa.get("question") or not qa.get("answer"):
            return False
        items.append({"id": f"h{len(items)}", "type": "multi" if kind == "chained" else "single",
                      "kind": kind, "question": qa["question"], "reference": qa["answer"],
                      "gold": gold, **(extra or {})})
        log(f"  {kind}: {qa['question'][:100]}")
        return True

    it = iter(pool)
    for kind, n, mk in (("paraphrase", n_paraphrase, _paraphrase_prompt), ("why", n_why, _why_prompt)):
        made = 0
        for c in it:
            if made >= n:
                break
            made += add(kind, mk(c["text"]), [c["id"]])
    taken = {g for i in items for g in i["gold"]}
    made = 0
    for entity, a, b in _multi_pairs(state, corpus, chunks, rng, n_chained * 2, used | taken):
        if made >= n_chained:
            break
        made += add("chained", _chained_prompt(entity, a["text"], b["text"]), [a["id"], b["id"]],
                    {"entity": entity})
    for q in HARD_OOC_QUESTIONS:
        items.append({"id": f"h{len(items)}", "type": "ooc", "kind": "ooc-near", "question": q,
                      "reference": None, "gold": []})
    ev = {"corpus": corpus, "name": "hard", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "seed": seed, "items": items}
    path = state.config.data_dir / "eval" / f"{corpus}-hard.json"
    path.write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {path} ({len(items)} items)")
    return ev


# ── judging ─────────────────────────────────────────────────────────────────

def _judge_prompt(question: str, reference: Optional[str], answer: str) -> str:
    ref = reference if reference else (
        "NONE -- the document collection does not contain this. The correct "
        "behavior is to say the information is not available."
    )
    return (
        "Grade an answer to a question about a document collection.\n"
        f"Question: {question}\n"
        f"Reference answer: {ref}\n"
        f"Candidate answer: {answer}\n\n"
        "score: 2 = correct and complete w.r.t. the reference (extra correct "
        "detail is fine); 1 = partially correct or incomplete; 0 = wrong, "
        "contradicts the reference, or does not answer.\n"
        "refused: true if the candidate says the information is not available "
        "/ not in the context (in any language), false otherwise.\n"
        "When the reference is NONE: score 2 if refused, 0 if it answered anyway.\n"
        'Respond ONLY with JSON: {"score": 0|1|2, "refused": true|false}'
    )


def judge(state: EngineState, question: str, reference: Optional[str], answer: str) -> dict[str, Any]:
    if not answer.strip():
        return {"score": 0, "refused": False, "judge_error": "empty answer"}
    # 3 attempts + a regex fallback: in the first real ablation run, 3 of
    # 180 judge replies weren't parseable JSON and were silently scored 0,
    # which looked like a config being worse when it wasn't.
    for _ in range(3):
        try:
            raw = _cloud(state, _judge_prompt(question, reference, answer))
        except RuntimeError as e:  # provider down even after backoff
            return {"score": 0, "refused": False, "judge_error": f"judge call failed: {e}"}
        v = _parse_json(raw)
        if v and v.get("score") in (0, 1, 2):
            return {"score": int(v["score"]), "refused": bool(v.get("refused"))}
        m = _SCORE_RE.search(raw or "")
        if m:
            r = _REFUSED_RE.search(raw)
            return {"score": int(m.group(1)), "refused": bool(r and r.group(1) == "true")}
    return {"score": 0, "refused": False, "judge_error": "unparseable judge output"}


# ── running ─────────────────────────────────────────────────────────────────

def _context_recall(item: dict[str, Any], hits: list[dict[str, Any]], chunk_text: dict[str, str]) -> Optional[float]:
    """Fraction of gold chunks whose text actually reached the answerer --
    by id, or contained in a parent-expanded hit's widened text."""
    if not item["gold"]:
        return None
    ids = {h.get("id") for h in hits}
    ctx = "\n".join(h.get("text", "") for h in hits)
    found = sum(1 for g in item["gold"] if g in ids or chunk_text.get(g, "\x00")[:300] in ctx)
    return found / len(item["gold"])


def run_config(
    state: EngineState, eval_set: dict[str, Any], config: dict[str, bool],
    answerer: str = "cloud", top_k: int = 6, workers: int = 3,
    log: Callable[[str], None] = print, prompt_style: str = "default",
    max_new_tokens: int = 200, kb_threshold: Optional[float] = None,
    kb_verify: str = "strict", kb_strategy: str = "lookup",
) -> list[dict[str, Any]]:
    """answerer: "cloud", "local:<registered model>", or
    "compiled:<kb name>:<reader model>" (knowledge_compiler.answer: compiled
    lookup first, then the verified local reader, else refusal)."""
    from . import rag
    from .model_loader import chat as core_chat

    corpus = eval_set["corpus"]
    meta = rag._paths(state, "semantic", corpus)["meta"]
    chunk_text = {c["id"]: c["text"] for c in json.loads(meta.read_text(encoding="utf-8"))}

    # retrieval: sequential (CPU-bound, shared embedder/reranker)
    rows = []
    for item in eval_set["items"]:
        t = time.perf_counter()
        hits, dbg = rag.retrieve_grounded(
            state, item["question"], top_k=top_k, corpus=corpus,
            multi_query=config["multi_query"], parent_child=config["parent_child"],
            kgraph_enabled=config["kgraph"], mmr=config["mmr"],
            graph_slot=config.get("graph_slot", False),
            allow_cloud=config["multi_query"],
        )
        rows.append({
            "id": item["id"], "type": item["type"], "question": item["question"],
            "retrieval_s": round(time.perf_counter() - t, 2),
            "context_recall": _context_recall(item, hits, chunk_text),
            "context_chars": sum(len(h.get("text", "")) for h in hits),
            "kgraph_rows": sum(1 for h in hits if h.get("kgraph")),
            "prompt": rag.build_prompt(item["question"], hits, prompt_style),
            "_item": item,
        })

    def answer_and_judge(row: dict[str, Any]) -> dict[str, Any]:
        item = row.pop("_item")
        prompt = row.pop("prompt")
        t = time.perf_counter()
        try:
            if answerer.startswith("compiled:"):
                from . import knowledge_compiler as kc

                _, kb, reader = answerer.split(":", 2)
                res = kc.answer(state, item["question"], corpus, reader, kb=kb,
                                threshold=kc.DEFAULT_MATCH_THRESHOLD if kb_threshold is None else kb_threshold,
                                top_k=top_k, max_new_tokens=max_new_tokens, verify=kb_verify,
                                reader_style=prompt_style if prompt_style != "default" else "native",
                                strategy=kb_strategy)
                ans = res["text"]
                row["mode"] = res["mode"]
                row["kb_score"] = res.get("score")
                row["matched_question"] = res.get("matched_question")
            elif answerer == "cloud":
                # empty answers are a provider failure (reasoning model ran
                # dry), not a wrong answer -- retry, and exclude if it persists
                ans = ""
                for _ in range(ANSWER_ATTEMPTS):
                    ans = _cloud(state, prompt)
                    if ans.strip():
                        break
            else:
                ans = rag.local_answer(state, prompt, answerer.split(":", 1)[1],
                                       rag.question_language(item["question"]), max_new_tokens)
        except Exception as e:  # one failed call scores 0, never kills the run
            ans, row["answer_error"] = "", f"{type(e).__name__}: {e}"
        row["answer_s"] = round(time.perf_counter() - t, 2)
        row["answer"] = ans
        row.update(judge(state, item["question"], item.get("reference"), ans))
        # failures of the measuring apparatus (no answer after retries,
        # judge never parseable) are excluded from quality metrics and
        # counted separately -- scoring them 0 conflated noise with quality
        row["excluded"] = bool(row.get("judge_error") or row.get("answer_error"))
        return row

    if answerer == "cloud":
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(answer_and_judge, rows))
    else:  # one local model instance: sequential
        results = [answer_and_judge(r) for r in rows]
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(xs: list[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 3) if xs else None

    scored = [r for r in results if not r.get("excluded")]
    ans = [r for r in scored if r["type"] != "ooc"]
    single = [r for r in scored if r["type"] == "single"]
    multi = [r for r in scored if r["type"] == "multi"]
    ooc = [r for r in scored if r["type"] == "ooc"]
    return {
        "n": len(results),
        # recall doesn't depend on the answer/judge -> over ALL rows
        "context_recall_single": mean([r["context_recall"] for r in results if r["type"] == "single"]),
        "context_recall_multi": mean([r["context_recall"] for r in results if r["type"] == "multi"]),
        "correctness_single": mean([r["score"] / 2 for r in single]),
        "correctness_multi": mean([r["score"] / 2 for r in multi]),
        "accuracy_answerable": mean([1.0 if r["score"] == 2 else 0.0 for r in ans]),
        "false_refusal_rate": mean([1.0 if r["refused"] else 0.0 for r in ans]),
        # PRECISION: of the answerable questions the system did answer, how
        # many answers were wrong (score 0)? A partial answer (score 1) is
        # correct-but-incomplete, not wrong. Failure analysis showed strict
        # accuracy (score 2 only) hides that most misses are incomplete,
        # not incorrect.
        "wrong_answer_rate": mean([1.0 if r["score"] == 0 else 0.0 for r in ans if not r["refused"]]),
        "correct_or_partial_rate": mean([1.0 if r["score"] >= 1 else 0.0 for r in ans]),
        "ooc_refusal_rate": mean([1.0 if r["refused"] else 0.0 for r in ooc]),
        "avg_retrieval_s": mean([r["retrieval_s"] for r in results]),
        "avg_answer_s": mean([r["answer_s"] for r in results]),
        "avg_context_chars": mean([r["context_chars"] for r in results]),
        "excluded": sum(1 for r in results if r.get("excluded")),
        "judge_errors": sum(1 for r in results if r.get("judge_error")),
        # compiled-answerer breakdown (absent for other answerers)
        **({f"mode_{m}": sum(1 for r in results if r.get("mode") == m) for m in ("compiled", "reader", "refusal")}
           if any("mode" in r for r in results) else {}),
        **({f"accuracy_{m}": mean([1.0 if r["score"] == 2 else 0.0 for r in ans if r.get("mode") == m])
            for m in ("compiled", "reader")} if any("mode" in r for r in results) else {}),
        "answer_errors": sum(1 for r in results if r.get("answer_error")),
    }
