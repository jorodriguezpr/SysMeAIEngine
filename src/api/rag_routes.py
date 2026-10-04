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

"""RAG endpoints: query the TF-IDF index directly, or chat with retrieved
passages injected into the prompt so answers are grounded in the actual
ingested documents instead of only what got baked into model weights."""
from __future__ import annotations

from typing import Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..core import model_loader
from ..core import rag as rag_mod
from ..core.model_loader import chat as core_chat
from ..core.state import EngineState
from .chat_routes import get_state, require_auth

router = APIRouter()


class RagQueryIn(BaseModel):
    query: str
    top_k: int = 5
    method: str = rag_mod.DEFAULT_METHOD
    corpus: Union[str, list[str]] = "default"  # a name, a list of names, or "all"


@router.post("/api/rag/query")
def rag_query(
    body: RagQueryIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    try:
        return {"hits": rag_mod.retrieve(state, body.query, body.top_k, method=body.method, corpus=body.corpus)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


class RagChatIn(BaseModel):
    model: Optional[str] = None
    prompt: str
    # 5, not 3: raised alongside the paragraph-aware RAG indexing fix -- more
    # (now cleaner, sentence-boundary-respecting) passages reach the model
    # per question instead of leaving corpus content unused
    top_k: int = 5
    max_new_tokens: int = 256
    # low by design: RAG-mode answers should stick close to the retrieved
    # passages rather than embellish — lower temperature means less creative
    # drift away from what's actually in the context
    temperature: float = 0.2
    top_p: float = 0.9
    # named to avoid colliding with the retrieval `top_k` above; this is the
    # generation-time nucleus/top-k sampling limit, not a passage count
    gen_top_k: int = 50
    method: str = rag_mod.DEFAULT_METHOD
    # None -> the loaded model's own paired corpus; a name; a list of names; or "all"
    # to search every corpus that has a built index and merge results by score.
    corpus: Optional[Union[str, list[str]]] = None
    # run a cheap second pass that asks the model to fix spelling/grammar in
    # its own answer without changing meaning. Default False: live-tested
    # against aprocrifos-v2 (494M) and it made output *worse*, not better --
    # the same small/undertrained model proofreading its own Spanish
    # introduced new errors instead of fixing the persistent ones (e.g.
    # "Nicodemo" still came out "Nicodmo"/"Nicodamo" either way). Worth
    # opting into only with a meaningfully stronger model doing the polish.
    polish: bool = False
    # None -> fall back to the saved Settings-page default; explicit
    # true/false overrides it for this one request. Local retrieval above
    # still always runs -- this only decides who writes the final answer
    # from the retrieved-passages prompt: the local model or the cloud one.
    use_cloud: Optional[bool] = None
    # enable the P1 retrieval upgrades (multi-query expansion + parent-child
    # context widening + MMR diversity) for this request. Default True:
    # this endpoint exists to ground answers, and the upgrades measurably
    # widen what retrieval finds for differently-phrased questions.
    grounded: bool = True
    # also search the knowledge graph (when a graph has been built) and
    # merge entity/relation facts into the context
    use_kgraph: bool = True


@router.post("/api/rag/chat")
def rag_chat(
    body: RagChatIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    import time

    from ..core.calculator import try_calculate
    from ..core.summarize import detect_summarize_request, summarize_document
    from ..core.translate import detect_translate_request
    from ..core.translate import translate as run_translate

    calc = try_calculate(body.prompt)
    if calc is not None:
        return {
            "model": "calculator", "prompt_tokens": 0, "completion_tokens": 0,
            "elapsed_ms": 0.0, "text": calc, "sources": [],
        }
    doc_path = detect_summarize_request(body.prompt)
    if doc_path is not None:
        t0 = time.perf_counter()
        try:
            result = summarize_document(state, doc_path, model_name=body.model)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "model": result["model"], "prompt_tokens": 0, "completion_tokens": 0,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "text": result["summary"], "raw_text": result["summary"], "sources": [],
        }
    translate_req = detect_translate_request(body.prompt)
    if translate_req is not None:
        text, direction = translate_req
        t0 = time.perf_counter()
        try:
            translation = run_translate(text, direction)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "model": f"opus-mt-{direction}", "prompt_tokens": 0, "completion_tokens": 0,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "text": translation, "raw_text": translation, "sources": [],
        }
    model_name = body.model or state.config.default_model
    corpus = body.corpus
    if corpus is None:
        entry = model_loader.get_model_entry(state, model_name)
        corpus = (entry.get("rag_corpus") if entry else None) or model_name

    from ..core.grounding_settings import get_grounding_settings

    from ..core.cloud_chat import cloud_generate
    from ..core.cloud_settings import get_cloud_settings

    gs = get_grounding_settings(state)
    cloud_settings = get_cloud_settings(state)
    use_cloud = body.use_cloud if body.use_cloud is not None else cloud_settings["enabled"]
    dbg = None
    try:
        if body.grounded:
            hits, dbg = rag_mod.retrieve_grounded(
                state, body.prompt, top_k=body.top_k, corpus=corpus,
                multi_query=bool(gs.get("chat_multi_query", True)),
                # widening only for the cloud answerer -- it measurably hurts
                # the small local model (see chat_routes._grounded_chat)
                parent_child=bool(use_cloud),
                kgraph_enabled=body.use_kgraph and bool(gs.get("chat_kgraph", True)),
                # use_cloud=false must keep the question local: skip the
                # cloud multi-query rewrite too, not just the answer call
                allow_cloud=use_cloud,
                graph_slot=bool(gs.get("chat_graph_slot", False)),
            )
        else:
            hits = rag_mod.retrieve(state, body.prompt, body.top_k, method=body.method, corpus=corpus)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not hits:
        hits = []
    augmented = rag_mod.build_prompt(body.prompt, hits)

    if use_cloud:
        t0 = time.perf_counter()
        try:
            cloud_text = cloud_generate(cloud_settings, augmented)
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        out = {
            "model": f"cloud:{cloud_settings['provider']}:{cloud_settings['model']}",
            "prompt_tokens": len(augmented) // 4 + 1, "completion_tokens": len(cloud_text) // 4 + 1,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1), "text": cloud_text,
        }
    elif body.grounded:
        # same measured offline recipe as /api/chat grounded mode (see
        # chat_routes._grounded_chat): question-language "complete" prompt,
        # temperature 0, script-drift guard
        t0 = time.perf_counter()
        style = gs.get("local_prompt_style") or "complete"
        local_prompt = rag_mod.build_prompt(body.prompt, hits, style)
        lang = rag_mod.question_language(body.prompt, hits)
        model_name = body.model or state.config.default_model
        text = rag_mod.local_answer(state, local_prompt, model_name, lang,
                                    max_new_tokens=max(body.max_new_tokens, 350))
        if rag_mod.is_refusal(text):
            text = rag_mod.REFUSALS.get(lang, rag_mod.REFUSALS["English"])
        out = {"model": model_name, "prompt_tokens": len(local_prompt) // 4 + 1,
               "completion_tokens": len(text) // 4 + 1,
               "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1), "text": text}
    else:
        out = core_chat(
            state,
            augmented,
            model_name=body.model,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.gen_top_k,
        )
    raw_text = out["text"]
    final_text = raw_text
    if body.polish and raw_text.strip():
        polished = _polish_answer(state, raw_text, model_name=body.model)
        if polished:
            final_text = polished
    return {
        **out,
        "text": final_text,
        "raw_text": raw_text,
        "sources": [
            {"corpus": h["corpus"], "source": h["source"], "score": h["score"]} for h in hits
        ],
        "grounding": dbg,
    }


def _polish_answer(state: EngineState, text: str, model_name: Optional[str]) -> Optional[str]:
    """Second, cheap generation pass that asks the model to fix its own
    spelling/grammar without changing meaning or adding content. Best-effort:
    on any failure, or if the model seems to have gone off the rails (wildly
    different length), fall back to the unpolished text rather than risk
    handing back a worse or fabricated answer."""
    prompt = (
        "Corrige unicamente la ortografia y gramatica del siguiente texto, "
        "sin cambiar su significado y sin anadir informacion nueva. "
        "Responde solo con el texto corregido, nada mas.\n\n"
        f"Texto:\n{text}"
    )
    try:
        result = core_chat(
            state,
            prompt,
            model_name=model_name,
            max_new_tokens=min(512, max(64, len(text) // 3 + 32)),
            temperature=0.1,
            top_p=0.9,
            top_k=50,
        )
    except Exception:
        return None
    corrected = result["text"].strip()
    if not corrected:
        return None
    # a correction pass that balloons or guts the answer likely drifted into
    # rambling or refusing rather than actually correcting -- discard it
    if len(corrected) > len(text) * 2 or len(corrected) < len(text) * 0.4:
        return None
    return corrected


# ── RAG-QA evaluation ─────────────────────────────────────────────────────
# Runs in a background thread: retrieval-only eval takes ~45s on a real
# corpus (mostly embedding/reranker model load), answer eval minutes (one
# real generation per held-out question) -- far too long for one request.
# In-memory like state.train_jobs: lost on server restart, which is fine
# for a result the caller polls for right away.
_eval_jobs: dict[str, dict] = {}


class RagEvalIn(BaseModel):
    corpus: str
    method: str = rag_mod.DEFAULT_METHOD
    top_k: int = 5
    held_out_fraction: float = 0.1
    question_mode: str = "first_sentence"  # | "keywords" (exact-word query, see rag.eval_questions)
    model: Optional[str] = None  # also score generated answers with this model
    max_new_tokens: int = 128
    seed: int = 42


@router.get("/api/rag/corpora")
def rag_corpora(state: EngineState = Depends(get_state), _: None = Depends(require_auth)):
    return {"corpora": rag_mod.list_corpora(state, method="semantic")}


@router.post("/api/rag/eval")
def rag_eval_start(
    body: RagEvalIn,
    state: EngineState = Depends(get_state),
    _: None = Depends(require_auth),
):
    import threading
    import time
    import uuid

    if body.question_mode not in rag_mod.QUESTION_MODES:
        raise HTTPException(status_code=400, detail=f"question_mode must be one of {rag_mod.QUESTION_MODES}")
    if not 0 < body.held_out_fraction <= 1:
        raise HTTPException(status_code=400, detail="held_out_fraction must be in (0, 1]")
    if body.corpus not in rag_mod.list_corpora(state, method="tfidf" if body.method == "tfidf" else "semantic"):
        raise HTTPException(status_code=404, detail=f"no RAG index built for corpus '{body.corpus}'")

    eval_id = uuid.uuid4().hex[:12]
    job = {
        "id": eval_id, "status": "running", "params": body.model_dump(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "finished_at": None,
        "result": None, "error": None,
    }
    _eval_jobs[eval_id] = job

    def _run() -> None:
        try:
            result = {"retrieval": rag_mod.evaluate_retrieval(
                state, body.corpus, method=body.method, held_out_fraction=body.held_out_fraction,
                top_k=body.top_k, seed=body.seed, question_mode=body.question_mode,
            )}
            if body.model:
                result["answers"] = rag_mod.evaluate_answers(
                    state, body.corpus, body.model, method=body.method,
                    held_out_fraction=body.held_out_fraction, top_k=min(body.top_k, 3),
                    max_new_tokens=body.max_new_tokens, seed=body.seed,
                    question_mode=body.question_mode,
                )
            job["result"] = result
            job["status"] = "done"
        except Exception as e:  # surfaced to the poller, not swallowed
            job["error"] = f"{type(e).__name__}: {e}"
            job["status"] = "failed"
        job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    threading.Thread(target=_run, daemon=True).start()
    return {"eval_id": eval_id, "status": "running"}


@router.get("/api/rag/eval/{eval_id}")
def rag_eval_status(eval_id: str, _: None = Depends(require_auth)):
    job = _eval_jobs.get(eval_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown eval id (evals are kept in memory until server restart)")
    return job
