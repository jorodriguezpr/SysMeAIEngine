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

"""Regression tests for bugs found by live-verifying the grounding layer
and knowledge graph (see docs/GROUNDING.md "Verification"). Every test here
reproduces a failure that was actually observed against the real
aprocrifos corpora / ollama_cloud provider before it was fixed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import cloud_chat  # noqa: E402
from src.core import kgraph as kg  # noqa: E402
from src.core import rag as rag_mod  # noqa: E402
from src.core.state import EngineState  # noqa: E402

CLOUD_ON = {"enabled": True, "api_key": "k", "model": "m", "provider": "ollama_cloud"}


@pytest.fixture
def state(tmp_path):
    cfg = type("Cfg", (), {
        "models_dir": tmp_path / "models", "data_dir": tmp_path / "data", "default_model": "minilm",
    })()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    return EngineState(cfg)


# ── cloud_chat: transport errors + null content ─────────────────────────────

def test_network_error_becomes_runtime_error(monkeypatch):
    def boom(*a, **kw):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(cloud_chat.requests, "post", boom)
    with pytest.raises(RuntimeError, match="request failed"):
        cloud_chat.cloud_generate(CLOUD_ON, "hi")


def test_reasoning_model_null_content_is_empty_string(monkeypatch):
    class R:
        ok = True

        def json(self):
            return {"choices": [{"message": {"content": None, "reasoning": "thinking..."}}]}

    monkeypatch.setattr(cloud_chat.requests, "post", lambda *a, **kw: R())
    assert cloud_chat.cloud_generate(CLOUD_ON, "hi") == ""


def test_empty_answer_retried_once_with_larger_budget(monkeypatch):
    budgets = []

    def fake_dispatch(settings, prompt, max_tokens):
        budgets.append(max_tokens)
        return "" if len(budgets) == 1 else "real answer"

    monkeypatch.setattr(cloud_chat, "_dispatch", fake_dispatch)
    assert cloud_chat.cloud_generate({**CLOUD_ON, "max_tokens": 1024}, "q") == "real answer"
    assert budgets == [None, cloud_chat.EMPTY_RETRY_MAX_TOKENS]


def test_non_empty_answer_is_not_retried(monkeypatch):
    budgets = []
    monkeypatch.setattr(cloud_chat, "_dispatch", lambda s, p, m: budgets.append(m) or "ok")
    assert cloud_chat.cloud_generate(CLOUD_ON, "q") == "ok" and len(budgets) == 1


# ── expand_query ────────────────────────────────────────────────────────────

def test_expand_query_budget_and_numbering(state, monkeypatch):
    import src.core.cloud_settings as cs

    monkeypatch.setattr(cs, "get_cloud_settings", lambda st: CLOUD_ON)
    seen = {}

    def fake(settings, prompt, max_tokens=None):
        seen["max_tokens"] = max_tokens
        return "1. ¿Quién administró el bautismo?\n2) ¿Quién bautizó a Cristo?"

    monkeypatch.setattr(cloud_chat, "cloud_generate", fake)
    out = rag_mod.expand_query(state, "¿Quién bautizó a Jesús?")
    # 200 returned empty content from glm-5.3-flash 2/2 times (all budget
    # spent reasoning); 600+ worked every time
    # no small explicit cap -> the 10000-token settings default applies
    assert seen["max_tokens"] is None or seen["max_tokens"] >= 600
    assert out[1:] == ["¿Quién administró el bautismo?", "¿Quién bautizó a Cristo?"]


def test_retrieve_grounded_allow_cloud_false_never_expands(state, monkeypatch):
    called = []
    monkeypatch.setattr(rag_mod, "expand_query", lambda *a, **kw: called.append(1) or ["q", "v"])
    monkeypatch.setattr(rag_mod, "retrieve", lambda *a, **kw: [])
    _, dbg = rag_mod.retrieve_grounded(state, "q", allow_cloud=False, parent_child=False)
    assert called == [] and dbg["variants"] == 1


# ── MMR ─────────────────────────────────────────────────────────────────────

def test_mmr_skips_cross_corpus_copies_of_the_same_passage():
    passage = "Pilatos le dijo a Jesús: ¿Eres tú el rey de los judíos? Y Jesús respondió."
    hits = [
        {"score": 8.0, "text": passage, "corpus": "a"},
        {"score": 7.9, "text": passage, "corpus": "b"},  # same book, other corpus
        {"score": 7.8, "text": passage + " ", "corpus": "c"},
        {"score": 3.0, "text": "María Magdalena fue al sepulcro al amanecer del primer día.", "corpus": "a"},
        {"score": 2.0, "text": "Los soldados echaron suertes sobre sus vestiduras en el Gólgota.", "corpus": "a"},
    ]
    out = rag_mod._mmr_diversify(hits, top_k=3)
    assert out[0]["corpus"] == "a"
    assert sum(1 for h in out if h["text"].strip() == passage) == 1
    assert len(out) == 3


def test_mmr_stops_instead_of_padding_with_duplicates():
    hits = [{"score": s, "text": "exactly the same passage text here"} for s in (3.0, 2.0, 1.0)]
    assert len(rag_mod._mmr_diversify(hits, top_k=2)) == 1


def test_mmr_works_on_negative_rerank_scores():
    hits = [{"score": -0.5, "text": "alpha beta gamma delta"},
            {"score": -3.0, "text": "epsilon zeta eta theta"},
            {"score": -4.7, "text": "iota kappa lambda mu"}]
    assert len(rag_mod._mmr_diversify(hits, top_k=2)) == 2


def test_grounded_candidates_single_pool_dedupes_same_chunk_across_corpora(state, monkeypatch):
    # same content id in two corpora (built from the same PDFs) -> kept once
    def fake_one(st, q, k, method, name):
        return [{"id": "same", "corpus": name, "source": "s", "text": "t", "score": 1.0},
                {"id": f"{name}-{method}", "corpus": name, "source": "s", "text": "u", "score": 0.5}]

    monkeypatch.setattr(rag_mod, "_retrieve_one", fake_one)
    pool = rag_mod._grounded_candidates(state, ["q1", "q2"], ["a", "b"], pool_size=30)
    ids = [h["id"] for h in pool]
    assert ids.count("same") == 1
    assert set(ids) == {"same", "a-semantic", "a-tfidf", "b-semantic", "b-tfidf"}
    assert len(rag_mod._grounded_candidates(state, ["q"], ["a", "b"], pool_size=2)) == 2


def test_grounded_candidates_no_index_raises(state):
    with pytest.raises(FileNotFoundError):
        rag_mod._grounded_candidates(state, ["q"], "all")


# ── expand_context ──────────────────────────────────────────────────────────

def _write_index(state, corpus, n):
    d = state.config.data_dir / "rag_index" / corpus / "semantic"
    d.mkdir(parents=True, exist_ok=True)
    chunks = [{"id": f"ch{i}", "source": "book", "text": f"Paragraph {i}."} for i in range(n)]
    (d / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    return chunks


def test_expand_context_keeps_document_order(state):
    chunks = _write_index(state, "c", 3)
    out = rag_mod.expand_context(
        state, [{"id": "ch1", "corpus": "c", "source": "book", "score": 1, "text": chunks[1]["text"]}]
    )
    assert out[0]["text"] == "Paragraph 0.\nParagraph 1.\nParagraph 2."


def test_expand_context_never_repeats_a_chunk_across_hits(state):
    chunks = _write_index(state, "c", 6)
    hits = [
        {"id": f"ch{i}", "corpus": "c", "source": "book", "score": 1, "text": chunks[i]["text"]}
        for i in (2, 3, 1)
    ]
    out = rag_mod.expand_context(state, hits)
    all_paras = [p for h in out for p in h["text"].split("\n")]
    assert len(all_paras) == len(set(all_paras))
    # ch3 and ch1 were already inside ch2's window -> dropped, not re-sent
    assert len(out) == 1


# ── knowledge graph ─────────────────────────────────────────────────────────

def _kg_index(state, corpus="kc", n=3):
    d = state.config.data_dir / "rag_index" / corpus / "semantic"
    d.mkdir(parents=True, exist_ok=True)
    (d / "chunks.json").write_text(json.dumps(
        [{"id": f"k{i}", "source": "s", "text": f"Texto número {i} sobre Pedro y Juan."} for i in range(n)]
    ), encoding="utf-8")


def _kg_stub(monkeypatch, fn):
    import src.core.cloud_distill as cd
    import src.core.summarize as sm

    monkeypatch.setattr(cd, "ensure_cloud_ready", lambda st: None)
    monkeypatch.setattr(sm, "_detect_language", lambda t: "Spanish")
    monkeypatch.setattr(kg, "extract_triples_for_chunk", fn)


def test_kgraph_rebuild_skips_chunks_that_yielded_zero_triples(state, monkeypatch):
    _kg_index(state)
    calls = []

    def fake(st, text, language, n_max=8):
        calls.append(text)
        return [] if "0" in text else [("Pedro", "habla_con", "Juan")]

    _kg_stub(monkeypatch, fake)
    kg.build_graph(state, "kc")
    assert len(calls) == 3
    kg.build_graph(state, "kc")
    assert len(calls) == 3  # previously re-sent the zero-triple chunk every rebuild


def test_kgraph_network_error_skips_chunk_not_whole_build(state, monkeypatch):
    _kg_index(state)

    def fake(st, text, language, n_max=8):
        if "1" in text:
            raise RuntimeError("cloud provider request failed: ConnectionError")
        return [("Pedro", "habla_con", "Juan")]

    _kg_stub(monkeypatch, fake)
    res = kg.build_graph(state, "kc")
    assert res["chunks_processed"] == 3 and res["chunks_failed"] == 1
    # the failed chunk is retried on the next build, the others aren't
    _kg_stub(monkeypatch, lambda st, text, language, n_max=8: [("Pedro", "x", "Juan")])
    assert kg.build_graph(state, "kc")["chunks_processed"] == 1


def test_kgraph_empty_extraction_output_is_a_failure_not_zero_triples(state, monkeypatch):
    import src.core.cloud_settings as cs

    monkeypatch.setattr(cs, "get_cloud_settings", lambda st: CLOUD_ON)
    monkeypatch.setattr(cloud_chat, "cloud_generate", lambda *a, **kw: "")
    with pytest.raises(RuntimeError, match="empty output"):
        kg.extract_triples_for_chunk(state, "texto", "Spanish")


def test_default_cloud_budget_is_10000():
    from src.core.cloud_settings import DEFAULTS

    assert DEFAULTS["max_tokens"] == 10000
    assert cloud_chat.EMPTY_RETRY_MAX_TOKENS == 10000


def test_kgraph_norm_merges_underscore_variants():
    assert kg._norm("rey_del_Norte") == kg._norm("rey del Norte")


@pytest.mark.parametrize("bad", ["..\\sentinel", "../x", "a/b", "", ".hidden", "x..y"])
def test_kgraph_rejects_path_like_corpus_names(state, bad):
    with pytest.raises(ValueError):
        kg._db_path(state, bad)


def test_kgraph_accepts_real_corpus_names(state):
    for ok in ("aprocrifos-v2", "aprocrifos-cloudqa-1.5b-v1", "my_corpus"):
        assert kg._db_path(state, ok).name == f"{ok}.db"


def test_grounding_settings_reject_out_of_range_top_k(state):
    from src.core.grounding_settings import save_grounding_settings

    for bad in (0, -5, 21):
        with pytest.raises(ValueError):
            save_grounding_settings(state, chat_top_k=bad)
    assert save_grounding_settings(state, chat_top_k=4)["chat_top_k"] == 4


def test_local_answer_regenerates_on_chinese_drift_then_cuts(state, monkeypatch):
    import src.core.model_loader as ml

    replies = iter(["Salomé reaccionó con desconfianza y pidió验证我的理解", "Salomé no le creyó a la partera."])
    prompts = []

    def fake(st, prompt, **k):
        prompts.append(prompt)
        return {"text": next(replies)}

    monkeypatch.setattr(ml, "chat", fake)
    assert rag_mod.local_answer(state, "P", "m", "Spanish") == "Salomé no le creyó a la partera."
    assert "responde únicamente en español" in prompts[1]
    # still drifting after the retry -> cut before the first CJK character
    monkeypatch.setattr(ml, "chat", lambda st, p, **k: {"text": "Salomé dudó y pidió验证"})
    assert rag_mod.local_answer(state, "P", "m", "Spanish") == "Salomé dudó y pidió"


def test_complete_style_asks_for_complete_answer():
    hits = [{"corpus": "c", "source": "s", "text": "Pedro habló con Juan en el templo."}]
    assert "COMPLETA" in rag_mod.build_prompt("¿Qué hizo Pedro?", hits, "complete")
    assert "breve y precisa" in rag_mod.build_prompt("¿Qué hizo Pedro?", hits, "native")
