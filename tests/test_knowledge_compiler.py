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

"""Knowledge Compiler: the precision guards (quote verification, answer
support), parsing, and the chat-time decision logic (compiled match vs
verified reader vs refusal). Models/embeddings are stubbed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import knowledge_compiler as kc  # noqa: E402
from src.core.state import EngineState  # noqa: E402

PASSAGE = ("Y José había ido a buscar comadronas. Mas, cuando estuvo de vuelta en la gruta, "
           "María había ya parido a su hijo. La comadrona Zelomi examinó a María. "
           "Y Salomé, al oír lo que había sucedido, no quiso creerlo hasta comprobarlo.")


@pytest.fixture
def state(tmp_path):
    cfg = type("Cfg", (), {"models_dir": tmp_path / "models", "data_dir": tmp_path / "data",
                           "default_model": "minilm"})()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    return EngineState(cfg)


def test_quote_verification_accepts_verbatim_ignoring_accents_case():
    assert kc.quote_in_text("cuando estuvo de vuelta en la gruta, MARIA habia ya parido", PASSAGE)


def test_quote_verification_tolerates_small_slips_rejects_paraphrase():
    assert kc.quote_in_text("La comadrona Zelomi examino a Maria", PASSAGE)
    assert not kc.quote_in_text("José fue a buscar parteras al pueblo vecino", PASSAGE)
    assert not kc.quote_in_text("gruta", PASSAGE)  # too short to prove anything


def test_answer_support_rejects_added_facts():
    assert kc.answer_supported("María ya había parido cuando José volvió a la gruta", PASSAGE)
    assert not kc.answer_supported("Zelomi era hermana de Salomé y vivía en Belén", PASSAGE)


def test_parse_items_keeps_only_complete_triples():
    raw = 'Here: [{"question": "q1", "answer": "a1", "quote": "x1"}, {"question": "q2", "answer": ""}]'
    assert kc._parse_items(raw) == [{"question": "q1", "answer": "a1", "quote": "x1"}]
    assert kc._parse_items("no json") == []


def test_parse_quoted_answer():
    assert kc.parse_quoted_answer(" «María había ya parido»\nRespuesta: Ya había dado a luz.") == \
        ("María había ya parido", "Ya había dado a luz.")
    assert kc.parse_quoted_answer("Quote: \"x y z\"\nAnswer: yes") == ("x y z", "yes")
    assert kc.parse_quoted_answer("solo texto") == (None, "solo texto")


def test_compile_verifies_and_drops_bad_items(state, monkeypatch):
    import src.core.rag as rag

    d = state.config.data_dir / "rag_index" / "corp" / "semantic"
    d.mkdir(parents=True)
    (d / "chunks.json").write_text(json.dumps([{"id": "c1", "source": "book", "text": PASSAGE}]), encoding="utf-8")
    generated = json.dumps([
        {"question": "¿Qué hizo Zelomi?", "answer": "Zelomi examinó a María.", "quote": "La comadrona Zelomi examinó a María."},
        {"question": "¿Dónde vivía Zelomi?", "answer": "En Belén.", "quote": "Zelomi vivía en Belén"},  # fake quote
        {"question": "¿Qué buscó José?", "answer": "José buscó reyes magos de Persia.", "quote": "José había ido a buscar comadronas"},  # unsupported
    ])
    monkeypatch.setattr(kc, "_generate", lambda st, g, p: generated)
    monkeypatch.setattr(rag, "_embed_texts", lambda texts: np.ones((len(texts), 4), dtype=np.float32))
    meta = kc.compile_corpus(state, "corp", workers=1, log=lambda m: None)
    assert (meta["kept"], meta["bad_quote"], meta["unsupported"]) == (1, 1, 1)
    items = json.loads((state.config.data_dir / "compiled" / "corp" / "items.json").read_text(encoding="utf-8"))
    assert items[0]["question"] == "¿Qué hizo Zelomi?" and items[0]["chunk_id"] == "c1"
    # resumable: a second run processes nothing new
    assert kc.compile_corpus(state, "corp", workers=1, log=lambda m: None)["generated"] == 0


def test_answer_returns_compiled_match_without_any_model(state, monkeypatch):
    monkeypatch.setattr(kc, "lookup", lambda st, kb, q, top_n=8: [
        {"question": "¿Qué hizo Zelomi?", "answer": "Examinó a María.", "quote": "Zelomi examinó a María",
         "source": "book", "chunk_id": "c1", "score": 7.5}])

    def no_model(*a, **k):
        raise AssertionError("no LLM call for a confident compiled match")

    import src.core.model_loader as ml
    monkeypatch.setattr(ml, "chat", no_model)
    res = kc.answer(state, "¿qué hizo la comadrona Zelomi?", "corp", "reader", threshold=3.0)
    assert res["mode"] == "compiled" and res["text"] == "Examinó a María." and res["quote"]


def test_reader_answer_with_invented_quote_is_refused(state, monkeypatch):
    import src.core.model_loader as ml
    import src.core.rag as rag

    monkeypatch.setattr(kc, "lookup", lambda st, kb, q, top_n=8: [{"score": -2.0}])
    monkeypatch.setattr(rag, "retrieve_grounded", lambda *a, **k: (
        [{"corpus": "corp", "source": "book", "id": "c1", "score": 1.0, "text": PASSAGE}], {}))
    monkeypatch.setattr(ml, "chat", lambda *a, **k: {"text": " «Zelomi era de Belén y tenía tres hijos»\nRespuesta: Era de Belén."})
    strict = kc.answer(state, "¿De dónde era Zelomi?", "corp", "reader", threshold=3.0, verify="strict", reader_style="quote")
    assert strict["mode"] == "refusal" and "no contiene" in strict["text"]
    lenient = kc.answer(state, "¿De dónde era Zelomi?", "corp", "reader", threshold=3.0, verify="lenient", reader_style="quote")
    assert lenient["mode"] == "reader" and lenient["verified"] is False


def test_reader_answer_with_real_quote_is_returned(state, monkeypatch):
    import src.core.model_loader as ml
    import src.core.rag as rag

    monkeypatch.setattr(kc, "lookup", lambda st, kb, q, top_n=8: [])
    monkeypatch.setattr(rag, "retrieve_grounded", lambda *a, **k: (
        [{"corpus": "corp", "source": "book", "id": "c1", "score": 1.0, "text": PASSAGE}], {}))
    monkeypatch.setattr(ml, "chat", lambda *a, **k: {"text": " «La comadrona Zelomi examinó a María»\nRespuesta: La examinó Zelomi."})
    res = kc.answer(state, "¿Quién examinó a María?", "corp", "reader", reader_style="quote")
    assert res["mode"] == "reader" and res["verified"] and res["text"] == "La examinó Zelomi."


def test_native_reader_answer_verified_by_content_support(state, monkeypatch):
    import src.core.model_loader as ml
    import src.core.rag as rag

    monkeypatch.setattr(kc, "lookup", lambda st, kb, q, top_n=8: [])
    monkeypatch.setattr(rag, "retrieve_grounded", lambda *a, **k: (
        [{"corpus": "corp", "source": "book", "id": "c1", "score": 1.0, "text": PASSAGE}], {}))
    monkeypatch.setattr(ml, "chat", lambda *a, **k: {"text": "La comadrona Zelomi examinó a María."})
    assert kc.answer(state, "¿Quién examinó a María?", "corp", "reader")["mode"] == "reader"
    monkeypatch.setattr(ml, "chat", lambda *a, **k: {"text": "Fue Herodes junto con tres soldados romanos."})
    assert kc.answer(state, "¿Quién examinó a María?", "corp", "reader")["mode"] == "refusal"


def test_facts_strategy_reads_compiled_facts_and_answers_directly_only_near_duplicates(state, monkeypatch):
    import src.core.model_loader as ml

    fact = {"question": "¿Qué hizo Zelomi?", "answer": "Examinó a María.", "quote": "La comadrona Zelomi examinó a María",
            "source": "book", "chunk_id": "c1", "corpus": "corp"}
    seen = {}

    def reader(state, prompt, **k):
        seen["prompt"] = prompt
        return {"text": "Zelomi examinó a María."}

    monkeypatch.setattr(ml, "chat", reader)
    monkeypatch.setattr(kc, "lookup", lambda st, kb, q, top_n=8: [{**fact, "score": 4.0}])
    res = kc.answer(state, "¿Qué hizo la partera?", "corp", "reader", strategy="facts")
    assert res["mode"] == "reader" and res["verified"]
    assert "Examinó a María." in seen["prompt"] and "«La comadrona Zelomi examinó a María»" in seen["prompt"]
    # near-duplicate question -> stored answer, no reader call
    monkeypatch.setattr(kc, "lookup", lambda st, kb, q, top_n=8: [{**fact, "score": 9.5}])
    monkeypatch.setattr(ml, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no reader")))
    assert kc.answer(state, "¿Qué hizo Zelomi?", "corp", "reader", strategy="facts")["mode"] == "compiled"
