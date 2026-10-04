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

"""Tests for src.core.rag's evaluate_retrieval / evaluate_answers -- the
RAG-QA evaluation loop. retrieve() and model_loader.chat() are mocked since
they need a real built index / real model; a real live run against an
existing corpus was done manually during development (see memory)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import rag as rag_mod  # noqa: E402


def _fake_state(tmp_path):
    return SimpleNamespace(config=SimpleNamespace(data_dir=tmp_path))


def _write_chunks(tmp_path, method, corpus, chunks):
    d = tmp_path / "rag_index" / corpus / method
    d.mkdir(parents=True, exist_ok=True)
    (d / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")


def test_evaluate_retrieval_missing_index_raises(tmp_path):
    state = _fake_state(tmp_path)
    try:
        rag_mod.evaluate_retrieval(state, "nope")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_evaluate_retrieval_all_hits(tmp_path):
    state = _fake_state(tmp_path)
    chunks = [
        {"id": "c1", "text": "The sky is blue. It reflects sunlight."},
        {"id": "c2", "text": "Water boils at 100C. This is a physical fact."},
    ]
    _write_chunks(tmp_path, "semantic", "mycorp", chunks)

    def fake_retrieve(state, question, top_k=5, method="hybrid", corpus="default"):
        # Always return both chunks -- simulates a perfect index.
        return [{"id": "c1"}, {"id": "c2"}]

    with patch.object(rag_mod, "retrieve", side_effect=fake_retrieve):
        result = rag_mod.evaluate_retrieval(state, "mycorp", method="hybrid", held_out_fraction=1.0)

    assert result["corpus"] == "mycorp"
    assert result["n_evaluated"] == 2
    assert result["recall_at_k"] == 1.0
    assert all(d["found_in_top_k"] for d in result["details"])


def test_evaluate_retrieval_no_hits(tmp_path):
    state = _fake_state(tmp_path)
    chunks = [{"id": "c1", "text": "Some content here. More text."}]
    _write_chunks(tmp_path, "semantic", "mycorp", chunks)

    with patch.object(rag_mod, "retrieve", return_value=[{"id": "other-id"}]):
        result = rag_mod.evaluate_retrieval(state, "mycorp", held_out_fraction=1.0)

    assert result["recall_at_k"] == 0.0
    assert result["details"][0]["found_in_top_k"] is False


def test_evaluate_retrieval_tfidf_method_reads_tfidf_meta(tmp_path):
    state = _fake_state(tmp_path)
    chunks = [{"id": "c1", "text": "Alpha beta gamma. Delta epsilon."}]
    _write_chunks(tmp_path, "tfidf", "mycorp", chunks)

    with patch.object(rag_mod, "retrieve", return_value=[{"id": "c1"}]) as mock_retrieve:
        result = rag_mod.evaluate_retrieval(state, "mycorp", method="tfidf", held_out_fraction=1.0)

    assert result["recall_at_k"] == 1.0
    assert mock_retrieve.call_args.kwargs["method"] == "tfidf"


def test_evaluate_answers_keyword_overlap(tmp_path):
    state = _fake_state(tmp_path)
    chunks = [{"id": "c1", "text": "Intro sentence. Rockets require significant thrust to escape gravity."}]
    _write_chunks(tmp_path, "semantic", "mycorp", chunks)

    def fake_chat(state, prompt, model_name=None, max_new_tokens=128, temperature=0.0):
        return {"text": "Rockets need thrust to escape gravity."}

    with patch.object(rag_mod, "retrieve", return_value=[]), \
         patch("src.core.model_loader.chat", side_effect=fake_chat):
        result = rag_mod.evaluate_answers(state, "mycorp", "some-model", held_out_fraction=1.0)

    assert result["model"] == "some-model"
    assert result["n_evaluated"] == 1
    assert result["avg_keyword_overlap"] > 0.5
    assert "generated" in result["details"][0]


def test_evaluate_retrieval_reports_rank_hit_at_1_and_mrr(tmp_path):
    state = _fake_state(tmp_path)
    chunks = [
        {"id": "c1", "text": "Alpha sentence. More alpha text."},
        {"id": "c2", "text": "Beta sentence. More beta text."},
    ]
    _write_chunks(tmp_path, "semantic", "mycorp", chunks)

    def fake_retrieve(state, question, top_k=5, method="hybrid", corpus="default"):
        # c1 always ranked first, c2 always second
        return [{"id": "c1"}, {"id": "c2"}]

    with patch.object(rag_mod, "retrieve", side_effect=fake_retrieve):
        r = rag_mod.evaluate_retrieval(state, "mycorp", held_out_fraction=1.0)
    ranks = {d["expected_id"]: d["rank"] for d in r["details"]}
    assert ranks == {"c1": 1, "c2": 2}
    assert r["recall_at_k"] == 1.0
    assert r["hit_at_1"] == 0.5
    assert r["mrr"] == 0.75


def test_keywords_mode_never_copies_a_sentence_verbatim(tmp_path):
    text = ("Photosynthesis converts sunlight into chemical energy inside chloroplasts. "
            "Plants release oxygen as a byproduct of this remarkable process.")
    qa = rag_mod.eval_questions([{"id": "c1", "text": text}], "keywords")
    q = qa[0]["question"]
    words = q.split()
    assert len(words) == 6
    assert all(len(w) >= 5 and w == w.lower() for w in words)
    assert "Photosynthesis converts sunlight" not in q
    # deterministic per chunk id
    assert rag_mod.eval_questions([{"id": "c1", "text": text}], "keywords")[0]["question"] == q


def test_keyword_regex_keeps_accented_words_whole():
    q = rag_mod._keyword_question("La preparación de esta edición", "k")
    assert "preparación" in q.split() and "edición" in q.split()


def test_eval_questions_rejects_unknown_mode():
    try:
        rag_mod.eval_questions([{"id": "c", "text": "x."}], "nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_evaluate_answers_unicode_overlap_and_refusal_rate(tmp_path):
    state = _fake_state(tmp_path)
    chunks = [
        {"id": "c1", "text": "Primera frase. La preparación de esta edición."},
        {"id": "c2", "text": "Otra frase. Algo totalmente distinto aquí."},
    ]
    _write_chunks(tmp_path, "semantic", "mycorp", chunks)
    answers = {
        "c1": "La preparación de esta edición.",
        "c2": "The provided context does not contain this information.",
    }

    def fake_chat(state, prompt, **kw):
        cid = "c1" if "Primera" in prompt else "c2"
        return {"text": answers[cid]}

    with patch.object(rag_mod, "retrieve", return_value=[]), \
         patch("src.core.model_loader.chat", side_effect=fake_chat):
        r = rag_mod.evaluate_answers(state, "mycorp", "m", held_out_fraction=1.0)
    by_q = {d["expected_answer"]: d for d in r["details"]}
    assert by_q["La preparación de esta edición."]["keyword_overlap"] == 1.0
    assert r["refusal_rate"] == 0.5
