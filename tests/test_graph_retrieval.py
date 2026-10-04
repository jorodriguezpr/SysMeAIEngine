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

"""Graph-walk retrieval: a question sharing NO words with the answering
passage must still reach it through the knowledge graph (role -> entity ->
passage), and retrieve_grounded(graph_slot=True) must reserve one slot for
it. Embedder and cross-encoder are stubbed; the graph math is real."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import graph_retrieval as gr  # noqa: E402
from src.core import rag  # noqa: E402
from src.core.state import EngineState  # noqa: E402

CHUNKS = [
    {"id": "p0", "source": "nicodemo", "text": "Pilatos mandó llamar a Jesús y le preguntó si era rey de los judíos."},
    {"id": "p1", "source": "nicodemo", "text": "Los ancianos del pueblo se reunieron en el templo al amanecer."},
    {"id": "p2", "source": "pseudo", "text": "Herodes ordenó matar a los niños de Bethlehem menores de dos años."},
    {"id": "p3", "source": "pseudo", "text": "José tomó al niño y a su madre y huyeron a Egipto de noche."},
]
TRIPLES = [  # (subject, relation, object, chunk)
    ("Pilatos", "es", "gobernador romano", "p0"),
    ("Pilatos", "interrogó_a", "Jesús", "p0"),
    ("Herodes", "ordenó_matar", "niños de Bethlehem", "p2"),
    ("José", "huyó_a", "Egipto", "p3"),
]


@pytest.fixture
def state(tmp_path, monkeypatch):
    cfg = type("Cfg", (), {"models_dir": tmp_path / "models", "data_dir": tmp_path / "data",
                           "default_model": "minilm"})()
    d = cfg.data_dir / "rag_index" / "corp" / "semantic"
    d.mkdir(parents=True)
    (d / "chunks.json").write_text(json.dumps(CHUNKS), encoding="utf-8")
    kg = cfg.data_dir / "kgraph"
    kg.mkdir(parents=True)
    con = sqlite3.connect(str(kg / "corp.db"))
    con.executescript("CREATE TABLE entities(id INTEGER PRIMARY KEY, name TEXT, norm TEXT UNIQUE);"
                      "CREATE TABLE triples(id INTEGER PRIMARY KEY, subject_id INT, relation TEXT, object_id INT,"
                      " chunk_id TEXT, source TEXT, created_at REAL);")
    ids = {}
    for s, r, o, c in TRIPLES:
        for name in (s, o):
            if name not in ids:
                ids[name] = con.execute("INSERT INTO entities(name, norm) VALUES (?,?)", (name, name.lower())).lastrowid
        con.execute("INSERT INTO triples(subject_id, relation, object_id, chunk_id, source, created_at) VALUES (?,?,?,?,?,0)",
                    (ids[s], r, ids[o], c, "x"))
    con.commit(); con.close()

    # Stub embedder: a bag of hand-picked "concepts" so similarity is
    # controlled -- the question about the Roman governor is close ONLY to
    # the descriptor of entity Pilatos ("... es gobernador romano ..."), and
    # shares no word with passage p0.
    concepts = ["gobernador", "herodes", "egipto", "templo"]

    def fake_embed(texts, **k):
        out = np.array([[1.0 if c in t.lower() else 0.0 for c in concepts] + [0.1] for t in texts], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    class FakeCE:
        def predict(self, pairs):
            return np.array([5.0 if ("gobernador" in q.lower() and "gobernador" in d.lower()) else -3.0
                             for q, d in pairs])

    monkeypatch.setattr(rag, "_embed_texts", fake_embed)
    monkeypatch.setattr(rag, "_reranker", lambda *a, **k: FakeCE())
    gr._CACHE.clear()
    return EngineState(cfg)


def test_graph_is_built_from_passages_entities_and_story_order(state):
    g = gr.build(state, "corp")
    assert g["nc"] == 4 and g["n"] == 4 + 7
    assert g["edges"] > 0


@pytest.mark.parametrize("seeding", ["embed", "cross"])
def test_walk_reaches_passage_with_no_shared_words(state, seeding):
    q = "¿Qué quiso saber el gobernador romano del acusado?"
    assert not ({w.strip("¿?").lower() for w in q.split()} & {w.lower() for w in CHUNKS[0]["text"].split()} - {"el", "de", "los", "si"})
    ranked = gr.rank(state, "corp", q, top_n=4, seeding=seeding)
    assert ranked[0]["id"] == "p0"


def test_walk_follows_story_order_to_neighbouring_passage(state):
    ranked = [h["id"] for h in gr.rank(state, "corp", "¿Qué hizo Herodes con los niños?", top_n=4, seeding="embed")]
    assert ranked[0] == "p2" and "p3" in ranked  # p3 reachable via passage adjacency


def test_no_graph_means_no_graph_results(state):
    (state.config.data_dir / "kgraph" / "corp.db").unlink()
    assert gr.rank(state, "corp", "cualquier cosa") == []


def test_retrieve_grounded_reserves_a_graph_slot(state, monkeypatch):
    # first-stage pool that does NOT contain p0 (the vocabulary gap)
    pool = [{"id": c["id"], "text": c["text"], "source": c["source"], "corpus": "corp", "score": -1.0}
            for c in CHUNKS if c["id"] != "p0"]
    monkeypatch.setattr(rag, "_grounded_candidates", lambda *a, **k: [dict(p) for p in pool])
    monkeypatch.setattr(rag, "_rerank", lambda q, hits: sorted(hits, key=lambda h: h["id"], reverse=True))
    q = "¿Qué quiso saber el gobernador romano del acusado?"
    hits, dbg = rag.retrieve_grounded(state, q, top_k=3, corpus="corp", multi_query=False,
                                      parent_child=False, allow_cloud=False, graph_slot=True)
    ids = [h["id"] for h in hits]
    # p0 is only reachable through the graph; it can land either in the
    # reserved slot or earlier (MMR may pick it once it is in the pool) --
    # either way the reserved slot holds a graph passage not otherwise chosen
    assert "p0" in ids, (ids, dbg)
    assert hits[-1].get("graph_slot") and dbg["graph_slot"] == ids[-1]
    plain, _ = rag.retrieve_grounded(state, q, top_k=3, corpus="corp", multi_query=False,
                                     parent_child=False, allow_cloud=False, graph_slot=False)
    assert "p0" not in [h["id"] for h in plain]


def test_graph_slot_skipped_when_retrieval_is_confident(state, monkeypatch):
    # an easy question: first-stage retrieval is confident (top rerank score
    # >= 0), so the slot must NOT displace a passage with a graph guess
    pool = [{"id": c["id"], "text": c["text"], "source": c["source"], "corpus": "corp", "score": 6.0}
            for c in CHUNKS if c["id"] != "p0"]
    monkeypatch.setattr(rag, "_grounded_candidates", lambda *a, **k: [dict(p) for p in pool])
    monkeypatch.setattr(rag, "_rerank", lambda q, hits: sorted(hits, key=lambda h: h["id"], reverse=True))
    hits, dbg = rag.retrieve_grounded(state, "¿Qué quiso saber el gobernador romano?", top_k=3, corpus="corp",
                                      multi_query=False, parent_child=False, allow_cloud=False, graph_slot=True)
    assert "p0" not in [h["id"] for h in hits]
    assert "graph_slot_skipped" in dbg and "graph_slot" not in dbg
