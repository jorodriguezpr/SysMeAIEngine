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

"""Tests for the P0/P1 grounding layer (grounding_settings, rag P1
upgrades) and the kgraph knowledge-graph addon. Cloud-dependent paths are
stubbed via monkeypatch — no network, no API key needed."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import kgraph as kg
from src.core import rag as rag_mod
from src.core.grounding_settings import (
    DEFAULTS as GROUNDING_DEFAULTS,
    get_grounding_settings,
    save_grounding_settings,
)
from src.core.state import EngineState


@pytest.fixture
def state(tmp_path):
    """EngineState pointed at a temp data dir — config fields are Path-typed
    but nothing here validates that, and tmp_path keeps tests hermetic."""
    cfg = type("Cfg", (), {
        "models_dir": tmp_path / "models",
        "data_dir": tmp_path / "data",
        "default_model": "minilm",
    })()
    (cfg.models_dir).mkdir(parents=True, exist_ok=True)
    return EngineState(cfg)


# ── grounding settings ──────────────────────────────────────────────────────

def test_grounding_defaults(state):
    s = get_grounding_settings(state)
    for k, v in GROUNDING_DEFAULTS.items():
        assert s[k] == v
    assert s["chat_mode"] == "plain"  # opt-in, never silently changes behavior


def test_grounding_save_roundtrip(state):
    save_grounding_settings(state, chat_mode="grounded", chat_top_k=4)
    s = get_grounding_settings(state)
    assert s["chat_mode"] == "grounded"
    assert s["chat_top_k"] == 4
    # untouched defaults survive
    assert s["chat_kgraph"] == GROUNDING_DEFAULTS["chat_kgraph"]


def test_grounding_rejects_bad_mode(state):
    with pytest.raises(ValueError):
        save_grounding_settings(state, chat_mode="bogus")


# ── triple parsing ──────────────────────────────────────────────────────────

def test_parse_triples_basic():
    raw = (
        "(Jesús | bautized | Juan el Bautista)\n"
        "(Nicodemo | visited | Jesús) \n"
        "garbage line\n"
        "(The Kingdom of God | teaches_about | repentance)"
    )
    triples = kg._parse_triples(raw)
    assert ("Jesús", "bautized", "Juan el Bautista") in triples
    assert ("Nicodemo", "visited", "Jesús") in triples
    assert len(triples) == 3  # garbage line excluded


def test_parse_triples_dedupes_case_insensitive():
    raw = "(jesus | loves | mary)\n(Jesus | loves | Mary)"
    assert len(kg._parse_triples(raw)) == 1


def test_parse_triples_rejects_overlong():
    raw = "(" + "x" * 200 + " | rel | obj)"
    assert kg._parse_triples(raw) == []


# ── entity normalization / matching ─────────────────────────────────────────

def test_norm_strips_accents_and_case():
    assert kg._norm("Jesús  de Nazaret ") == "jesus de nazaret"


def test_graph_facts_matches_entities(tmp_path, state):
    db = kg._db_path(state, "corpus-a")
    conn = kg._connect(db)
    kg._init_schema(conn)
    sid = kg._entity_id(conn, "Nicodemo")
    oid = kg._entity_id(conn, "Jesús")
    conn.execute(
        "INSERT INTO triples(subject_id, relation, object_id, chunk_id, source, created_at) "
        "VALUES (?,?,?,?,?,?)", (sid, "visited", oid, "abc123", "book1.txt", 0.0),
    )
    conn.commit()
    conn.close()

    res = kg.graph_facts(state, "corpus-a", "Quién es Nicodemo y qué hizo?")
    assert res["entities"] and res["entities"][0]["name"] == "Nicodemo"
    assert res["facts"][0]["subject"] == "Nicodemo"
    assert res["facts"][0]["chunk_id"] == "abc123"  # provenance preserved


def test_graph_facts_no_match_returns_empty(state, tmp_path):
    db = kg._db_path(state, "corpus-b")
    conn = kg._connect(db)
    kg._init_schema(conn)
    kg._entity_id(conn, "Pilato")
    conn.commit()
    conn.close()
    res = kg.graph_facts(state, "corpus-b", "something entirely unrelated here")
    assert res["entities"] == [] and res["facts"] == []


def test_graph_stats(tmp_path, state):
    db = kg._db_path(state, "corpus-c")
    conn = kg._connect(db)
    kg._init_schema(conn)
    sid = kg._entity_id(conn, "Jesús")
    oid = kg._entity_id(conn, "Pedro")
    conn.execute(
        "INSERT INTO triples(subject_id, relation, object_id, chunk_id, source, created_at) "
        "VALUES (?,?,?,?,?,?)", (sid, "called", oid, "c1", "s.txt", 0.0),
    )
    conn.commit()
    conn.close()
    stats = kg.graph_stats(state, "corpus-c")
    assert stats["exists"] and stats["entities"] == 2 and stats["triples"] == 1


def test_delete_graph(state, tmp_path):
    db = kg._db_path(state, "corpus-d")
    conn = kg._connect(db)
    kg._init_schema(conn)
    conn.commit()
    conn.close()
    assert kg.delete_graph(state, "corpus-d") is True
    assert kg.delete_graph(state, "corpus-d") is False


# ── P1 retrieval helpers ────────────────────────────────────────────────────

def test_expand_query_falls_back_without_provider(state):
    # no cloud settings file -> provider disabled -> returns [question]
    assert rag_mod.expand_query(state, "¿Quién bautizó a Jesús?") == ["¿Quién bautizó a Jesús?"]


def test_expand_query_uses_cloud_variants(state, monkeypatch):
    class FakeSettings:
        def __init__(self):
            self.store = {"enabled": True, "api_key": "k", "model": "m", "provider": "openai"}

    import src.core.cloud_settings as cs

    monkeypatch.setattr(cs, "get_cloud_settings", lambda st: {
        "enabled": True, "api_key": "k", "model": "m", "provider": "openai",
    })
    import src.core.cloud_chat as cc

    monkeypatch.setattr(cc, "cloud_generate", lambda s, p, max_tokens=None: "1. El bautismo de Jesús\n2. Quién administró el bautismo\n3. Extra")
    out = rag_mod.expand_query(state, "quien bautizo a jesus", n_variants=2)
    assert out[0] == "quien bautizo a jesus"
    assert len(out) == 3
    assert any("bautismo" in v for v in out[1:])


def test_mmr_diversify_prefers_distinct_hits():
    # 3 near-identical hits + 1 distinct one; top_k=2 must keep the distinct one
    base = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod"
    hits = [
        {"score": 0.9, "text": base},
        {"score": 0.85, "text": base},
        {"score": 0.5, "text": "completamente diferente contenido sobre apocrifos y libros antiguos"},
    ]
    out = rag_mod._mmr_diversify(hits, top_k=2)
    texts = [h["text"] for h in out]
    assert texts[0] == base  # most relevant first
    assert any("apocrifos" in t for t in texts[1:])


def test_expand_context_widens_from_index(state, tmp_path):
    # build a fake semantic index with 3 chunks in one source doc
    corpus = "c1"
    d = state.config.data_dir / "rag_index" / corpus / "semantic"
    d.mkdir(parents=True, exist_ok=True)
    chunks = [
        {"id": f"ch{i}", "chunk_index": i, "source": "book.txt",
         "text": f"Paragraph {i} text about the apostles and their journeys."}
        for i in range(3)
    ]
    (d / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    (d / "embeddings.npy").write_bytes(b"fake")  # only meta is read here
    hit = {"id": "ch1", "corpus": corpus, "source": "book.txt", "score": 0.9, "text": chunks[1]["text"]}
    out = rag_mod.expand_context(state, [hit])
    assert out[0].get("parent_expanded") is True
    assert "Paragraph 0" in out[0]["text"] and "Paragraph 2" in out[0]["text"]


def test_retrieve_grounded_no_indexes_returns_empty(state):
    hits, dbg = rag_mod.retrieve_grounded(state, "any question", corpus="all")
    assert hits == []
    assert dbg["variants"] == 1  # no provider -> no expansion