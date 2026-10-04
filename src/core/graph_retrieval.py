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

"""Graph-walk retrieval -- "connect the points" between a question and the
passages that answer it, when the question and the passage share no words.

Why: on paraphrased / multi-step questions ("the Roman governor", "the
ruler who ordered the killing of the infants") the passage says "Pilatos",
"Herodes" -- no lexical overlap, and the multilingual MiniLM embedder isn't
strong enough to bridge it (measured: 8 of 15 missed passages never even
entered the 30-candidate pool). The knowledge graph already holds the
bridge: (Pilatos | es | gobernador), (Herodes | mandó_matar |
niños_de_Bethlehem), plus which passage each fact came from.

Model (discrete math, no LLM at query time):
  * Undirected weighted graph G = (V, E), V = passages ∪ entities.
      entity — passage   the fact was extracted from that passage (w=1),
                         or the entity's name occurs in the passage (w=0.5)
      entity — entity    a fact links them (w=1)
      passage — passage  consecutive passages of the same book (w=0.3)
  * Seeding: every entity gets a *descriptor* = its name + every fact about
    it ("Pilatos: es gobernador; gobernador_de Palestina; ..."). The question
    is embedded and matched against descriptors (cosine), so a question that
    only says "gobernador romano" activates Pilatos. Entity names found
    literally in the question are seeded too.
  * Personalized PageRank from the seed distribution s:
        p ← (1-d)·s + d·P·p,   P = column-normalized adjacency, d = 0.5
    Probability mass flows question → roles → entities → their passages →
    linked entities → further passages, so the second passage of a chained
    question is reachable in two hops. Passages ranked by p.
  * The graph ranking is then combined with the other rankers by reciprocal
    rank fusion (see rag._grounded_rank_fusion), so no single judge can bury
    a passage the others agree on.
"""
from __future__ import annotations

import json
import sqlite3
import unicodedata
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .state import EngineState

DAMPING = 0.5
N_SEED_ENTITIES = 8
_CACHE: dict[str, dict[str, Any]] = {}


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[\s_]+", " ", t).strip()


def _paths(state: EngineState, corpus: str) -> dict[str, Path]:
    return {
        "db": state.config.data_dir / "kgraph" / f"{corpus}.db",
        "meta": state.config.data_dir / "rag_index" / corpus / "semantic" / "chunks.json",
        "cache": state.config.data_dir / "graph_index" / corpus,
    }


def available(state: EngineState, corpus: str) -> bool:
    p = _paths(state, corpus)
    return p["db"].exists() and p["meta"].exists()


def build(state: EngineState, corpus: str, force: bool = False) -> dict[str, Any]:
    """Build (or load cached) graph + entity descriptor embeddings. Cached on
    disk; rebuilt automatically when the knowledge graph or index changes."""
    from scipy import sparse

    from .rag import _embed_texts

    p = _paths(state, corpus)
    stamp = f"{p['db'].stat().st_mtime_ns}-{p['meta'].stat().st_mtime_ns}"
    key = f"{corpus}:{stamp}"
    if not force and key in _CACHE:
        return _CACHE[key]

    chunks = json.loads(p["meta"].read_text(encoding="utf-8"))
    cid = {c["id"]: i for i, c in enumerate(chunks)}
    nc = len(chunks)
    con = sqlite3.connect(str(p["db"]))
    ents = con.execute("SELECT id, name, norm FROM entities").fetchall()
    triples = con.execute(
        "SELECT t.subject_id, t.relation, t.object_id, t.chunk_id, es.name, eo.name FROM triples t "
        "JOIN entities es ON es.id=t.subject_id JOIN entities eo ON eo.id=t.object_id"
    ).fetchall()
    con.close()
    eidx = {eid: nc + i for i, (eid, _, _) in enumerate(ents)}
    n = nc + len(ents)

    rows, cols, vals = [], [], []

    def link(a: int, b: int, w: float) -> None:
        rows.extend((a, b)); cols.extend((b, a)); vals.extend((w, w))

    facts: dict[int, list[str]] = {}
    for s, rel, o, chunk_id, sname, oname in triples:
        a, b = eidx[s], eidx[o]
        link(a, b, 1.0)
        if chunk_id in cid:
            link(a, cid[chunk_id], 1.0)
            link(b, cid[chunk_id], 1.0)
        r = rel.replace("_", " ")
        facts.setdefault(s, []).append(f"{r} {oname.replace('_', ' ')}")
        facts.setdefault(o, []).append(f"{sname.replace('_', ' ')} {r}")

    # literal mentions: entity name inside a passage (catches passages the
    # extractor didn't attach a fact to)
    norm_texts = [_norm(c["text"]) for c in chunks]
    for eid, _, enorm in ents:
        if len(enorm) < 4:
            continue
        pat = f" {enorm} "
        for i, t in enumerate(norm_texts):
            if pat in f" {t} ":
                link(eidx[eid], i, 0.5)

    # story order: consecutive passages of the same book
    for i in range(nc - 1):
        if chunks[i].get("source") == chunks[i + 1].get("source"):
            link(i, i + 1, 0.3)

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=0)).ravel()
    deg[deg == 0] = 1.0
    P = A @ sparse.diags(1.0 / deg)  # column-stochastic transition matrix

    desc = [f"{name.replace('_', ' ')}: " + "; ".join(facts.get(eid, [])[:12]) for eid, name, _ in ents]
    cache_dir = p["cache"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_file, stamp_file = cache_dir / "entity_desc_emb.npy", cache_dir / "stamp.txt"
    if emb_file.exists() and stamp_file.exists() and stamp_file.read_text() == stamp and not force:
        demb = np.load(emb_file)
    else:
        demb = _embed_texts(desc).astype(np.float32)
        np.save(emb_file, demb)
        stamp_file.write_text(stamp)

    g = {"chunks": chunks, "nc": nc, "P": P, "ents": ents, "eidx": eidx,
         "desc_emb": demb, "desc_text": desc, "n": n, "edges": int(A.nnz // 2)}
    _CACHE.clear()
    _CACHE[key] = g
    return g


def rank(state: EngineState, corpus: str, question: str, top_n: int = 50,
         extra_seed_chunks: Optional[list[str]] = None, seeding: str = "cross") -> list[dict[str, Any]]:
    """Personalized-PageRank passage ranking for `question`.

    seeding: "embed" -- entity descriptors matched by embedding cosine only;
             "cross" -- cosine shortlist (60) re-scored by the cross-encoder
             (question vs "entity: its facts"), which is far better at
             linking a role ("el gobernador romano") to a name than the
             small bi-encoder that fails on paraphrases in the first place.
    """
    from .rag import _embed_texts, _reranker

    if not available(state, corpus):
        return []
    g = build(state, corpus)
    n, nc = g["n"], g["nc"]
    seed = np.zeros(n, dtype=np.float64)

    q = _embed_texts([question])[0]
    sims = g["desc_emb"] @ q
    if seeding == "cross":
        short = np.argsort(-sims)[:60]
        descs = g.get("desc_text")
        ce = _reranker().predict([(question, descs[i]) for i in short])
        order = np.argsort(-ce)[:N_SEED_ENTITIES]
        for j in order:
            seed[nc + short[j]] += float(np.exp(ce[j] / 2.0))
    else:
        top = np.argsort(-sims)[:N_SEED_ENTITIES]
        for i in top:
            if sims[i] > 0:
                seed[nc + i] += float(sims[i]) ** 2  # sharpen: strongest links dominate
    qn = f" {_norm(question)} "
    for i, (_, _, enorm) in enumerate(g["ents"]):
        if len(enorm) >= 4 and f" {enorm} " in qn:
            seed[nc + i] += 1.0
    for c in extra_seed_chunks or []:
        for j, ch in enumerate(g["chunks"]):
            if ch["id"] == c:
                seed[j] += 0.3
    if seed.sum() == 0:
        return []
    seed /= seed.sum()

    prob = seed.copy()
    for _ in range(40):
        prob = (1 - DAMPING) * seed + DAMPING * (g["P"] @ prob)
    scores = prob[:nc]
    order = np.argsort(-scores)[:top_n]
    return [{"id": g["chunks"][i]["id"], "graph_score": float(scores[i]),
             "source": g["chunks"][i].get("source", ""), "text": g["chunks"][i]["text"],
             "corpus": corpus} for i in order if scores[i] > 0]
