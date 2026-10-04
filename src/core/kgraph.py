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

"""Lightweight knowledge-graph addon (P2): entities + relations extracted
from already-ingested chunks, stored in SQLite, queried by traversal.

Why: retrieval (even hybrid + rerank) only finds passages that *look like*
the question. Questions that need facts from two different places joined --
"who is X and what did he do?", "how are A and B related?" -- need the
knowledge to be in a form a machine can traverse, not a bag of paragraphs.
This module is the KR layer for this engine, kept deliberately small:

  * extraction  — one cloud call per chunk (same pattern/bounding as
    cloud_distill.py) writes (subject, relation, object) triples in the
    chunk's own language, each linked back to its source chunk id for
    citation provenance.
  * storage     — SQLite at data/kgraph/<corpus>.db; content-addressed so
    rebuilding the same corpus reuses work instead of duplicating.
  * retrieval   — detect corpus entities in the question (exact + light
    normalization), take 1-hop neighborhoods around them, and emit graph
    context rows that ride alongside RAG passages in the same prompt.

Everything is optional and non-fatal by design: chat grounding works with
or without a graph; a graph build failure never touches the RAG indexes.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .state import EngineState

DEFAULT_MAX_TRIPLES_PER_CHUNK = 8

_TRIPLES_RE = re.compile(
    r"\(\s*(?P<s>[^()|\n]+?)\s*\|\s*(?P<r>[^()|\n]+?)\s*\|\s*(?P<o>[^()|\n]+?)\s*\)"
)


_CORPUS_NAME_RE = re.compile(r"[\w][\w.\- ]*")


def _db_path(state: EngineState, corpus: str) -> Path:
    # The corpus name comes straight from API paths/bodies and becomes a
    # file name: without this, DELETE /api/kgraph/..%5Csentinel deleted a
    # .db file OUTSIDE data/kgraph/ (live-verified path traversal).
    if not _CORPUS_NAME_RE.fullmatch(corpus or "") or ".." in corpus:
        raise ValueError(f"invalid corpus name: {corpus!r}")
    # no mkdir here: read-only calls (stats/query/delete) must not create
    # data/kgraph/ as a side effect -- _connect() creates it, and the read
    # paths only connect after checking the .db already exists
    return state.config.data_dir / "kgraph" / f"{corpus}.db"


def _connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            norm TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS triples (
            id INTEGER PRIMARY KEY,
            subject_id INTEGER NOT NULL REFERENCES entities(id),
            relation TEXT NOT NULL,
            object_id INTEGER NOT NULL REFERENCES entities(id),
            chunk_id TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_triples_subj ON triples(subject_id);
        CREATE INDEX IF NOT EXISTS ix_triples_obj ON triples(object_id);
        CREATE INDEX IF NOT EXISTS ix_triples_chunk ON triples(chunk_id);
        -- every chunk that got a successful extraction call, INCLUDING ones
        -- that yielded zero triples -- so a rebuild doesn't pay for them
        -- again (the triples table alone can't tell "done, nothing found"
        -- from "never tried")
        CREATE TABLE IF NOT EXISTS processed_chunks (
            chunk_id TEXT PRIMARY KEY,
            n_triples INTEGER NOT NULL,
            processed_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS build_log (
            id INTEGER PRIMARY KEY,
            corpus TEXT NOT NULL,
            chunks_total INTEGER, chunks_done INTEGER, triples INTEGER,
            failed INTEGER, started_at REAL, finished_at REAL
        );
        """
    )
    conn.commit()


def _norm(name: str) -> str:
    """Case/space-normalized entity key so 'Jesús' and 'jesus ' dedupe.
    Accent-insensitive on purpose: extraction and the question come back in
    different accents more often than not (live behavior), and treating
    'Jesús' vs 'Jesus' as one entity costs almost nothing here."""
    import unicodedata

    n = unicodedata.normalize("NFKD", name.strip().lower())
    n = "".join(c for c in n if not unicodedata.combining(c))
    # underscores too: the extraction prompt's snake_case relation examples
    # leak into entity names ("rey_del_Norte" next to "rey del Norte" --
    # live-verified duplicate entities in a real 5-chunk build)
    return re.sub(r"[\s_]+", " ", n).strip()


# ── extraction ──────────────────────────────────────────────────────────────

def _triple_prompt(chunk_text: str, language: str, n_max: int) -> str:
    return (
        "Read the passage and extract factual knowledge as triples. A triple "
        "looks exactly like: (subject | relation | object). Extract up to "
        f"{n_max} of the most important facts: entities (people, places, "
        "groups, events, concepts, books, dates) and how they relate. Use "
        "short noun-phrase subjects/objects and concise relations (e.g. "
        "bautized, wrote_to, leader_of, located_in, teaches, happened_in). "
        f"Write everything in {language}. Respond ONLY with triples, one per "
        "line, nothing else.\n\n"
        f"Passage:\n{chunk_text}"
    )


def _parse_triples(raw: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for m in _TRIPLES_RE.finditer(raw):
        s, r, o = (m.group(k).strip().strip("'\"") for k in ("s", "r", "o"))
        if s and r and o and len(s) < 120 and len(o) < 120 and len(r) < 80:
            key = (s.lower(), r.lower(), o.lower())
            if key not in seen:
                seen.add(key)
                out.append((s, r, o))
    return out


def extract_triples_for_chunk(
    state: EngineState, chunk_text: str, language: str,
    n_max: int = DEFAULT_MAX_TRIPLES_PER_CHUNK,
) -> list[tuple[str, str, str]]:
    from .cloud_chat import cloud_generate
    from .cloud_settings import get_cloud_settings

    settings = get_cloud_settings(state)
    if not (settings.get("enabled") and settings.get("api_key") and settings.get("model")):
        raise ValueError(
            "cloud AI provider is not enabled/configured -- the knowledge "
            "graph extraction needs it (Settings page)"
        )
    raw = cloud_generate(
        settings, _triple_prompt(chunk_text, language, n_max),
    )  # default budget: a 1024 cap risked empty output from reasoning models,
    # which would then be recorded as a "processed, zero triples" chunk
    if not raw.strip():
        # empty output is a failed call (e.g. reasoning ate the budget), not
        # "this chunk has no facts" -- raising keeps it out of
        # processed_chunks so the next build retries it
        raise RuntimeError("cloud provider returned empty output for this chunk")
    return _parse_triples(raw)


def build_graph(
    state: EngineState,
    corpus: str,
    max_chunks: Optional[int] = None,
    on_progress: Optional[Callable[[int, int, int], None]] = None,
) -> dict[str, Any]:
    """Build the knowledge graph for a corpus from its RAG-indexed chunks.
    One cloud call per chunk (bounded by max_chunks), best-effort per chunk:
    transient failures skip that chunk instead of losing the whole build.
    Content-addressed: chunks already in the graph (by chunk id) are
    skipped, so a rebuild after adding files only processes new chunks.
    """
    from .cloud_distill import ensure_cloud_ready
    from .rag import _paths
    from .summarize import _detect_language

    ensure_cloud_ready(state)

    meta_path = _paths(state, "semantic", corpus)["meta"]
    if not meta_path.exists():
        meta_path = _paths(state, "tfidf", corpus)["meta"]
    if not meta_path.exists():
        raise FileNotFoundError(
            f"no RAG index found for corpus '{corpus}' -- build one first "
            "(the graph is extracted from the same indexed chunks)"
        )
    chunks = json.loads(meta_path.read_text(encoding="utf-8"))
    if max_chunks:
        chunks = chunks[:max_chunks]

    db = _db_path(state, corpus)
    conn = _connect(db)
    _init_schema(conn)

    existing = {r[0] for r in conn.execute("SELECT chunk_id FROM processed_chunks")}
    # graphs built before processed_chunks existed: triples still prove a chunk was done
    existing |= {r[0] for r in conn.execute("SELECT DISTINCT chunk_id FROM triples")}
    todo = [c for c in chunks if c["id"] not in existing]
    log_id = conn.execute(
        "INSERT INTO build_log(corpus, chunks_total, started_at) VALUES (?,?,?)",
        (corpus, len(todo), time.time()),
    ).lastrowid
    conn.commit()

    sample = " ".join(c["text"] for c in chunks[:3])[:3000]
    language = _detect_language(sample) if chunks else "English"

    done = failed = 0
    for c in todo:
        try:
            triples = extract_triples_for_chunk(state, c["text"], language)
            for s, r, o in triples:
                sid = _entity_id(conn, s)
                oid = _entity_id(conn, o)
                conn.execute(
                    "INSERT INTO triples(subject_id, relation, object_id, chunk_id, source, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (sid, r, oid, c["id"], c.get("source", ""), time.time()),
                )
            conn.execute(
                "INSERT OR REPLACE INTO processed_chunks(chunk_id, n_triples, processed_at) VALUES (?,?,?)",
                (c["id"], len(triples), time.time()),
            )
            conn.commit()
        except (ValueError, RuntimeError, sqlite3.Error):
            # RuntimeError includes network failures (cloud_chat._post), so
            # one timeout skips one chunk instead of aborting the build;
            # failed chunks aren't recorded as processed and retry next build
            conn.rollback()
            failed += 1
        done += 1
        if on_progress:
            on_progress(done, len(todo), failed)

    n_triples = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
    conn.execute(
        "UPDATE build_log SET chunks_done=?, triples=?, failed=?, finished_at=? WHERE id=?",
        (done, n_triples, failed, time.time(), log_id),
    )
    conn.commit()
    conn.close()
    return {
        "corpus": corpus, "chunks_processed": done, "chunks_failed": failed,
        "triples_total": n_triples, "db": str(db),
    }


def _entity_id(conn: sqlite3.Connection, name: str) -> int:
    norm = _norm(name)
    row = conn.execute("SELECT id FROM entities WHERE norm=?", (norm,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO entities(name, norm) VALUES (?,?)", (name.strip(), norm))
    return cur.lastrowid


# ── query ───────────────────────────────────────────────────────────────────

def _detect_entities(conn: sqlite3.Connection, question: str) -> list[dict[str, Any]]:
    """Entities whose normalized name appears in the question. Exact match
    on the normalized form handles case/accent drift; a token-prefix check
    catches plural/short forms ('apostol' matching 'apostoles')."""
    qnorm = _norm(question)
    qtokens = set(qnorm.split())
    found: dict[int, dict[str, Any]] = {}
    for eid, name, norm in conn.execute("SELECT id, name, norm FROM entities").fetchall():
        if len(norm) < 3:
            continue
        if norm in qnorm or norm in qtokens:
            found[eid] = {"id": eid, "name": name}
            continue
        toks = norm.split()
        if len(toks) == 1 and toks[0] in qtokens and len(toks[0]) >= 4:
            found[eid] = {"id": eid, "name": name}
    return list(found.values())


def graph_facts(
    state: EngineState, corpus: str, question: str, max_facts: int = 10,
) -> dict[str, Any]:
    """1-hop facts around entities detected in the question, newest first,
    each carrying its source chunk id (provenance for citations)."""
    db = _db_path(state, corpus)
    if not db.exists():
        return {"corpus": corpus, "entities": [], "facts": []}
    conn = _connect(db)
    ents = _detect_entities(conn, question)
    if not ents:
        conn.close()
        return {"corpus": corpus, "entities": [], "facts": []}

    facts: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int]] = set()
    for e in ents[:4]:
        # created_at must be a SELECTed column for ORDER BY to work over a
        # UNION ALL result (SQLite rejects ordering on non-result terms).
        rows = conn.execute(
            """
            SELECT es.name, t.relation, eo.name, t.chunk_id, t.source, t.created_at
            FROM triples t
            JOIN entities es ON es.id = t.subject_id
            JOIN entities eo ON eo.id = t.object_id
            WHERE t.subject_id = ?
            UNION ALL
            SELECT es.name, t.relation, eo.name, t.chunk_id, t.source, t.created_at
            FROM triples t
            JOIN entities es ON es.id = t.subject_id
            JOIN entities eo ON eo.id = t.object_id
            WHERE t.object_id = ?
            ORDER BY created_at DESC
            """,
            (e["id"], e["id"]),
        ).fetchall()
        for s, r, o, chunk_id, source, _created in rows:
            key = (e["id"], r.lower(), hash((s, o)))
            if key in seen:
                continue
            seen.add(key)
            facts.append({"subject": s, "relation": r, "object": o, "chunk_id": chunk_id, "source": source})
            if len(facts) >= max_facts:
                break
        if len(facts) >= max_facts:
            break
    conn.close()
    return {"corpus": corpus, "entities": ents, "facts": facts}


def _format_graph_context(facts: list[dict[str, Any]]) -> str:
    lines = [f"{f['subject']} --[{f['relation']}]--> {f['object']}" for f in facts]
    return "\n".join(lines)


def retrieve_graph_context(
    state: EngineState, questions: list[str], corpus: str | list[str] = "all",
    max_facts: int = 8,
) -> list[dict[str, Any]]:
    """Graph context rows shaped like RAG hits so they can ride in the same
    prompt and MMR pipeline: text = fact lines, source = the primary
    entity, score = 1.0 (graph facts are exact matches by construction).
    Dedupes across questions/corpora."""
    names = (
        list_corpora(state) if corpus == "all"
        else [corpus] if isinstance(corpus, str) else list(corpus)
    )
    seen_facts: set[str] = set()
    hits: list[dict[str, Any]] = []
    for name in names:
        for q in questions[:2]:  # original + first variant is plenty
            res = graph_facts(state, name, q, max_facts=max_facts)
            fresh = [f for f in res["facts"] if f["subject"] not in seen_facts]
            if not fresh:
                continue
            for f in fresh:
                seen_facts.add(f["subject"])
            hits.append({
                "score": 1.0,
                "source": f"kgraph:{res['entities'][0]['name'] if res['entities'] else name}",
                "text": _format_graph_context(fresh),
                "corpus": name,
                "id": f"kgraph:{name}:{hash(tuple(sorted(seen_facts))) & 0xFFFF:04x}",
                "kgraph": True,
            })
            break
    return hits


def list_corpora(state: EngineState) -> list[str]:
    root = state.config.data_dir / "kgraph"
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.db"))


def list_rag_corpora(state: EngineState) -> list[str]:
    """Corpora that have a built semantic RAG index — the graph is always
    extracted from a RAG index's chunks, so these are the buildable ones."""
    root = state.config.data_dir / "rag_index"
    if not root.exists():
        return []
    return sorted(d.parent.parent.name for d in root.glob("*/semantic/embeddings.npy"))


def graph_stats(state: EngineState, corpus: str) -> dict[str, Any]:
    db = _db_path(state, corpus)
    if not db.exists():
        return {"corpus": corpus, "exists": False}
    conn = _connect(db)
    n_ent = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    n_tri = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
    n_chunks = conn.execute("SELECT COUNT(DISTINCT chunk_id) FROM triples").fetchone()[0]
    top = conn.execute(
        """
        SELECT e.name, COUNT(*) AS n FROM triples t
        JOIN entities e ON e.id = t.subject_id
        GROUP BY t.subject_id ORDER BY n DESC LIMIT 15
        """
    ).fetchall()
    conn.close()
    return {
        "corpus": corpus, "exists": True, "entities": n_ent, "triples": n_tri,
        "chunks_covered": n_chunks,
        "top_entities": [{"name": n, "facts": c} for n, c in top],
    }


def delete_graph(state: EngineState, corpus: str) -> bool:
    db = _db_path(state, corpus)
    if db.exists():
        db.unlink()
        # WAL sidecars too
        for suffix in ("-wal", "-shm"):
            side = Path(str(db) + suffix)
            if side.exists():
                side.unlink()
        return True
    return False