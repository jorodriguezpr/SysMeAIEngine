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

"""Tests for src.core.rag -- specifically the incremental semantic-embedding
logic (_embed_incremental), which is the correctness-critical part of the
"don't re-embed the whole corpus for one added file" fix. Mocks _embed_texts
so this stays fast/offline; a real end-to-end run with the actual embedding
model was done manually during development (see memory) confirming a
same-corpus rebuild embeds nothing new and an added file embeds only its
own chunks."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import rag as rag_mod  # noqa: E402


def _fake_embed(texts, model_id=None, batch_size=64):
    # Deterministic per-text "embedding" so equality is easy to assert on.
    return np.array([[float(len(t)), float(ord(t[0]))] for t in texts], dtype=np.float32)


def test_chunk_source_dir_is_paragraph_aware(tmp_path):
    """Regression test for a real bug found live: RAG indexing used a plain
    fixed-size sliding window regardless of sentence/paragraph boundaries,
    so a retrieved chunk could (and did, confirmed by inspecting a real
    index) end mid-sentence -- e.g. "...pero las s". Paragraph-aware
    chunking means chunk boundaries only ever fall between paragraphs."""
    text = (
        "Primer parrafo con contenido real y suficiente para ser un parrafo "
        "completo por si mismo, sin cortes raros en medio de la oracion.\n\n"
        "Segundo parrafo, tambien completo, que habla de otra cosa distinta "
        "al primero para que quede claro que son dos unidades separadas."
    )
    (tmp_path / "doc.txt").write_text(text, encoding="utf-8")

    docs = rag_mod._chunk_source_dir(tmp_path)

    assert len(docs) >= 1
    for d in docs:
        assert d["text"].strip().endswith((".", "?", "!"))  # never cuts mid-word/mid-sentence


def test_list_corpora_default_method_finds_semantic_index(tmp_path, monkeypatch):
    """Regression test for a real bug found live via the Models page: the
    RAG-corpus dropdown was always empty because list_corpora(state) with no
    method arg defaults to "hybrid", but "hybrid" isn't its own stored index
    -- it fuses "semantic" + "tfidf" at query time (see build_index()).
    retrieve() already translates "hybrid" -> "semantic" for its own listing;
    list_corpora() didn't, so it searched for a literal "hybrid" directory
    that never exists and always returned []."""
    from src.core.state import EngineState

    root = tmp_path / "data"
    (root / "rag_index" / "my-corpus" / "semantic").mkdir(parents=True)
    (root / "rag_index" / "my-corpus" / "semantic" / "embeddings.npy").write_bytes(b"x")

    state = EngineState.__new__(EngineState)
    state.config = type("Cfg", (), {"data_dir": root})()

    assert rag_mod.list_corpora(state) == ["my-corpus"]  # default method="hybrid"
    assert rag_mod.list_corpora(state, method="semantic") == ["my-corpus"]
    assert rag_mod.list_corpora(state, method="tfidf") == []  # no tfidf index built


def test_embed_incremental_first_build_embeds_everything(tmp_path):
    p = {"embeddings": tmp_path / "embeddings.npy", "meta": tmp_path / "chunks.json"}
    docs = [{"id": "a", "text": "hello"}, {"id": "b", "text": "world"}]
    with patch.object(rag_mod, "_embed_texts", side_effect=_fake_embed):
        vecs, n_embedded, n_reused = rag_mod._embed_incremental(p, docs)
    assert n_embedded == 2
    assert n_reused == 0
    assert vecs.shape == (2, 2)


def test_embed_incremental_reuses_unchanged_ids(tmp_path):
    p = {"embeddings": tmp_path / "embeddings.npy", "meta": tmp_path / "chunks.json"}
    old_docs = [{"id": "a", "text": "hello", "source": "s"}, {"id": "b", "text": "world", "source": "s"}]
    old_vecs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.save(p["embeddings"], old_vecs)
    p["meta"].write_text(json.dumps(old_docs), encoding="utf-8")

    # Same two ids, one new id "c" added.
    new_docs = [{"id": "a", "text": "hello"}, {"id": "b", "text": "world"}, {"id": "c", "text": "new chunk"}]
    with patch.object(rag_mod, "_embed_texts", side_effect=_fake_embed) as mock_embed:
        vecs, n_embedded, n_reused = rag_mod._embed_incremental(p, new_docs)

    assert n_embedded == 1
    assert n_reused == 2
    # Only the genuinely new text was ever sent to the (expensive) embedder.
    mock_embed.assert_called_once_with(["new chunk"])
    # Reused vectors are byte-identical to the originals, not recomputed.
    assert (vecs[0] == old_vecs[0]).all()
    assert (vecs[1] == old_vecs[1]).all()


def test_embed_incremental_drops_removed_chunks(tmp_path):
    p = {"embeddings": tmp_path / "embeddings.npy", "meta": tmp_path / "chunks.json"}
    old_docs = [{"id": "a", "text": "hello"}, {"id": "b", "text": "world"}]
    np.save(p["embeddings"], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    p["meta"].write_text(json.dumps(old_docs), encoding="utf-8")

    # "b" no longer present -- should not appear in the rebuilt vectors at all.
    new_docs = [{"id": "a", "text": "hello"}]
    with patch.object(rag_mod, "_embed_texts", side_effect=_fake_embed) as mock_embed:
        vecs, n_embedded, n_reused = rag_mod._embed_incremental(p, new_docs)

    assert vecs.shape == (1, 2)
    assert n_embedded == 0
    assert n_reused == 1
    mock_embed.assert_not_called()


def test_embed_incremental_falls_back_cleanly_on_corrupt_old_index(tmp_path):
    p = {"embeddings": tmp_path / "embeddings.npy", "meta": tmp_path / "chunks.json"}
    p["meta"].write_text("not valid json", encoding="utf-8")
    p["embeddings"].write_bytes(b"not a valid npy file")

    docs = [{"id": "a", "text": "hello"}]
    with patch.object(rag_mod, "_embed_texts", side_effect=_fake_embed):
        vecs, n_embedded, n_reused = rag_mod._embed_incremental(p, docs)
    assert n_embedded == 1
    assert n_reused == 0
