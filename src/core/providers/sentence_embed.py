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

"""Dedicated sentence-embedding provider (sentence-transformers).

`HFProvider.embed()` mean-pools hidden states out of a *causal* LM — that
works in a pinch but those models were never trained to place semantically
similar sentences near each other in vector space. This provider loads a
model actually trained for that, giving much better retrieval quality for
RAG. Embedding-only: `generate()` is not supported.

DEFAULT_MODEL_ID is multilingual (50+ languages, Spanish included as a
high-resource language) rather than the English-only all-MiniLM-L6-v2 --
this project's own corpora are Spanish-language documents, and an
English-trained embedder places semantically-similar Spanish sentences no
closer together than unrelated ones, which quietly degrades every
"semantic"/"hybrid" RAG query on non-English content. Changing this
invalidates any already-built semantic index's vectors (different model =
different vector space, not just different numbers) -- existing corpora
need `rag build` re-run after an upgrade, they don't silently break, but
they also won't silently improve until rebuilt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SentenceEmbedProvider:
    name = "sentence_embed"

    def load(self, name: str, registry_path: Path, **kwargs: Any) -> tuple[Any, Any]:
        from sentence_transformers import SentenceTransformer  # lazy

        model_id = kwargs.get("path") or DEFAULT_MODEL_ID
        model = SentenceTransformer(model_id, device="cpu")
        return model, None

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError(
            "sentence_embed is an embedding-only provider; it has no generate()"
        )

    def embed(
        self, model: Any, tokenizer: Any, text: str | list[str]
    ) -> list[list[float]]:
        texts = [text] if isinstance(text, str) else text
        vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.tolist()

    def info(self, model: Any) -> dict[str, Any]:
        return {
            "kind": "SentenceTransformer",
            "dimension": model.get_sentence_embedding_dimension(),
        }
