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

"""API-level regression tests for grounded chat + knowledge graph routes --
each reproduces a failure observed live before it was fixed (see
docs/GROUNDING.md "Verification"). Retrieval and generation are stubbed:
this tests the route wiring (cloud opt-out, fallback, response shape), not
retrieval quality."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.server import build_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    temp_dir = tempfile.mkdtemp(prefix="sysmeai-grounding-test-")
    for key, sub in (("DATA", ""), ("RAW", "raw"), ("CHUNKS", "chunks"), ("DATASETS", "datasets"),
                     ("MODELS", "models"), ("CHECKPOINTS", "checkpoints"), ("WEIGHTS", "weights")):
        os.environ[f"ENGINE_{key}_DIR"] = str(Path(temp_dir) / sub)
    return TestClient(build_app())


HIT = {"corpus": "c", "source": "s", "id": "x", "score": 1.0, "text": "passage"}
LOCAL = {"model": "local", "prompt_tokens": 1, "completion_tokens": 1, "elapsed_ms": 1.0, "text": "local answer"}


def test_grounded_chat_use_cloud_false_sends_nothing_to_cloud(client, monkeypatch):
    import src.api.chat_routes as cr
    import src.core.cloud_chat as cc
    from src.core import rag as rag_mod

    seen = {}

    def fake_retrieve_grounded(state, q, **kw):
        seen["allow_cloud"] = kw.get("allow_cloud")
        return [HIT], {"variants": 1}

    def no_cloud(*a, **kw):
        raise AssertionError("cloud must not be called when use_cloud=false")

    monkeypatch.setattr(rag_mod, "retrieve_grounded", fake_retrieve_grounded)
    monkeypatch.setattr(cc, "cloud_generate", no_cloud)
    monkeypatch.setattr(rag_mod, "local_answer", lambda state, prompt, model, lang, **kw: "local answer")
    r = client.post("/api/chat", json={"prompt": "q", "chat_mode": "grounded", "use_cloud": False})
    assert r.status_code == 200, r.text
    j = r.json()
    assert seen["allow_cloud"] is False
    assert j["model"].startswith("grounded:local:")
    assert j["sources"][0]["id"] == "x"
    assert j["grounding"]["use_cloud"] is False


def test_grounded_chat_cloud_network_failure_degrades_to_local(client, monkeypatch):
    import src.api.chat_routes as cr
    import src.core.cloud_chat as cc
    from src.core import rag as rag_mod

    def down(*a, **kw):
        raise RuntimeError("cloud provider request failed: ConnectionError: down")

    monkeypatch.setattr(rag_mod, "retrieve_grounded", lambda state, q, **kw: ([HIT], {"variants": 1}))
    monkeypatch.setattr(cc, "cloud_generate", down)
    monkeypatch.setattr(rag_mod, "local_answer", lambda state, prompt, model, lang, **kw: "local answer")
    r = client.post("/api/chat", json={"prompt": "q", "chat_mode": "grounded", "use_cloud": True})
    assert r.status_code == 200, r.text
    assert r.json()["model"].startswith("grounded:local:")
    assert "cloud failed" in r.json()["grounding"]["fallback"]


def test_grounded_chat_empty_cloud_answer_degrades_to_local(client, monkeypatch):
    import src.api.chat_routes as cr
    import src.core.cloud_chat as cc
    from src.core import rag as rag_mod

    monkeypatch.setattr(rag_mod, "retrieve_grounded", lambda state, q, **kw: ([HIT], {"variants": 1}))
    monkeypatch.setattr(cc, "cloud_generate", lambda *a, **kw: "")
    monkeypatch.setattr(rag_mod, "local_answer", lambda state, prompt, model, lang, **kw: "local answer")
    j = client.post("/api/chat", json={"prompt": "q", "chat_mode": "grounded", "use_cloud": True}).json()
    assert j["text"] == "local answer"
    assert "empty answer" in j["grounding"]["fallback"]


def test_plain_chat_response_has_no_grounding_fields(client, monkeypatch):
    import src.api.chat_routes as cr

    monkeypatch.setattr(cr, "core_chat", lambda state, prompt, **kw: dict(LOCAL))
    j = client.post("/api/chat", json={"prompt": "q", "chat_mode": "plain", "use_cloud": False}).json()
    assert j["text"] == "local answer"
    assert j["sources"] is None and j["grounding"] is None


@pytest.mark.parametrize("path", ["/api/kgraph/..%5Csentinel", "/api/kgraph/stats/..%5Cx"])
def test_kgraph_routes_reject_path_traversal(client, path):
    r = client.delete(path) if "stats" not in path else client.get(path)
    assert r.status_code == 400


def test_local_answers_get_compact_context_cloud_gets_widened(client, monkeypatch):
    import src.api.chat_routes as cr
    import src.core.cloud_chat as cc
    from src.core import rag as rag_mod

    calls = []

    def fake_rg(state, q, **kw):
        calls.append(kw["parent_child"])
        return [HIT], {"variants": 1}

    monkeypatch.setattr(rag_mod, "retrieve_grounded", fake_rg)
    monkeypatch.setattr(rag_mod, "local_answer", lambda state, prompt, model, lang, **kw: "local answer")
    # local-only request: compact from the start
    client.post("/api/chat", json={"prompt": "q", "chat_mode": "grounded", "use_cloud": False})
    assert calls == [False]
    # cloud answers: widened; cloud fails -> re-retrieve compact for the local model
    calls.clear()

    def down(*a, **kw):
        raise RuntimeError("down")

    monkeypatch.setattr(cc, "cloud_generate", down)
    client.post("/api/chat", json={"prompt": "q", "chat_mode": "grounded", "use_cloud": True})
    assert calls == [True, False]


def test_local_grounded_answer_uses_measured_complete_prompt(client, monkeypatch):
    from src.core import rag as rag_mod

    seen = {}
    monkeypatch.setattr(rag_mod, "retrieve_grounded", lambda state, q, **kw: (
        [{"corpus": "c", "source": "s", "id": "x", "score": 1.0, "text": "Pedro habló con Juan en el templo."}], {}))

    def fake_local(state, prompt, model, lang, **kw):
        seen.update(prompt=prompt, lang=lang, max_new_tokens=kw.get("max_new_tokens"))
        return "Pedro habló con Juan."

    monkeypatch.setattr(rag_mod, "local_answer", fake_local)
    j = client.post("/api/chat", json={"prompt": "¿Con quién habló Pedro?", "chat_mode": "grounded",
                                       "use_cloud": False, "max_new_tokens": 100}).json()
    assert j["text"] == "Pedro habló con Juan."
    assert "COMPLETA" in seen["prompt"] and seen["lang"] == "Spanish"
    assert seen["max_new_tokens"] >= 350
    assert j["grounding"]["local_prompt_style"] == "complete"
