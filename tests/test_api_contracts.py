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

"""API contract tests — hit the FastAPI app without booting a server."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.server import build_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # isolate storage per test run to stay hermetic
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="sysadmin-engine-test-")
    os.environ["ENGINE_DATA_DIR"] = temp_dir
    os.environ["ENGINE_RAW_DIR"] = str(Path(temp_dir) / "raw")
    os.environ["ENGINE_CHUNKS_DIR"] = str(Path(temp_dir) / "chunks")
    os.environ["ENGINE_DATASETS_DIR"] = str(Path(temp_dir) / "datasets")
    os.environ["ENGINE_MODELS_DIR"] = str(Path(temp_dir) / "models")
    os.environ["ENGINE_CHECKPOINTS_DIR"] = str(Path(temp_dir) / "checkpoints")
    os.environ["ENGINE_WEIGHTS_DIR"] = str(Path(temp_dir) / "weights")
    return TestClient(build_app())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_models_registry_flow(client):
    r = client.post("/api/registry/register", json={"name": "test-minilm", "provider": "minilm", "path": ""})
    assert r.status_code == 200

    r = client.get("/v1/models")
    assert r.status_code == 200
    names = [m["id"] for m in r.json()["data"]]
    assert "test-minilm" in names


def test_register_accepts_rag_corpus_and_device(client):
    r = client.post(
        "/api/registry/register",
        json={"name": "test-corpus-model", "provider": "minilm", "path": "", "device": "cpu", "rag_corpus": "some-corpus"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["rag_corpus"] == "some-corpus"
    assert body["device"] == "cpu"


def test_set_rag_corpus_route_updates_existing_model(client):
    client.post("/api/registry/register", json={"name": "test-set-corpus", "provider": "minilm", "path": ""})

    r = client.post("/api/registry/set-rag-corpus", json={"name": "test-set-corpus", "rag_corpus": "new-corpus"})
    assert r.status_code == 200
    assert r.json()["rag_corpus"] == "new-corpus"

    r = client.get("/api/registry")
    entry = next(m for m in r.json()["models"] if m["name"] == "test-set-corpus")
    assert entry["rag_corpus"] == "new-corpus"


def test_set_rag_corpus_route_404s_for_unknown_model(client):
    r = client.post("/api/registry/set-rag-corpus", json={"name": "does-not-exist", "rag_corpus": "x"})
    assert r.status_code == 400


def test_summarize_endpoint(client, tmp_path, monkeypatch):
    from src.core import summarize as summarize_mod

    doc = tmp_path / "book.txt"
    doc.write_text("Just one short paragraph of content to summarize.", encoding="utf-8")
    monkeypatch.setattr(
        summarize_mod, "core_chat",
        lambda state, prompt, model_name=None, max_new_tokens=256, temperature=0.7: {
            "model": "test-minilm", "prompt_tokens": 1, "completion_tokens": 1,
            "elapsed_ms": 0.0, "text": "A short summary.",
        },
    )
    r = client.post("/api/summarize", json={"source": str(doc)})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"] == "A short summary."
    assert body["chunk_count"] == 1


def test_summarize_endpoint_404s_for_missing_file(client, tmp_path):
    r = client.post("/api/summarize", json={"source": str(tmp_path / "nope.pdf")})
    assert r.status_code == 404


def test_chat_routes_to_summarizer_on_summarize_intent(client, tmp_path, monkeypatch):
    from src.core import summarize as summarize_mod

    doc = tmp_path / "book.txt"
    doc.write_text("Just one short paragraph of content to summarize.", encoding="utf-8")
    monkeypatch.setattr(
        summarize_mod, "core_chat",
        lambda state, prompt, model_name=None, max_new_tokens=256, temperature=0.7: {
            "model": "test-minilm", "prompt_tokens": 1, "completion_tokens": 1,
            "elapsed_ms": 0.0, "text": "A short summary.",
        },
    )
    r = client.post("/api/chat", json={"prompt": f"please summarize {doc}"})
    assert r.status_code == 200
    assert r.json()["text"] == "A short summary."


def test_translate_endpoint(client, monkeypatch):
    from src.core import translate as translate_mod

    monkeypatch.setattr(translate_mod, "translate", lambda text, direction: "translated text")
    r = client.post("/api/translate", json={"text": "hola mundo", "direction": "es-en"})
    assert r.status_code == 200
    body = r.json()
    assert body["translation"] == "translated text"
    assert body["direction"] == "es-en"


def test_translate_endpoint_rejects_unsupported_direction(client):
    r = client.post("/api/translate", json={"text": "hola", "direction": "fr-en"})
    assert r.status_code == 400


def test_chat_routes_to_translator_on_translate_intent(client, monkeypatch):
    from src.core import translate as translate_mod

    # chat_routes.chat() does `from ..core.translate import translate as
    # run_translate` as a *local* import inside the function body, re-run on
    # every call -- so patching the source attribute here (rather than some
    # cached name on chat_routes itself) is what actually takes effect.
    monkeypatch.setattr(translate_mod, "translate", lambda text, direction: "translated text")
    r = client.post("/api/chat", json={"prompt": 'translate "hola mundo" to english'})
    assert r.status_code == 200
    assert r.json()["text"] == "translated text"


def test_cloud_settings_get_defaults(client):
    r = client.get("/api/settings/cloud")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert body["api_key_set"] is False


def test_cloud_settings_save_and_key_masking(client):
    r = client.post("/api/settings/cloud", json={"enabled": True, "provider": "openai", "model": "gpt-4o", "api_key": "sk-realkey1234"})
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["model"] == "gpt-4o"
    assert body["api_key"] != "sk-realkey1234"  # never echo the real key back
    assert body["api_key_set"] is True

    # a later save with an empty api_key must NOT clear the previously saved one
    r2 = client.post("/api/settings/cloud", json={"model": "gpt-4o-mini", "api_key": ""})
    assert r2.status_code == 200
    assert r2.json()["api_key_set"] is True
    assert r2.json()["model"] == "gpt-4o-mini"


def test_chat_uses_cloud_when_requested(client, monkeypatch):
    from src.core import cloud_chat as cloud_chat_mod

    client.post("/api/settings/cloud", json={"enabled": True, "provider": "anthropic", "model": "test-cloud-model", "api_key": "sk-ant-test"})
    monkeypatch.setattr(cloud_chat_mod, "cloud_generate", lambda settings, prompt: "cloud answer")
    r = client.post("/api/chat", json={"prompt": "hello", "use_cloud": True})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "cloud answer"
    assert body["model"] == "cloud:anthropic:test-cloud-model"


def test_rag_chat_uses_cloud_when_requested(client, monkeypatch):
    from src.api import rag_routes as rag_routes_mod
    from src.core import cloud_chat as cloud_chat_mod

    client.post("/api/settings/cloud", json={"enabled": True, "provider": "anthropic", "model": "test-cloud-model", "api_key": "sk-ant-test"})
    fake_hits = [{"corpus": "c", "source": "doc1", "text": "some retrieved passage", "id": "1", "score": 0.9}]
    monkeypatch.setattr(rag_routes_mod.rag_mod, "retrieve", lambda *a, **k: fake_hits)
    # the default grounded path gathers candidates itself (single rerank,
    # see rag._grounded_candidates) instead of calling retrieve() -- stub
    # its entry point so this stays a route test, not a retrieval test
    monkeypatch.setattr(rag_routes_mod.rag_mod, "retrieve_grounded", lambda *a, **k: (list(fake_hits), {}))
    captured_prompt = {}

    def fake_cloud_generate(settings, prompt, max_tokens=None):
        captured_prompt["value"] = prompt
        return "grounded cloud answer"

    monkeypatch.setattr(cloud_chat_mod, "cloud_generate", fake_cloud_generate)
    r = client.post("/api/rag/chat", json={"prompt": "what does the document say?", "use_cloud": True})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "grounded cloud answer"
    assert body["model"] == "cloud:anthropic:test-cloud-model"
    assert body["sources"][0]["corpus"] == "c"
    assert body["sources"][0]["source"] == "doc1"
    # the cloud call must have actually received the RAG-augmented prompt
    # (retrieved passage folded in), not just the bare user question --
    # this is the whole point: local retrieval, cloud generation.
    assert "some retrieved passage" in captured_prompt["value"]


def test_rag_chat_routes_to_summarizer_on_summarize_intent(client, tmp_path, monkeypatch):
    from src.core import summarize as summarize_mod

    doc = tmp_path / "book.txt"
    doc.write_text("Just one short paragraph of content to summarize.", encoding="utf-8")
    monkeypatch.setattr(
        summarize_mod, "core_chat",
        lambda state, prompt, model_name=None, max_new_tokens=256, temperature=0.7: {
            "model": "test-minilm", "prompt_tokens": 1, "completion_tokens": 1,
            "elapsed_ms": 0.0, "text": "A short summary.",
        },
    )
    r = client.post("/api/rag/chat", json={"prompt": f"please summarize {doc}"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "A short summary."
    assert body["sources"] == []


def test_chat_openai_shape(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-minilm",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
        },
    )
    assert r.status_code == 200
    j = r.json()
    assert j["object"] == "chat.completion"
    assert j["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in j


def test_embeddings(client):
    r = client.post("/v1/embeddings", json={"model": "test-minilm", "input": "hello"})
    assert r.status_code == 200
    j = r.json()
    assert j["data"][0]["embedding"]


def test_metrics(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    j = r.json()
    assert "metrics" in j and "devices" in j
    assert j["metrics"]["requests"] >= 1


def test_mcp_initialize_and_chat_tool(client):
    r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert r.status_code == 200
    assert r.json()["result"]["serverInfo"]["name"] == "SysMeAIEngine"

    r = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "engine.chat", "arguments": {"prompt": "say hi", "max_tokens": 4}},
        },
    )
    assert r.status_code == 200
    content = r.json()["result"]["content"][0]
    assert content["type"] == "text"


def test_mcp_rag_tools(client):
    # hermetic temp dir has no RAG indexes built — must degrade to an empty
    # list rather than erroring (regression: list_corpora() was previously
    # called with the default method="hybrid", which is never a stored index
    # kind, so it silently always returned [] even when corpora existed).
    r = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "rag.list_corpora", "arguments": {}}},
    )
    assert r.status_code == 200
    body = json.loads(r.json()["result"]["content"][0]["text"])
    assert body["corpora"] == []

    r = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "rag.search", "arguments": {"query": "anything"}},
        },
    )
    assert r.status_code == 200
    # no index built -> retrieve() raises FileNotFoundError -> JSON-RPC error, not a 500
    assert "error" in r.json()


def test_chat_completions_with_tools_declared(client):
    # test-minilm has no chat template, so `tools` can't reach a native
    # tool-calling format — this just needs to not crash the request.
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-minilm",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "description": "x", "parameters": {"type": "object"}},
                }
            ],
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["finish_reason"] in ("stop", "tool_calls")


def test_anthropic_messages_shape(client):
    r = client.post(
        "/anthropic/v1/messages",
        json={
            "model": "test-minilm",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    j = r.json()
    assert j["role"] == "assistant"
    assert j["content"][0]["type"] == "text"


def test_ui_pages_render(client):
    for path in ("/ui", "/ui/models", "/ui/weights", "/ui/training", "/ui/agent", "/ui/mcp", "/ui/train-folder", "/ui/help", "/ui/settings", "/ui/kgraph"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "SysMeAIEngine" in r.text


def test_train_from_folder_ram_preflight_blocks_impossible_job(client, tmp_path, monkeypatch):
    import src.admin.pdf_training_routes as routes_mod

    # Force a tiny total RAM so any model's estimate exceeds it -- confirms
    # the endpoint returns 400 *without* ever spawning the training
    # subprocess, rather than actually launching a real (slow, network-
    # dependent) job just to prove the check works.
    monkeypatch.setattr(
        routes_mod, "probe_memory",
        lambda: {"total_gb": 0.001, "used_gb": 0.0, "available_gb": 0.0005, "percent": 50.0},
    )
    doc = tmp_path / "doc.txt"
    doc.write_text("some content " * 50, encoding="utf-8")
    r = client.post(
        "/api/train/from-folder",
        json={"source": str(tmp_path), "job_name": "preflight-block-test", "base_model": "gpt2"},
    )
    assert r.status_code == 400
    assert "exceeds this machine's total RAM" in r.json()["detail"]


def test_train_from_folder_passes_chunking_to_subprocess(client, tmp_path, monkeypatch):
    """Confirms --chunking actually reaches the launched subprocess's
    command line, without spawning a real (slow) training process --
    subprocess.Popen is replaced with a stub that just records the cmd
    it was called with."""
    import src.admin.pdf_training_routes as routes_mod

    doc = tmp_path / "doc.txt"
    doc.write_text("some content " * 50, encoding="utf-8")

    captured = {}

    class _FakeProc:
        pid = 99999

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(routes_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(routes_mod, "_watch_job", lambda proc, job, log_f: log_f.close())

    r = client.post(
        "/api/train/from-folder",
        json={"source": str(tmp_path), "job_name": "chunking-arg-test", "chunking": "paragraph"},
    )
    assert r.status_code == 200, r.text
    cmd = captured["cmd"]
    assert "--chunking" in cmd
    assert cmd[cmd.index("--chunking") + 1] == "paragraph"


def test_train_from_folder_rejects_nodes_with_minilm(client, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("some content " * 50, encoding="utf-8")
    r = client.post(
        "/api/train/from-folder",
        json={
            "source": str(tmp_path), "job_name": "minilm-cluster-test",
            "base_model": "minilm", "nodes": ["laptop-a"],
        },
    )
    assert r.status_code == 400
    assert "single-node only" in r.json()["detail"]


def test_train_from_folder_rejects_unregistered_node(client, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("some content " * 50, encoding="utf-8")
    r = client.post(
        "/api/train/from-folder",
        json={
            "source": str(tmp_path), "job_name": "unreg-node-test",
            "base_model": "gpt2", "nodes": ["totally-not-registered"],
        },
    )
    assert r.status_code == 400
    assert "not registered" in r.json()["detail"]


def test_train_from_folder_rejects_cloud_qa_when_cloud_not_configured(client, tmp_path):
    # ensure a clean disabled state regardless of test execution order
    client.post("/api/settings/cloud", json={"enabled": False})
    doc = tmp_path / "doc.txt"
    doc.write_text("some content " * 50, encoding="utf-8")
    r = client.post(
        "/api/train/from-folder",
        json={
            "source": str(tmp_path), "job_name": "cloud-qa-not-configured-test",
            "base_model": "gpt2", "dataset_format": "cloud_qa",
        },
    )
    assert r.status_code == 400
    assert "cloud AI provider is not enabled" in r.json()["detail"]


def test_train_from_folder_allows_cloud_qa_when_configured(client, tmp_path, monkeypatch):
    import src.admin.pdf_training_routes as routes_mod

    client.post(
        "/api/settings/cloud",
        json={"enabled": True, "provider": "anthropic", "model": "test-cloud-model", "api_key": "sk-test"},
    )
    doc = tmp_path / "doc.txt"
    doc.write_text("some content " * 50, encoding="utf-8")

    captured = {}

    class _FakeProc:
        pid = 99999

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(routes_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(routes_mod, "_watch_job", lambda proc, job, log_f: log_f.close())

    r = client.post(
        "/api/train/from-folder",
        json={
            "source": str(tmp_path), "job_name": "cloud-qa-configured-test",
            "base_model": "gpt2", "dataset_format": "cloud_qa",
        },
    )
    assert r.status_code == 200, r.text
    assert "--dataset-format" in captured["cmd"]
    idx = captured["cmd"].index("--dataset-format")
    assert captured["cmd"][idx + 1] == "cloud_qa"
    client.post("/api/settings/cloud", json={"enabled": False})  # leave clean for other tests


def test_train_from_folder_page_lists_registered_cluster_nodes(client):
    r = client.post("/api/cluster/nodes", json={"name": "ui-test-node", "host": "127.0.0.1", "port": 9999})
    assert r.status_code == 200
    try:
        r = client.get("/ui/train-folder")
        assert r.status_code == 200
        assert "ui-test-node" in r.text
    finally:
        client.delete("/api/cluster/nodes/ui-test-node")


def test_preview_train_from_folder_estimates_without_starting_a_job(client, tmp_path):
    for i in range(3):
        (tmp_path / f"doc{i}.txt").write_text("Some real prose content here. " * 20, encoding="utf-8")
    r = client.post(
        "/api/train/from-folder/preview",
        json={"source": str(tmp_path), "dataset_format": "lm", "max_rows": 5000, "max_sample_files": 8},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["total_files"] == 3
    assert j["sampled_files"] == 3
    assert j["estimated_rows"] > 0
    assert j["estimated_tokens"] > 0
    assert "no sampling needed" in j["note"]


def test_preview_train_from_folder_scales_up_when_sampled(client, tmp_path):
    for i in range(5):
        (tmp_path / f"doc{i}.txt").write_text("Some real prose content here. " * 20, encoding="utf-8")
    r = client.post(
        "/api/train/from-folder/preview",
        json={"source": str(tmp_path), "max_sample_files": 2},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["total_files"] == 5
    assert j["sampled_files"] == 2
    assert "estimated from a sample" in j["note"]


def test_preview_train_from_folder_missing_folder_404(client, tmp_path):
    r = client.post(
        "/api/train/from-folder/preview",
        json={"source": str(tmp_path / "nope")},
    )
    assert r.status_code == 404


def test_preview_train_from_folder_respects_chunking_param(client, tmp_path):
    # This exact paragraph/repeat count is verified (not guessed) to yield
    # a different chunk count under window (4) vs paragraph (5) chunking --
    # see the manual check that produced these numbers before writing this
    # test. Proves the preview endpoint actually uses body.chunking rather
    # than silently always using the "window" default.
    paras = [
        f"This is real sentence number {i} with enough distinct content to matter "
        "here today, filler filler filler. " * 8
        for i in range(5)
    ]
    (tmp_path / "doc.txt").write_text("\n\n".join(paras), encoding="utf-8")

    r_window = client.post(
        "/api/train/from-folder/preview",
        json={"source": str(tmp_path), "chunking": "window"},
    )
    r_paragraph = client.post(
        "/api/train/from-folder/preview",
        json={"source": str(tmp_path), "chunking": "paragraph"},
    )
    assert r_window.status_code == 200 and r_paragraph.status_code == 200
    assert r_window.json()["estimated_rows"] == 4
    assert r_paragraph.json()["estimated_rows"] == 5


def test_preview_train_from_folder_respects_max_rows_cap(client, tmp_path):
    for i in range(20):
        (tmp_path / f"doc{i}.txt").write_text(f"Unique document number {i} with distinct content. " * 5, encoding="utf-8")
    r = client.post(
        "/api/train/from-folder/preview",
        json={"source": str(tmp_path), "max_rows": 3, "max_sample_files": 20},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["estimated_rows"] <= 3
    assert j["capped_by_max_rows"] is True


def test_preview_reports_exact_token_counts_and_truncation(client, tmp_path):
    (tmp_path / "doc.txt").write_text(
        "Plenty of real prose to push each chunk well past a tiny max_seq limit. " * 60,
        encoding="utf-8",
    )
    r = client.post(
        "/api/train/from-folder/preview",
        json={"source": str(tmp_path), "base_model": "minilm", "max_seq": 64},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["token_count_method"].startswith("exact (TinyGPT")
    assert j["avg_tokens_per_row"] > 0
    assert j["max_seq"] == 64
    for key in ("estimated_tokens_trained", "rows_truncated_pct", "text_truncated_pct", "truncation_warning"):
        assert key in j


def test_preview_truncation_warning_with_fallback_counts(client, tmp_path, monkeypatch):
    from src.core import token_preview

    monkeypatch.setattr(token_preview, "load_preview_tokenizer", lambda m: (None, "offline"))
    (tmp_path / "doc.txt").write_text("Long prose sentence number one here. " * 80, encoding="utf-8")
    j = client.post(
        "/api/train/from-folder/preview",
        json={"source": str(tmp_path), "base_model": "gpt2", "max_seq": 32},
    ).json()
    assert j["token_count_method"].startswith("estimate")
    assert j["rows_truncated_pct"] == 100.0
    assert j["estimated_tokens_trained"] < j["estimated_tokens"]
    assert "cut off" in j["truncation_warning"]


def test_rag_eval_api_runs_in_background_and_reports(client, monkeypatch):
    import time as _time

    from src.core import rag as rag_mod

    monkeypatch.setattr(rag_mod, "list_corpora", lambda state, method="hybrid": ["evalcorp"])
    monkeypatch.setattr(
        rag_mod, "evaluate_retrieval",
        lambda state, corpus, **kw: {"corpus": corpus, "recall_at_k": 1.0, "hit_at_1": 1.0, "mrr": 1.0,
                                     "question_mode": kw["question_mode"], "details": []},
    )
    assert client.get("/api/rag/corpora").json() == {"corpora": ["evalcorp"]}
    r = client.post("/api/rag/eval", json={"corpus": "evalcorp", "question_mode": "keywords"})
    assert r.status_code == 200
    eval_id = r.json()["eval_id"]
    for _ in range(50):
        s = client.get(f"/api/rag/eval/{eval_id}").json()
        if s["status"] != "running":
            break
        _time.sleep(0.05)
    assert s["status"] == "done", s
    assert s["result"]["retrieval"]["question_mode"] == "keywords"
    assert "answers" not in s["result"]


def test_rag_eval_api_validates_input(client, monkeypatch):
    from src.core import rag as rag_mod

    monkeypatch.setattr(rag_mod, "list_corpora", lambda state, method="hybrid": ["evalcorp"])
    assert client.post("/api/rag/eval", json={"corpus": "missing"}).status_code == 404
    assert client.post("/api/rag/eval", json={"corpus": "evalcorp", "question_mode": "bad"}).status_code == 400
    assert client.post("/api/rag/eval", json={"corpus": "evalcorp", "held_out_fraction": 0}).status_code == 400
    assert client.get("/api/rag/eval/nope").status_code == 404


def test_train_from_folder_passes_max_seq_to_subprocess_and_preflight(client, tmp_path, monkeypatch):
    import src.admin.pdf_training_routes as routes_mod

    (tmp_path / "doc.txt").write_text("some content " * 50, encoding="utf-8")
    captured = {}

    class _FakeProc:
        pid = 99999

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    real_preflight = routes_mod.check_ram_preflight

    def _spy_preflight(*a, **kw):
        captured["preflight_max_seq"] = kw["max_seq"]
        return real_preflight(*a, **kw)

    monkeypatch.setattr(routes_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(routes_mod, "_watch_job", lambda proc, job, log_f: log_f.close())
    monkeypatch.setattr(routes_mod, "check_ram_preflight", _spy_preflight)

    r = client.post("/api/train/from-folder", json={"source": str(tmp_path), "job_name": "maxseq-test", "max_seq": 384})
    assert r.status_code == 200, r.text
    cmd = captured["cmd"]
    assert cmd[cmd.index("--max-seq") + 1] == "384"
    assert captured["preflight_max_seq"] == 384

    # omitted -> previous hardcoded default, unchanged behavior
    r = client.post("/api/train/from-folder", json={"source": str(tmp_path), "job_name": "maxseq-default"})
    assert captured["cmd"][captured["cmd"].index("--max-seq") + 1] == "128"
