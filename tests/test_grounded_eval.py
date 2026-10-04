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

"""Tests for src.core.grounded_eval -- the measuring tool itself: context
recall, summary math, judge parsing/retry, multi-hop pairing through the
knowledge graph. Cloud calls are stubbed; real runs are in data/eval/."""
from __future__ import annotations

import json
import random
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import grounded_eval as ge  # noqa: E402
from src.core.state import EngineState  # noqa: E402


@pytest.fixture
def state(tmp_path):
    cfg = type("Cfg", (), {"models_dir": tmp_path / "models", "data_dir": tmp_path / "data",
                           "default_model": "minilm"})()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    return EngineState(cfg)


def test_configs_isolate_one_feature_each():
    base = ge.CONFIGS["baseline"]
    assert not any(base.values())
    for name, cfg in ge.CONFIGS.items():
        if name not in ("baseline", "all", "+graph_slot"):
            assert sum(cfg.values()) == 1, name
    assert all(ge.CONFIGS["all"].values())
    # graph slot is measured against the offline default (mmr_only): exactly
    # one feature differs
    diff = {k for k in set(ge.CONFIGS["+graph_slot"]) | set(ge.CONFIGS["mmr_only"])
            if ge.CONFIGS["+graph_slot"].get(k, False) != ge.CONFIGS["mmr_only"].get(k, False)}
    assert diff == {"graph_slot"}


def test_context_recall_by_id_and_by_expanded_text():
    texts = {"g1": "A" * 400, "g2": "B" * 400}
    item = {"gold": ["g1", "g2"]}
    assert ge._context_recall(item, [{"id": "g1", "text": "x"}], texts) == 0.5
    # g2 only present inside a parent-expanded neighbor's widened text
    widened = [{"id": "other", "text": "prefix " + "B" * 400}]
    assert ge._context_recall(item, widened, texts) == 0.5
    assert ge._context_recall({"gold": []}, [], texts) is None


def test_parse_json_tolerates_surrounding_text():
    assert ge._parse_json('Sure:\n{"score": 2, "refused": false}\nDone') == {"score": 2, "refused": False}
    assert ge._parse_json("no json here") is None


def test_judge_retries_unparseable_then_scores(state, monkeypatch):
    outputs = iter(["garbage", '{"score": 1, "refused": false}'])
    monkeypatch.setattr(ge, "_cloud", lambda st, p: next(outputs))
    assert ge.judge(state, "q", "ref", "some answer") == {"score": 1, "refused": False}


def test_judge_empty_answer_is_zero_without_cloud(state, monkeypatch):
    monkeypatch.setattr(ge, "_cloud", lambda st, p: pytest.fail("no judge call for empty answer"))
    assert ge.judge(state, "q", "ref", "  ")["score"] == 0


def test_summarize_math():
    rows = [
        {"type": "single", "score": 2, "refused": False, "context_recall": 1.0, "retrieval_s": 1, "answer_s": 2, "context_chars": 100},
        {"type": "single", "score": 0, "refused": True, "context_recall": 0.0, "retrieval_s": 1, "answer_s": 2, "context_chars": 100},
        {"type": "multi", "score": 1, "refused": False, "context_recall": 0.5, "retrieval_s": 1, "answer_s": 2, "context_chars": 100},
        {"type": "ooc", "score": 2, "refused": True, "context_recall": None, "retrieval_s": 1, "answer_s": 2, "context_chars": 100},
    ]
    s = ge.summarize(rows)
    assert s["correctness_single"] == 0.5 and s["correctness_multi"] == 0.5
    assert s["accuracy_answerable"] == round(1 / 3, 3)
    assert s["false_refusal_rate"] == round(1 / 3, 3)
    assert s["ooc_refusal_rate"] == 1.0
    assert s["context_recall_multi"] == 0.5


def test_multi_pairs_use_shared_graph_entity_and_skip_neighbors(state):
    chunks = [{"id": f"c{i}", "source": "s", "text": f"t{i}"} for i in range(8)]
    d = state.config.data_dir / "kgraph"
    d.mkdir(parents=True)
    con = sqlite3.connect(str(d / "corp.db"))
    con.executescript(
        "CREATE TABLE entities(id INTEGER PRIMARY KEY, name TEXT, norm TEXT);"
        "CREATE TABLE triples(id INTEGER PRIMARY KEY, subject_id INT, relation TEXT, object_id INT, chunk_id TEXT, source TEXT, created_at REAL);"
        "INSERT INTO entities VALUES (1,'Pedro','pedro'),(2,'Juan','juan');"
        # Pedro in c0, c1 (neighbors) and c5; Juan only in c2
        "INSERT INTO triples VALUES (1,1,'r',2,'c0','s',0),(2,1,'r',2,'c1','s',0),(3,1,'r',2,'c5','s',0);"
    )
    con.commit(); con.close()
    pairs = ge._multi_pairs(state, "corp", chunks, random.Random(1), n=1, exclude=set())
    assert pairs
    ent, a, b = pairs[0]
    assert abs(int(a["id"][1:]) - int(b["id"][1:])) >= 3


def test_judge_regex_fallback_when_not_json(state, monkeypatch):
    monkeypatch.setattr(ge, "_cloud", lambda st, p: "score: 2, refused = false (the answer matches)")
    assert ge.judge(state, "q", "ref", "ans") == {"score": 2, "refused": False}


def test_summarize_excludes_apparatus_failures():
    ok = {"type": "single", "score": 2, "refused": False, "context_recall": 1.0,
          "retrieval_s": 1, "answer_s": 1, "context_chars": 1}
    failed = {**ok, "score": 0, "context_recall": 0.0, "judge_error": "empty answer", "excluded": True}
    s = ge.summarize([ok, failed])
    assert s["correctness_single"] == 1.0  # failure not counted as a wrong answer
    assert s["excluded"] == 1
    assert s["context_recall_single"] == 0.5  # retrieval metric still covers every row


def test_cloud_retries_transient_failures(state, monkeypatch):
    import src.core.cloud_chat as cc

    calls = []

    def flaky(settings, prompt, max_tokens=None):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("cloud provider request failed: RemoteDisconnected")
        return "ok"

    monkeypatch.setattr(cc, "cloud_generate", flaky)
    monkeypatch.setattr(ge, "CLOUD_RETRY_WAITS", (0, 0, 0))
    assert ge._cloud(state, "p") == "ok" and len(calls) == 3


def test_judge_provider_down_is_excluded_not_crash(state, monkeypatch):
    def down(st, p):
        raise RuntimeError("down")

    monkeypatch.setattr(ge, "_cloud", down)
    assert "judge call failed" in ge.judge(state, "q", "r", "a")["judge_error"]
