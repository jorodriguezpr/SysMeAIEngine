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

"""Tests for src.core.cloud_distill -- generating QA training rows via the
cloud provider instead of the cheap heuristic. Mocks cloud_generate so this
stays fast/offline (this hits a real paid API otherwise)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import cloud_distill as cloud_distill_mod  # noqa: E402


def _fake_state(tmp_path):
    cfg = type("Cfg", (), {"models_dir": tmp_path})()
    return type("FakeState", (), {"config": cfg})()


def _enable_cloud(tmp_path):
    from src.core.cloud_settings import save_cloud_settings

    save_cloud_settings(
        _fake_state(tmp_path), enabled=True, provider="anthropic",
        model="test-cloud-model", api_key="sk-test",
    )
    return _fake_state(tmp_path)


# ── ensure_cloud_ready ─────────────────────────────────────────────────────
def test_ensure_cloud_ready_rejects_when_disabled(tmp_path):
    with pytest.raises(ValueError, match="not enabled"):
        cloud_distill_mod.ensure_cloud_ready(_fake_state(tmp_path))


def test_ensure_cloud_ready_passes_when_configured(tmp_path):
    state = _enable_cloud(tmp_path)
    settings = cloud_distill_mod.ensure_cloud_ready(state)
    assert settings["provider"] == "anthropic"


# ── _parse_qa_pairs ────────────────────────────────────────────────────────
def test_parse_qa_pairs_single():
    text = "Question: Who is Nicodemo?\nAnswer: A biblical figure."
    assert cloud_distill_mod._parse_qa_pairs(text) == [("Who is Nicodemo?", "A biblical figure.")]


def test_parse_qa_pairs_multiple():
    text = (
        "Question: Who is Nicodemo?\nAnswer: A biblical figure.\n"
        "Question: what did nicodemo do?\nAnswer: He defended Jesus before Pilate.\n"
        "Question: In which language did Nicodemo write?\nAnswer: Hebrew."
    )
    pairs = cloud_distill_mod._parse_qa_pairs(text)
    assert pairs == [
        ("Who is Nicodemo?", "A biblical figure."),
        ("what did nicodemo do?", "He defended Jesus before Pilate."),
        ("In which language did Nicodemo write?", "Hebrew."),
    ]


def test_parse_qa_pairs_case_insensitive():
    text = "question: ¿Quién es?\nanswer: Un personaje."
    assert cloud_distill_mod._parse_qa_pairs(text) == [("¿Quién es?", "Un personaje.")]


def test_parse_qa_pairs_rejects_malformed():
    assert cloud_distill_mod._parse_qa_pairs("I refuse to answer that.") == []


# ── truncation handling (real bug: max_tokens cut a multi-pair response
# off mid-answer, and truncated fragments made it into a real dataset) ────
def test_parse_qa_pairs_drops_truncated_final_pair():
    text = (
        "Question: Who is Nicodemo?\nAnswer: A biblical figure.\n"
        "Question: What did he do?\nAnswer: He descended and luch"  # cut off mid-word
    )
    pairs = cloud_distill_mod._parse_qa_pairs(text)
    assert pairs == [("Who is Nicodemo?", "A biblical figure.")]  # complete pair kept, cut-off one dropped


def test_parse_qa_pairs_strips_dangling_question_label_from_truncated_answer():
    # confirmed live: a response cut off right as the next "Question" label
    # started left it stuck onto the end of the previous, otherwise-complete
    # answer instead of a clean boundary.
    text = "Question: Why?\nAnswer: Porque Dios tuvo misericordia de nosotros.\n\nQuestion"
    pairs = cloud_distill_mod._parse_qa_pairs(text)
    assert pairs == [("Why?", "Porque Dios tuvo misericordia de nosotros.")]


def test_parse_qa_pairs_keeps_complete_final_pair():
    text = "Question: Q1?\nAnswer: A1.\nQuestion: Q2?\nAnswer: A2 ends properly."
    pairs = cloud_distill_mod._parse_qa_pairs(text)
    assert len(pairs) == 2
    assert pairs[1] == ("Q2?", "A2 ends properly.")


def test_looks_truncated():
    assert cloud_distill_mod._looks_truncated("ella res") is True
    assert cloud_distill_mod._looks_truncated("") is True
    assert cloud_distill_mod._looks_truncated("A complete sentence.") is False
    assert cloud_distill_mod._looks_truncated('A quoted sentence."') is False


# ── build_cloud_qa_dataset ─────────────────────────────────────────────────
def test_build_cloud_qa_dataset_empty_chunks(tmp_path):
    state = _enable_cloud(tmp_path)
    assert cloud_distill_mod.build_cloud_qa_dataset(state, []) == []


def test_build_cloud_qa_dataset_happy_path_single_pair_response(tmp_path):
    state = _enable_cloud(tmp_path)
    chunks = [
        {"id": "a", "text": "Nicodemo fue ante Pilato."},
        {"id": "b", "text": "El evangelio narra la resurreccion."},
    ]
    with patch.object(
        cloud_distill_mod, "cloud_generate",
        return_value="Question: Quien fue ante Pilato?\nAnswer: Nicodemo.",
    ):
        rows = cloud_distill_mod.build_cloud_qa_dataset(state, chunks)
    assert len(rows) == 2
    assert rows[0] == {"question": "Quien fue ante Pilato?", "answer": "Nicodemo.", "id": "a-0"}
    assert rows[1]["id"] == "b-0"


def test_build_cloud_qa_dataset_extracts_multiple_pairs_per_chunk(tmp_path):
    """The core fix this session: one real API call per chunk, but several
    differently-phrased QA rows extracted from that single response --
    verifies phrasing diversity comes from more rows, not more API calls."""
    state = _enable_cloud(tmp_path)
    chunks = [{"id": "a", "text": "Nicodemo fue ante Pilato."}]
    multi_response = (
        "Question: Quien fue ante Pilato?\nAnswer: Nicodemo.\n"
        "Question: que hizo nicodemo?\nAnswer: Se presento ante Pilato.\n"
        "Question: ante quien se presento Nicodemo?\nAnswer: Ante Pilato."
    )
    with patch.object(cloud_distill_mod, "cloud_generate", return_value=multi_response) as mock_gen:
        rows = cloud_distill_mod.build_cloud_qa_dataset(state, chunks)
    mock_gen.assert_called_once()  # one API call...
    assert len(rows) == 3  # ...but three distinct training rows
    assert {r["id"] for r in rows} == {"a-0", "a-1", "a-2"}
    assert len({r["question"] for r in rows}) == 3  # all differently phrased


def test_build_cloud_qa_dataset_skips_unparseable_chunks(tmp_path):
    state = _enable_cloud(tmp_path)
    chunks = [{"id": "a", "text": "some text"}, {"id": "b", "text": "more text"}]
    with patch.object(cloud_distill_mod, "cloud_generate", side_effect=["not in the right format", "Question: Q?\nAnswer: A."]):
        rows = cloud_distill_mod.build_cloud_qa_dataset(state, chunks)
    assert len(rows) == 1
    assert rows[0]["id"] == "b-0"


def test_build_cloud_qa_dataset_skips_on_api_error(tmp_path):
    state = _enable_cloud(tmp_path)
    chunks = [{"id": "a", "text": "some text"}, {"id": "b", "text": "more text"}]
    with patch.object(cloud_distill_mod, "cloud_generate", side_effect=[RuntimeError("API down"), "Question: Q?\nAnswer: A."]):
        rows = cloud_distill_mod.build_cloud_qa_dataset(state, chunks)
    assert len(rows) == 1
    assert rows[0]["id"] == "b-0"


def test_build_cloud_qa_dataset_requests_the_larger_qa_token_budget(tmp_path):
    """Regression test: this used to rely on the interactive-chat max_tokens
    setting for a batch-generation call, and got real answers truncated
    mid-sentence as a result -- must always ask for the larger QA-specific
    budget regardless of what's saved in chat settings."""
    state = _enable_cloud(tmp_path)
    chunks = [{"id": "a", "text": "some text"}]
    with patch.object(cloud_distill_mod, "cloud_generate", return_value="Question: Q?\nAnswer: A.") as mock_gen:
        cloud_distill_mod.build_cloud_qa_dataset(state, chunks)
    _, kwargs = mock_gen.call_args
    assert kwargs["max_tokens"] == cloud_distill_mod.QA_GENERATION_MAX_TOKENS


def test_build_cloud_qa_dataset_calls_progress_callback(tmp_path):
    state = _enable_cloud(tmp_path)
    chunks = [{"id": str(i), "text": f"text {i}"} for i in range(3)]
    calls = []
    with patch.object(cloud_distill_mod, "cloud_generate", return_value="Question: Q?\nAnswer: A."):
        cloud_distill_mod.build_cloud_qa_dataset(state, chunks, on_progress=lambda d, t, f: calls.append((d, t, f)))
    assert calls == [(1, 3, 0), (2, 3, 0), (3, 3, 0)]


def test_build_cloud_qa_dataset_detects_language_once_from_sample(tmp_path):
    state = _enable_cloud(tmp_path)
    chunks = [{"id": "a", "text": "El evangelio de Nicodemo es un texto apocrifo en español."}]
    captured_prompts = []

    def fake_generate(settings, prompt, **kwargs):
        captured_prompts.append(prompt)
        return "Question: Q?\nAnswer: A."

    with patch.object(cloud_distill_mod, "cloud_generate", side_effect=fake_generate):
        cloud_distill_mod.build_cloud_qa_dataset(state, chunks)
    assert "in Spanish" in captured_prompts[0]
