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

"""Tests for cloud_settings.py (persistence) and cloud_chat.py (dispatch to
Anthropic/OpenAI/Ollama Cloud). HTTP calls are mocked -- these hit real paid
APIs otherwise."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import cloud_chat as cloud_chat_mod  # noqa: E402
from src.core import cloud_settings as cloud_settings_mod  # noqa: E402


def _fake_state(tmp_path):
    cfg = type("Cfg", (), {"models_dir": tmp_path})()
    return type("FakeState", (), {"config": cfg})()


# ── cloud_settings ─────────────────────────────────────────────────────────
def test_get_cloud_settings_returns_defaults_when_no_file(tmp_path):
    settings = cloud_settings_mod.get_cloud_settings(_fake_state(tmp_path))
    assert settings["enabled"] is False
    assert settings["provider"] == "anthropic"


def test_save_and_get_cloud_settings_roundtrip(tmp_path):
    state = _fake_state(tmp_path)
    cloud_settings_mod.save_cloud_settings(state, enabled=True, provider="openai", model="gpt-4o", api_key="sk-test123")
    settings = cloud_settings_mod.get_cloud_settings(state)
    assert settings["enabled"] is True
    assert settings["provider"] == "openai"
    assert settings["model"] == "gpt-4o"
    assert settings["api_key"] == "sk-test123"


def test_save_cloud_settings_ignores_none_fields(tmp_path):
    state = _fake_state(tmp_path)
    cloud_settings_mod.save_cloud_settings(state, enabled=True, model="test-cloud-model")
    cloud_settings_mod.save_cloud_settings(state, model=None)  # should not clear the saved model
    settings = cloud_settings_mod.get_cloud_settings(state)
    assert settings["model"] == "test-cloud-model"


def test_masked_hides_real_key():
    out = cloud_settings_mod.masked({**cloud_settings_mod.DEFAULTS, "api_key": "sk-abcdefgh1234"})
    assert out["api_key"] != "sk-abcdefgh1234"
    assert out["api_key"].endswith("1234")
    assert out["api_key_set"] is True


def test_masked_empty_key():
    out = cloud_settings_mod.masked(dict(cloud_settings_mod.DEFAULTS))
    assert out["api_key"] == ""
    assert out["api_key_set"] is False


# ── cloud_chat ─────────────────────────────────────────────────────────────
def _settings(**overrides):
    return {**cloud_settings_mod.DEFAULTS, "enabled": True, "api_key": "test-key", "model": "test-model", **overrides}


def test_cloud_generate_rejects_when_disabled():
    with pytest.raises(ValueError):
        cloud_chat_mod.cloud_generate(_settings(enabled=False), "hello")


def test_cloud_generate_rejects_missing_api_key():
    with pytest.raises(ValueError):
        cloud_chat_mod.cloud_generate(_settings(api_key=""), "hello")


def test_cloud_generate_rejects_missing_model():
    with pytest.raises(ValueError):
        cloud_chat_mod.cloud_generate(_settings(model=""), "hello")


def test_cloud_generate_rejects_unknown_provider():
    with pytest.raises(ValueError):
        cloud_chat_mod.cloud_generate(_settings(provider="not-a-real-provider"), "hello")


def _fake_response(json_body, ok=True, status_code=200, text=""):
    r = MagicMock()
    r.ok = ok
    r.status_code = status_code
    r.text = text
    r.json.return_value = json_body
    return r


def test_cloud_generate_anthropic_parses_response():
    settings = _settings(provider="anthropic")
    fake = _fake_response({"content": [{"type": "text", "text": "hola mundo"}]})
    with patch.object(cloud_chat_mod.requests, "post", return_value=fake) as post:
        out = cloud_chat_mod.cloud_generate(settings, "hi")
    assert out == "hola mundo"
    args, kwargs = post.call_args
    assert "anthropic.com" in args[0]
    assert kwargs["headers"]["x-api-key"] == "test-key"


def test_cloud_generate_openai_parses_response():
    settings = _settings(provider="openai")
    fake = _fake_response({"choices": [{"message": {"content": "hello there"}}]})
    with patch.object(cloud_chat_mod.requests, "post", return_value=fake) as post:
        out = cloud_chat_mod.cloud_generate(settings, "hi")
    assert out == "hello there"
    args, kwargs = post.call_args
    assert "api.openai.com" in args[0]
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_cloud_generate_ollama_cloud_parses_response():
    settings = _settings(provider="ollama_cloud")
    fake = _fake_response({"choices": [{"message": {"content": "respuesta"}}]})
    with patch.object(cloud_chat_mod.requests, "post", return_value=fake) as post:
        out = cloud_chat_mod.cloud_generate(settings, "hi")
    assert out == "respuesta"
    args, kwargs = post.call_args
    assert "ollama.com" in args[0]


def test_cloud_generate_respects_base_url_override():
    settings = _settings(provider="openai", base_url="https://my-proxy.example.com/v1")
    fake = _fake_response({"choices": [{"message": {"content": "ok"}}]})
    with patch.object(cloud_chat_mod.requests, "post", return_value=fake) as post:
        cloud_chat_mod.cloud_generate(settings, "hi")
    args, _ = post.call_args
    assert args[0].startswith("https://my-proxy.example.com/v1")


def test_cloud_generate_includes_system_prompt():
    settings = _settings(provider="openai", system_prompt="Always answer in Spanish.")
    fake = _fake_response({"choices": [{"message": {"content": "ok"}}]})
    with patch.object(cloud_chat_mod.requests, "post", return_value=fake) as post:
        cloud_chat_mod.cloud_generate(settings, "hi")
    _, kwargs = post.call_args
    assert kwargs["json"]["messages"][0] == {"role": "system", "content": "Always answer in Spanish."}


def test_cloud_generate_max_tokens_override_beats_settings():
    """Regression test: a real cloud_qa training run got truncated mid-answer
    because it used the chat settings' max_tokens (1024) for a batch use
    case that needed much more room -- callers must be able to ask for more
    per-call without touching the user's saved chat settings."""
    settings = _settings(provider="anthropic", max_tokens=512)
    fake = _fake_response({"content": [{"type": "text", "text": "ok"}]})
    with patch.object(cloud_chat_mod.requests, "post", return_value=fake) as post:
        cloud_chat_mod.cloud_generate(settings, "hi", max_tokens=2048)
    _, kwargs = post.call_args
    assert kwargs["json"]["max_tokens"] == 2048


def test_cloud_generate_raises_on_http_error():
    settings = _settings(provider="anthropic")
    fake = _fake_response({}, ok=False, status_code=401, text="invalid api key")
    with patch.object(cloud_chat_mod.requests, "post", return_value=fake):
        with pytest.raises(RuntimeError, match="401"):
            cloud_chat_mod.cloud_generate(settings, "hi")
