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

"""A model with weights left on the 'meta' device (never loaded) must be
refused at load time -- live, such a model silently generated multilingual
garbage for every prompt, and the only trace was "Cannot copy out of meta
tensor" on /ui/weights."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.providers.hf import _raise_if_unmaterialized  # noqa: E402


def test_fully_loaded_model_passes():
    _raise_if_unmaterialized(torch.nn.Linear(4, 4), "ok-model")


def test_meta_weights_are_refused_with_names():
    m = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with torch.device("meta"):
        m.append(torch.nn.Linear(4, 4))
    with pytest.raises(RuntimeError, match="no data") as e:
        _raise_if_unmaterialized(m, "broken-model")
    assert "1.weight" in str(e.value) and "broken-model" in str(e.value)
