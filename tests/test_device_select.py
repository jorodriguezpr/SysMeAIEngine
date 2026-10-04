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

"""device_select.py tests.

This dev machine has no NVIDIA GPU (AMD Radeon 860M iGPU only), so the
'no CUDA present' assertions below are real, unmocked verifications against
actual hardware — not guesses. The 'what if CUDA were available' cases are
monkeypatched since there's no way to exercise them for real here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import device_select  # noqa: E402


def test_cuda_available_returns_bool_without_crashing():
    assert isinstance(device_select.cuda_available(), bool)


def test_cuda_info_reports_unavailable_on_this_machine():
    # Real assertion: this dev machine is AMD-only, no CUDA GPU present.
    info = device_select.cuda_info()
    assert info == {"available": False}


def test_resolve_hf_device_map_cpu_is_always_cpu():
    assert device_select.resolve_hf_device_map("cpu") == "cpu"


def test_resolve_hf_device_map_auto_falls_back_to_cpu_here():
    # Real assertion: no CUDA GPU on this machine, so 'auto' must resolve to cpu.
    assert device_select.resolve_hf_device_map("auto") == "cpu"


def test_resolve_hf_device_map_cuda_raises_when_unavailable():
    # Real assertion: requesting cuda explicitly on this GPU-less machine
    # must fail loudly, not silently fall back.
    import pytest

    with pytest.raises(RuntimeError):
        device_select.resolve_hf_device_map("cuda")


def test_resolve_hf_device_map_unknown_raises_value_error():
    import pytest

    with pytest.raises(ValueError):
        device_select.resolve_hf_device_map("tpu")


def test_resolve_hf_device_map_cuda_when_available(monkeypatch):
    monkeypatch.setattr(device_select, "cuda_available", lambda: True)
    assert device_select.resolve_hf_device_map("cuda") == "cuda"
    assert device_select.resolve_hf_device_map("auto") == "cuda"


def test_get_torch_device_cpu_here():
    label, device = device_select.get_torch_device("cpu")
    assert label == "cpu"
    assert str(device) == "cpu"


def test_get_torch_device_auto_falls_back_to_cpu_here():
    # Real assertion: no CUDA GPU on this machine.
    label, device = device_select.get_torch_device("auto")
    assert label == "cpu"


def test_get_torch_device_cuda_raises_here():
    import pytest

    with pytest.raises(RuntimeError):
        device_select.get_torch_device("cuda")


def test_get_torch_device_directml_still_disabled():
    import pytest

    with pytest.raises(RuntimeError):
        device_select.get_torch_device("directml")


def test_device_report_includes_gpu_cuda_section():
    report = device_select.device_report()
    assert "gpu_cuda" in report
    # Real assertion: unavailable on this machine, and therefore not listed
    # as a trainable path.
    assert report["gpu_cuda"]["available"] is False
    assert report["gpu_cuda"]["trainable_paths"] == []


def test_device_report_gpu_cuda_lists_trainable_paths_when_available(monkeypatch):
    monkeypatch.setattr(device_select, "cuda_info", lambda: {"available": True})
    report = device_select.device_report()
    assert report["gpu_cuda"]["trainable_paths"] == [
        "minilm (TinyGPT)",
        "hf (SFT/LoRA fine-tuning)",
    ]
