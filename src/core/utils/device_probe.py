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

"""AMD CPU / GPU / NPU (Ryzen AI) probing — Windows only, no admin required.

Best-effort: returns 'unknown' / unavailable when information is missing.
"""
from __future__ import annotations

import platform
import subprocess
from typing import Any

import psutil


def _run(cmd: list[str], timeout: int = 6) -> str:
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return out.stdout + out.stderr
    except Exception:
        return ""


def probe_cpu() -> dict[str, Any]:
    name = platform.processor() or "unknown"
    return {
        "name": name,
        "logical": psutil.cpu_count(logical=True),
        "physical": psutil.cpu_count(logical=False),
        "usage_percent": psutil.cpu_percent(interval=0.05),
        "freq_mhz": getattr(psutil.cpu_freq(), "current", None),
    }


def probe_gpu() -> dict[str, Any]:
    """Detect GPU. NVIDIA (via torch/CUDA — already a dependency, no extra
    probing needed) takes priority when present; otherwise falls back to the
    AMD (DX) dxdiag probe below."""
    from ..device_select import cuda_info

    cuda = cuda_info()
    if cuda["available"]:
        return {"present": True, "vendor": "NVIDIA", **cuda}
    return _probe_amd_gpu()


def _probe_amd_gpu() -> dict[str, Any]:
    """Detect AMD (DX) GPUs via dxdiag — cheap probe."""
    sysinfo = _run(["dxdiag", "/t", "dxdiag.tmp"])
    # dxdiag writes to a file; read if we can
    text = ""
    try:
        with open("dxdiag.tmp", "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        pass
    lower = text.lower()
    is_amd = "amd" in lower or "radeon" in lower
    return {
        "present": is_amd,
        "vendor": "AMD" if is_amd else "unknown",
        "raw_hint": sysinfo[:120] if sysinfo else None,
    }


def probe_npu() -> dict[str, Any]:
    """Detect Ryzen AI NPU (AMD IPU/XDNA).

    Ryzen AI shows up in Device Manager as "AMD IPU" or "XDNA" device.
    We try pnputil enum and CIM Win32_PnPEntity for a driver-level check.
    """
    pnputil = _run(["pnputil", "/enum-devices"])
    cim = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.Name -match 'IPU|XDNA|Ryzen|AMD' } | "
            "Select-Object -ExpandProperty Name",
        ]
    )
    blob = (pnputil + "\n" + cim).lower()
    found = any(k in blob for k in ("amd ipu", "xdna", "ryzen ai", "npu"))
    return {
        "present": found,
        "kind": "Ryzen AI (XDNA)" if "xdna" in blob else ("Ryzen AI" if found else "none"),
        "hint": (cim or pnputil)[:180] or None,
    }


def probe_memory() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / 1e9, 1),
        "used_gb": round(vm.used / 1e9, 1),
        "available_gb": round(vm.available / 1e9, 1),  # psutil's own reclaimable-cache-aware estimate
        "percent": vm.percent,
    }


def probe_disk() -> dict[str, Any]:
    du = psutil.disk_usage("/")
    return {
        "total_gb": round(du.total / 1e9, 1),
        "used_gb": round(du.used / 1e9, 1),
        "percent": du.percent,
    }


def probe_devices() -> dict[str, Any]:
    return {
        "cpu": probe_cpu(),
        "gpu": probe_gpu(),
        "npu": probe_npu(),
        "memory": probe_memory(),
        "disk": probe_disk(),
    }
