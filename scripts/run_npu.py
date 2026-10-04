#!/usr/bin/env python
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

"""Run ONNX inference on the Ryzen AI NPU via the RAI conda env (or fall back to local).

Two modes:

  1. **Auto-detect RAI env** and re-exec this script with that python.exe. Use this when
     SysMeAIEngine lives in its own .venv and the NPU provider only exists inside the
     RAI conda env.

  2. **Direct mode** — if `onnxruntime-vitisai` is somehow already installed locally
     (e.g. you ran `engine serve` from the RAI conda env), just run the model here.

Usage:
    python scripts/run_npu.py models/checkpoints/proyecto-adapter/minilm-static.onnx
    python scripts/run_npu.py <model.onnx> --prompt "Read the order/parts form and answer:"
    python scripts/run_npu.py <model.onnx> --no-reexec       # don't auto-switch pythons
    python scripts/run_npu.py <model.onnx> --config path\\to\\vitis_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ── RAI discovery ──────────────────────────────────────────────────────────
DEFAULT_RAI_DIR = Path(r"C:\Program Files\RyzenAI")
RAI_ENV_NAME_PREFIX = "ryzen-ai"


def find_rai_python() -> Path | None:
    """Find the python.exe for the most recent Ryzen AI conda env."""
    if not DEFAULT_RAI_DIR.exists():
        return None
    # pattern: C:\Program Files\RyzenAI\<version>\envs\<env-name>\python.exe
    envs: list[Path] = []
    for ver_dir in sorted(DEFAULT_RAI_DIR.iterdir()):
        envs_root = ver_dir / "envs"
        if not envs_root.exists():
            continue
        for env_dir in sorted(envs_root.iterdir()):
            if env_dir.name.startswith(RAI_ENV_NAME_PREFIX):
                exe = env_dir / "python.exe"
                if exe.exists():
                    envs.append(exe)
    return envs[-1] if envs else None  # last = highest version, alphabetically


def in_rai_env() -> bool:
    """Are we currently running inside the RAI conda env?"""
    return "ryzen-ai" in (os.environ.get("CONDA_DEFAULT_ENV") or "").lower()


def has_vitis_provider() -> bool:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        return False
    return "VitisAIExecutionProvider" in ort.get_available_providers()


# ── inference ─────────────────────────────────────────────────────────────
def run_inference(onnx_path: Path, prompt: str, config: Path | None, verbose: bool) -> int:
    import numpy as np
    import onnxruntime as ort

    providers = ["VitisAIExecutionProvider", "CPUExecutionProvider"]
    provider_options = [{"config_file": str(config)} if config else {}, {}]

    if verbose:
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 0
        sess = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=providers,
            provider_options=provider_options,
        )
    else:
        sess = ort.InferenceSession(
            str(onnx_path),
            providers=providers,
            provider_options=provider_options,
        )

    print(f"[NPU] providers active: {sess.get_providers()}")

    # byte-level tokenizer matching TinyGPT — matches ONNX export
    ids = [ord(c) if ord(c) < 256 else 32 for c in prompt[:128].ljust(128)]
    X = np.array([ids], dtype=np.int64)
    logits = sess.run(["logits"], {"input_ids": X})[0]
    next_id = int(np.argmax(logits[0, -1]))
    print(f"[NPU] input_len={len(prompt)}  seq={X.shape[1]}  next_token_id={next_id}")
    if 32 <= next_id < 127:
        print(f"[NPU] next_token_char={chr(next_id)!r}")
    return 0


# ── main ───────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path, help="path to .onnx model (INT8/INT4 preferred)")
    p.add_argument("--prompt", default="Hello", help="prompt to run a single forward pass on")
    p.add_argument("--config", type=Path, default=None, help="Vitis AI config json")
    p.add_argument("--no-reexec", action="store_true", help="don't auto-switch to RAI python")
    p.add_argument("--verbose", action="store_true", help="enable onnxruntime verbose logs")
    args = p.parse_args()

    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 1

    # 1) already inside the RAI env → run directly
    if in_rai_env() or has_vitis_provider():
        return run_inference(args.model, args.prompt, args.config, args.verbose)

    if args.no_reexec:
        print("VitisAIExecutionProvider not available in `ort.get_available_providers()`.")
        print("Install the RAI conda env first (see README § AMD NPU).")
        return 1

    # 2) not in RAI env → re-exec with RAI python
    rai_python = find_rai_python()
    if not rai_python:
        print(
            "ERROR: Ryzen AI Python not found.\n"
            "Expected at: " + str(DEFAULT_RAI_DIR / "<version>\\envs\\ryzen-ai-*\\python.exe"),
            file=sys.stderr,
        )
        print(
            "Install the Ryzen AI Software first — see README § AMD NPU.",
            file=sys.stderr,
        )
        return 1

    print(f"[NPU] re-exec with {rai_python}")
    cmd = [str(rai_python), str(Path(__file__).resolve())] + [
        str(args.model),
        "--prompt", args.prompt,
        "--no-reexec",
    ]
    if args.config:
        cmd += ["--config", str(args.config)]
    if args.verbose:
        cmd.append("--verbose")
    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
