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

"""Training- and inference-device selection.

Honest hardware reality, established by direct measurement, not assumption:

TRAINING — always CPU:
  * NPU (AMD Vitis AI / XDNA) — inference-only, on every known AMD software
    stack. There is no backprop path through it. It can never train anything,
    regardless of what's installed.
  * GPU (DirectML) — TRIED AND REJECTED for training. It measurably speeds
    up raw matmuls at gpt2-shaped dimensions (~1.4x, benchmarked on this
    hardware), but two problems make it unsafe here: (1) installing
    `torch-directml` silently downgrades `torch` to an old pinned version
    (2.4.1), which breaks other dependencies (sentence-transformers/RAG
    required >=2.5 and stopped working); (2) the TinyGPT training loop
    produced `loss=nan` on the DirectML backend — a real numerical
    correctness failure, not just a performance question.

INFERENCE (via ONNX, see scripts/export_onnx.py) — a separate story:
  * GPU (DirectML) — TRIED AND WORKS, but not worth it here. Exporting with
    torch's legacy TorchScript-based exporter (dynamo=False — the newer
    default exporter produces a graph that crashes DmlExecutionProvider,
    both at runtime and at session-init, verified directly) produces a
    correct graph (logits match CPU output; argmax identical). But measured
    end-to-end generation is ~4.3x SLOWER than CPU (2.82s vs 0.64s for 30
    greedy-decoded tokens) — this engine's workload (small models,
    single-sequence, no batching, no KV-cache reuse) is dominated by
    GPU data-transfer and kernel-launch overhead, not compute, on an
    integrated GPU. Not recommended as a default.
  * NPU — would need the RAI wheels installed (untested) plus INT8/INT4
    quantization on top of export. Given the GPU result above, and that
    NPUs typically need even more batching/sustained load than a GPU to
    amortize overhead, treat this as exploratory, not an expected win.

NVIDIA CUDA — a different case from DirectML (no project-specific rejection
reason — no loss=nan, no dependency downgrade), but NOT a zero-install case:
  * CORRECTED after live testing on a real CUDA machine (see
    docs/CLUSTER.md/[[project_sysmeaiengine_windows10_vcredist_bug]]): this
    file previously claimed a plain `pip install torch` always includes
    CUDA support automatically. That's wrong — confirmed live, `pip install
    torch` on a real machine with an NVIDIA GPU resolved to a CPU-only
    build (`torch.__version__` reported "...+cpu"). Getting a CUDA build
    requires installing from PyTorch's own CUDA package index instead:
      pip install torch --index-url https://download.pytorch.org/whl/cu121
    (or whichever CUDA tag matches the GPU driver — see pytorch.org's
    install selector for the current one). `requirements.txt` documents
    this; `scripts/setup_node.ps1` does not yet auto-detect a GPU and pick
    the CUDA index automatically — that's a real gap, not handled today.
  * Once a CUDA build is actually installed, the rest of the story here
    still holds: `torch.cuda.is_available()` simply reports False when
    there's no NVIDIA GPU/driver (safe no-op on the AMD hardware this
    engine was originally developed on), and there's nothing else
    project-specific to reject the way DirectML training was rejected.
  * Still not independently *benchmarked* for training/inference speed on
    real NVIDIA hardware — only the install/detection story has real
    evidence behind it so far. `device_report()["gpu_cuda"]` and
    `python -m src.cli metrics` show what was actually detected on a given
    machine.
"""
from __future__ import annotations

from typing import Any


def directml_available() -> bool:
    try:
        import torch_directml  # noqa: F401

        return True
    except ImportError:
        return False


def onnx_directml_available() -> bool:
    """DirectML for ONNX *inference* — independent of torch_directml, which
    is deliberately not installed (see module docstring)."""
    try:
        import onnxruntime as ort  # type: ignore

        return "DmlExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cuda_info() -> dict[str, Any]:
    """Best-effort detail on the first CUDA device, for the dashboard/CLI."""
    if not cuda_available():
        return {"available": False}
    try:
        import torch

        idx = 0
        free_b, total_b = torch.cuda.mem_get_info(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(idx),
            "capability": f"{major}.{minor}",
            "memory_total_gb": round(total_b / 1e9, 1),
            "memory_free_gb": round(free_b / 1e9, 1),
        }
    except Exception:
        return {"available": True, "detail_error": True}


def resolve_hf_device_map(requested: str = "auto") -> str:
    """Resolve a requested selector ('auto' | 'cpu' | 'cuda') to a concrete
    `device_map` value for `AutoModelForCausalLM.from_pretrained()` /
    `Llama(n_gpu_layers=...)` callers. 'auto' means "use CUDA if a real GPU
    is present, else CPU" — safe on any machine since it never requires
    CUDA to be there. Explicitly requesting 'cuda' on a machine without one
    fails loudly instead of silently falling back, so a misconfigured
    registry entry doesn't quietly run on the wrong device.
    """
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not cuda_available():
            raise RuntimeError(
                "device='cuda' requested but no CUDA-capable GPU was detected "
                "(torch.cuda.is_available() is False). Use 'auto' or 'cpu'."
            )
        return "cuda"
    if requested == "auto":
        return "cuda" if cuda_available() else "cpu"
    raise ValueError(f"Unknown device '{requested}'. Use 'auto', 'cpu', or 'cuda'.")


def get_torch_device(requested: str = "auto") -> tuple[str, Any]:
    """Returns (label, torch.device) for the raw-PyTorch TinyGPT training loop.

    'auto'/'cuda' use a real NVIDIA GPU when present — see the CUDA section
    of the module docstring for why this is safe where DirectML wasn't.
    """
    import torch

    if requested == "directml":
        raise RuntimeError(
            "DirectML training is disabled: it produced loss=nan in testing and "
            "installing torch-directml breaks other dependencies. See device_select.py."
        )
    if requested == "cuda":
        if not cuda_available():
            raise RuntimeError(
                "device='cuda' requested but no CUDA-capable GPU was detected "
                "(torch.cuda.is_available() is False)."
            )
        return "cuda", torch.device("cuda")
    if requested == "auto" and cuda_available():
        return "cuda", torch.device("cuda")
    return "cpu", torch.device("cpu")


def device_report() -> dict[str, Any]:
    """Summary for the UI: what's real, what isn't, and why."""
    cuda = cuda_info()
    return {
        "npu": {
            "trainable": False,
            "reason": "AMD Vitis AI / XDNA execution provider is inference-only on every "
            "known software stack — there is no backward pass through it.",
        },
        "gpu_directml": {
            "installed": directml_available(),
            "trainable_paths": [],
            "not_trainable_paths": ["minilm (TinyGPT)", "hf (gpt2 and other HF causal LMs)"],
            "reason": "Tried and rejected for training: produced loss=nan when training "
            "TinyGPT on this hardware, and installing torch-directml downgrades torch in "
            "a way that breaks other features (RAG's sentence-transformers needs a newer "
            "torch). All training runs on CPU.",
        },
        "gpu_cuda": {
            **cuda,
            "trainable_paths": ["minilm (TinyGPT)", "hf (SFT/LoRA fine-tuning)"] if cuda["available"] else [],
            "reason": "Standard NVIDIA CUDA via the already-installed torch build — no "
            "separate package (unlike torch-directml) and no known correctness or "
            "dependency issues in this project. Used automatically by device='auto' "
            "(the default) whenever a CUDA GPU is detected; falls back to CPU otherwise.",
        },
        "onnx_inference": {
            "gpu_available": onnx_directml_available(),
            "reason": "GPU (DirectML) inference via ONNX works (see scripts/export_onnx.py) "
            "and is correct, but measured ~4.3x SLOWER than CPU for this engine's actual "
            "workload (small models, single-sequence greedy decoding, no batching, no "
            "KV-cache) — overhead dominates on an integrated GPU. Not recommended by "
            "default; exists for experimentation.",
        },
    }
