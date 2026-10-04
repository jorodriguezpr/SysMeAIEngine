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

"""ONNX Runtime with the AMD Ryzen AI NPU execution provider (Vitis AI EP)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class ONNXXNPUProvider:
    name = "onnx_npu"

    def load(self, name: str, registry_path: Path, **kwargs: Any) -> tuple[Any, Any]:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise ImportError(
                "onnxruntime-vitisai is required for the Ryzen AI NPU. "
                "See requirements-amd.txt and https://ryzenai.amd.com"
            ) from e
        model_path = kwargs.get("path")
        if not model_path:
            raise ValueError("onnx_npu requires an .onnx path (INT8/INT4 quantized)")
        sess = ort.InferenceSession(
            model_path,
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            provider_options=[{"config_file": kwargs.get("vitis_config")}, {}],
        )
        # the registry only ever passes `path` — model_loader.load_model()
        # has no field for a second "tokenizer" path — so fall back to the
        # model's own directory, where scripts/export_onnx.py always saves
        # the tokenizer alongside model.onnx
        tok_path = kwargs.get("tokenizer") or str(Path(model_path).parent)
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tok_path)
        except Exception:
            tok = None
        return sess, tok

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        import numpy as np

        if tokenizer is None:
            raise ValueError("onnx_npu generate requires a tokenizer")
        ids = tokenizer(prompt, return_tensors="np").input_ids
        for _ in range(max_new_tokens):
            out = model.run(None, {"input_ids": ids})[0]
            next_id = int(np.argmax(out[0, -1]))
            ids = np.concatenate([ids, [[next_id]]], axis=1)
            if next_id == tokenizer.eos_token_id:
                break
        return tokenizer.decode(ids[0], skip_special_tokens=True)

    def embed(
        self, model: Any, tokenizer: Any, text: str | list[str]
    ) -> list[list[float]]:
        raise NotImplementedError("Embeddings not implemented for onnx_npu yet")

    def info(self, model: Any) -> dict[str, Any]:
        return {
            "kind": "onnx-npu (Ryzen AI)",
            "providers": model.get_providers(),
        }
