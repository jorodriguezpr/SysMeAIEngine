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

"""Lightweight reference model that runs on any CPU.

Implements a *tiny* autoregressive transformer (~2M params) built with plain
torch so the whole engine is testable without downloading a real checkpoint.
Not meant for real generation quality — it exists so `--provider minilm`
works everywhere and the full pipeline (chat → FT → distillation → dashboard)
can be exercised offline.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Byte-level tokenizer ────────────────────────────────────────────────────
class _Tokenizer:
    vocab_size = 256
    eos_id = 10  # "\n"
    pad_id = 0

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, ids: list[int]) -> str:
        return bytes(b for b in ids if 0 <= b < 256).decode("utf-8", errors="ignore")


# ── Tiny GPT ────────────────────────────────────────────────────────────────
@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.MultiheadAttention(
            cfg.n_embd, cfg.n_head, batch_first=True, dropout=0.0
        )
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1
        )
        y = self.ln1(x)
        attn_out, _ = self.attn(y, y, y, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GPTConfig()
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.n_embd)
        self.pos_emb = nn.Embedding(self.cfg.block_size, self.cfg.n_embd)
        self.blocks = nn.ModuleList(Block(self.cfg) for _ in range(self.cfg.n_layer))
        self.ln_f = nn.LayerNorm(self.cfg.n_embd)
        self.head = nn.Linear(self.cfg.n_embd, self.cfg.vocab_size, bias=False)
        self.apply(self._init)
        self._tokenizer = _Tokenizer()

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # tokenizer shim (parquet-style duck-typing)
    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    @property
    def tokenizer(self) -> _Tokenizer:
        return self._tokenizer

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        T = min(T, self.cfg.block_size)
        idx = idx[:, -T:]
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate_ids(
        self, ids: list[int], max_new: int = 256, temperature: float = 0.7
    ) -> list[int]:
        self.eval()
        idx = torch.tensor([ids], dtype=torch.long)
        for _ in range(max_new):
            logits = self(idx)
            logits = logits[0, -1, :] / max(temperature, 1e-4)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)
            if next_id == self._tokenizer.eos_id:
                break
            idx = torch.tensor([ids], dtype=torch.long)
        return ids

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def tensor_stats(self) -> list[dict]:
        out = []
        for name, p in self.named_parameters():
            data = p.detach().float()
            out.append(
                {
                    "name": name,
                    "shape": list(p.shape),
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "min": float(data.min()),
                    "max": float(data.max()),
                    "sparsity": float((data.abs() < 1e-8).float().mean()),
                }
            )
        return out


# ── Provider wiring ─────────────────────────────────────────────────────────
class MiniLMProvider:
    name = "minilm"

    def load(self, name: str, registry_path: Path, **kwargs: Any) -> tuple[Any, Any]:
        """Create (or load) a TinyGPT keyed by model name. `registry_path` is the
        model dir under `models/weights/<name>/` if weights were previously saved.
        """
        storage = registry_path
        storage.mkdir(parents=True, exist_ok=True)
        ckpt = storage / "minilm.pt"
        if ckpt.exists():
            model = TinyGPT()
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        else:
            model = TinyGPT()
            # tiny deterministic seed so everyone gets same baseline
            torch.manual_seed(42)
            torch.save(model.state_dict(), ckpt)
        return model, model.tokenizer

    def generate(
        self,
        model: TinyGPT,
        tokenizer: _Tokenizer,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        ids = tokenizer.encode(prompt)
        ids = model.generate_ids(ids, max_new=max_new_tokens, temperature=temperature)
        return tokenizer.decode(ids)

    def embed(
        self, model: TinyGPT, tokenizer: _Tokenizer, text: str | list[str]
    ) -> list[list[float]]:
        texts = [text] if isinstance(text, str) else text
        model.eval()
        out: list[list[float]] = []
        with torch.no_grad():
            for t in texts:
                ids = tokenizer.encode(t)[: model.cfg.block_size]
                idx = torch.tensor([ids], dtype=torch.long)
                emb = model.tok_emb(idx).mean(dim=1).squeeze(0)  # [n_embd]
                # normalise
                v = emb / math.sqrt(float((emb * emb).sum().clamp(min=1e-8)))
                out.append(v.tolist())
        return out

    def info(self, model: TinyGPT) -> dict[str, Any]:
        return {
            "kind": "TinyGPT",
            "params": model.num_parameters(),
            "layers": model.cfg.n_layer,
            "heads": model.cfg.n_head,
            "embd": model.cfg.n_embd,
            "block_size": model.cfg.block_size,
            "vocab_size": model.cfg.vocab_size,
        }


# ── helper for tests ────────────────────────────────────────────────────────
_WORD_RE = re.compile(r"\w+")

def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())
