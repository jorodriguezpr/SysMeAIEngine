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

"""Engine configuration loaded from env (.env supported via python-dotenv)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - dotenv optional
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[2]
if load_dotenv:
    load_dotenv(ROOT / ".env", override=False)


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


@dataclass
class Config:
    # server
    host: str = "127.0.0.1"
    port: int = 11534
    api_key: Optional[str] = None

    # storage
    models_dir: Path = ROOT / "models"
    data_dir: Path = ROOT / "data"
    datasets_dir: Path = ROOT / "data" / "datasets"
    chunks_dir: Path = ROOT / "data" / "chunks"
    raw_dir: Path = ROOT / "data" / "raw"
    checkpoints_dir: Path = ROOT / "models" / "checkpoints"
    weights_dir: Path = ROOT / "models" / "weights"

    # provider defaults
    default_model: str = "minilm"
    default_provider: str = "minilm"
    hf_token: Optional[str] = None
    # 'auto' uses a real NVIDIA CUDA GPU when present, else CPU — see
    # src/core/device_select.py. Per-model registry entries can override this.
    default_device: str = "auto"

    # trainer
    train_method: str = "lora"
    train_epochs: int = 1
    train_lr: float = 2e-4
    train_batch: int = 1
    train_grad_accum: int = 8
    train_max_seq_len: int = 1024

    # agent
    agent_max_pages: int = 6
    agent_interval_s: int = 3600
    agent_allowlist: list[str] = field(default_factory=list)
    agent_user_agent: str = "SysMeAIEngine/1.0"

    # logging
    log_level: str = "INFO"

    # ui auth: list[(user, pass)]
    ui_auth: list[tuple[str, str]] = field(default_factory=list)

    def ensure_layout(self) -> None:
        for p in (
            self.models_dir,
            self.data_dir,
            self.datasets_dir,
            self.chunks_dir,
            self.raw_dir,
            self.checkpoints_dir,
            self.weights_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, base: Optional[Path] = None) -> "Config":
        base = base or ROOT
        cfg = cls()
        m = {f.name: f for f in fields(cls)}

        def parse_path(fieldname: str, out_attr: str) -> None:
            raw = os.getenv(f"ENGINE_{out_attr.upper()}")
            if raw:
                setattr(cfg, fieldname, _resolve(base, raw))

        # simple scalars
        cfg.host = os.getenv("ENGINE_HOST", cfg.host)
        cfg.port = int(os.getenv("ENGINE_PORT", cfg.port))
        cfg.api_key = os.getenv("ENGINE_API_KEY") or None
        cfg.default_model = os.getenv("ENGINE_DEFAULT_MODEL", cfg.default_model)
        cfg.default_provider = os.getenv("ENGINE_DEFAULT_PROVIDER", cfg.default_provider)
        cfg.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None
        cfg.default_device = os.getenv("ENGINE_DEFAULT_DEVICE", cfg.default_device)

        # paths
        parse_path("models_dir", "MODELS_DIR")
        parse_path("data_dir", "DATA_DIR")
        parse_path("datasets_dir", "DATASETS_DIR")
        parse_path("chunks_dir", "CHUNKS_DIR")
        parse_path("raw_dir", "RAW_DIR")
        parse_path("checkpoints_dir", "CHECKPOINTS_DIR")
        parse_path("weights_dir", "WEIGHTS_DIR")

        # trainer
        cfg.train_method = os.getenv("ENGINE_TRAIN_METHOD", cfg.train_method)
        cfg.train_epochs = int(os.getenv("ENGINE_TRAIN_EPOCHS", cfg.train_epochs))
        cfg.train_lr = float(os.getenv("ENGINE_TRAIN_LR", cfg.train_lr))
        cfg.train_batch = int(os.getenv("ENGINE_TRAIN_BATCH", cfg.train_batch))
        cfg.train_grad_accum = int(
            os.getenv("ENGINE_TRAIN_GRAD_ACCUM", cfg.train_grad_accum)
        )
        cfg.train_max_seq_len = int(
            os.getenv("ENGINE_MAX_SEQ_LEN", cfg.train_max_seq_len)
        )

        # agent
        cfg.agent_max_pages = int(
            os.getenv("ENGINE_AGENT_MAX_PAGES", cfg.agent_max_pages)
        )
        cfg.agent_interval_s = int(
            os.getenv("ENGINE_AGENT_INTERVAL_S", cfg.agent_interval_s)
        )
        allow = os.getenv("ENGINE_AGENT_ALLOWLIST", "").strip()
        cfg.agent_allowlist = [s.strip().lower() for s in allow.split(",") if s.strip()]
        cfg.agent_user_agent = os.getenv("ENGINE_AGENT_UA", cfg.agent_user_agent)

        cfg.log_level = os.getenv("ENGINE_LOG_LEVEL", cfg.log_level).upper()

        auth = os.getenv("ENGINE_UI_AUTH", "").strip()
        if auth:
            pairs = []
            for item in auth.split(","):
                if ":" in item:
                    u, pw = item.split(":", 1)
                    pairs.append((u.strip(), pw))
            cfg.ui_auth = pairs

        cfg.ensure_layout()
        return cfg
