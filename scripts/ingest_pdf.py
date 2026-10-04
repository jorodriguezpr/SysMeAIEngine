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

"""Extract text from a PDF → save into data/raw/. Cleans whitespace, keeps filename stem."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.pipeline import clean_text, extract_pdf  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--min-chars", type=int, default=50, help="skip if shorter than this")
    args = p.parse_args()

    if not args.file.exists():
        print(f"ERROR: file not found: {args.file}", file=sys.stderr)
        return 1
    text = clean_text(extract_pdf(args.file))
    if len(text) < args.min_chars:
        print(f"SKIP: {args.file.name} — only {len(text)} chars extracted")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / (args.file.stem + ".txt")
    out_file.write_text(text, encoding="utf-8")
    print(f"WROTE {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
