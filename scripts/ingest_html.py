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

"""Extract visible text from an HTML file/URL snapshot → save to data/raw/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.pipeline import clean_text, extract_html  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, help="HTML file on disk (use this OR --url)")
    p.add_argument("--url", type=str, help="URL to fetch first, then parse")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--name", type=str, help="override output filename stem")
    args = p.parse_args()

    if args.url:
        import requests  # lazy

        r = requests.get(args.url, timeout=30, headers={"User-Agent": "SysMeAIEngine/1.0"})
        r.raise_for_status()
        tmp = args.out / f"__tmp_{hash(args.url) & 0xFFFFFF:x}.html"
        args.out.mkdir(parents=True, exist_ok=True)
        tmp.write_text(r.text, encoding="utf-8")
        src = tmp
        stem = args.name or (args.url.replace("https://", "").replace("http://", "").split("/")[0])
    elif args.file:
        src = args.file
        stem = args.name or src.stem
    else:
        p.error("either --file or --url is required")

    text = clean_text(extract_html(src))
    if not text:
        print(f"SKIP: {src} — empty")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / (stem + ".txt")
    out_file.write_text(text, encoding="utf-8")
    print(f"WROTE {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
