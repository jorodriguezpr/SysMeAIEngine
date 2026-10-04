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

"""Manifest-based incremental ingestion: track which source documents have
already been extracted, so re-running scripts/train_from_folder.py on a
folder only re-extracts files that actually changed. Extraction (pdfminer,
python-docx, etc.) is the second-most expensive ingest step after
embedding, and previously the only "skip" check was "does the output .txt
already exist" -- which never noticed an edited source file, so an updated
PDF silently kept training on its old, stale extracted text forever.

Staleness is decided by mtime+size, not a full-file hash -- cheap to check
on every run (no need to read a possibly-large PDF just to see if it
changed) and sufficient for the actual failure mode this fixes (a file
someone edited, not an adversarial same-size-same-mtime tamper scenario).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}  # corrupt/unreadable -- treat as empty; worst case is one extra full re-extract


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _fingerprint(doc: Path) -> dict[str, Any]:
    stat = doc.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def is_stale(manifest: dict[str, Any], doc: Path) -> bool:
    """True if `doc` needs (re-)extraction: never seen before, or its
    mtime/size changed since the last extraction recorded in the manifest."""
    entry = manifest.get(str(doc.resolve()))
    if entry is None:
        return True
    current = _fingerprint(doc)
    return entry.get("mtime") != current["mtime"] or entry.get("size") != current["size"]


def record(manifest: dict[str, Any], doc: Path, out_file: Path) -> None:
    manifest[str(doc.resolve())] = {**_fingerprint(doc), "out_file": str(out_file)}


def sweep_removed(manifest: dict[str, Any]) -> list[Path]:
    """Drops manifest entries for source files that no longer exist, and
    returns their recorded out_file paths so the caller can delete the now-
    orphaned extracted text too (otherwise a removed source's content keeps
    getting chunked/trained/indexed forever)."""
    removed_out_files = []
    for key in [k for k in manifest if not Path(k).exists()]:
        out_file = manifest[key].get("out_file")
        if out_file:
            removed_out_files.append(Path(out_file))
        del manifest[key]
    return removed_out_files
