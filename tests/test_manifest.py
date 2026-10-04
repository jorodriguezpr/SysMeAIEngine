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

"""Tests for src.ingestion.manifest -- the mtime/size staleness tracking
that makes scripts/train_from_folder.py's re-extraction step incremental."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.manifest import (  # noqa: E402
    is_stale, load_manifest, record, save_manifest, sweep_removed,
)


def test_new_file_is_always_stale(tmp_path):
    doc = tmp_path / "a.pdf"
    doc.write_text("content")
    assert is_stale({}, doc) is True


def test_unchanged_file_is_not_stale_after_record(tmp_path):
    doc = tmp_path / "a.pdf"
    doc.write_text("content")
    manifest = {}
    record(manifest, doc, tmp_path / "a.txt")
    assert is_stale(manifest, doc) is False


def test_edited_file_becomes_stale(tmp_path):
    doc = tmp_path / "a.pdf"
    doc.write_text("content")
    manifest = {}
    record(manifest, doc, tmp_path / "a.txt")
    assert is_stale(manifest, doc) is False

    time.sleep(0.05)
    doc.write_text("different, longer content")  # changes size and mtime
    assert is_stale(manifest, doc) is True


def test_sweep_removed_drops_missing_sources_and_returns_their_outputs(tmp_path):
    doc = tmp_path / "a.pdf"
    doc.write_text("content")
    out_file = tmp_path / "a.txt"
    manifest = {}
    record(manifest, doc, out_file)
    assert len(manifest) == 1

    doc.unlink()  # source removed
    removed = sweep_removed(manifest)
    assert removed == [out_file]
    assert manifest == {}


def test_save_and_load_roundtrip(tmp_path):
    doc = tmp_path / "a.pdf"
    doc.write_text("content")
    manifest = {}
    record(manifest, doc, tmp_path / "a.txt")

    manifest_path = tmp_path / "_manifest.json"
    save_manifest(manifest_path, manifest)
    loaded = load_manifest(manifest_path)
    assert is_stale(loaded, doc) is False


def test_load_manifest_missing_file_returns_empty(tmp_path):
    assert load_manifest(tmp_path / "does_not_exist.json") == {}


def test_load_manifest_corrupt_file_returns_empty_not_raises(tmp_path):
    p = tmp_path / "_manifest.json"
    p.write_text("not valid json {{{", encoding="utf-8")
    assert load_manifest(p) == {}
