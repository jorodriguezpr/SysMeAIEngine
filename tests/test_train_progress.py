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

"""Tests for src.core.train_progress -- parses the tqdm progress bar
HuggingFace's Trainer already prints to a job's log file into a percent-
complete estimate for the dashboard. Fixtures below are real captured
output from this project's own training runs (see project memory), not
synthesized -- including the multi-phase case (Loading weights -> Map ->
real training bar) and the completed-run case with no trailing bar."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.train_progress import job_progress, parse_training_progress  # noqa: E402


def test_parse_training_progress_mid_run():
    log = (
        "  0%|          | 0/2 [00:00<?, ?it/s]"
        "C:\\...\\dataloader.py:759: UserWarning: 'pin_memory'...\n"
        "  super().__init__(loader)\n"
        " 50%|#####     | 1/2 [03:52<03:52, 232.71s/it]"
    )
    result = parse_training_progress(log)
    assert result == {"percent": 50, "current_step": 1, "total_steps": 2}


def test_parse_training_progress_skips_labeled_setup_bars():
    # Real logs contain several bars in sequence (weight loading, dataset
    # mapping, then the actual training bar) -- the labeled ones must be
    # skipped in favor of the real (unlabeled) training bar even though
    # they appear earlier in the text.
    log = (
        "Loading weights: 100%|##########| 290/290 [00:01<00:00, 262.14it/s]\n"
        "Map: 100%|##########| 201/201 [00:00<00:00, 2783.05 examples/s]\n"
        "  0%|          | 0/1 [00:00<?, ?it/s]"
    )
    result = parse_training_progress(log)
    assert result == {"percent": 0, "current_step": 0, "total_steps": 1}


def test_parse_training_progress_returns_none_during_setup_phase():
    # Regression test for a real bug found live: a job still loading its
    # base model (no unlabeled training bar has printed yet) showed a
    # misleading "100%" because "Loading weights: 100%|...|290/290" was
    # (wrongly) reported as training progress. Correct behavior is to
    # report nothing yet, not the completed setup bar.
    log = (
        "Warning: You are sending unauthenticated requests to the HF Hub...\n"
        "Loading weights:   0%|          | 0/290 [00:00<?, ?it/s]"
        "Loading weights: 100%|##########| 290/290 [00:01<00:00, 262.14it/s]\n"
    )
    assert parse_training_progress(log) is None


def test_parse_training_progress_completed_run():
    log = (
        " 50%|#####     | 1/2 [03:52<03:52, 232.71s/it]"
        "100%|##########| 2/2 [05:56<00:00, 168.84s/it]"
        "100%|##########| 2/2 [05:58<00:00, 179.24s/it]\n"
        "{'train_runtime': '358.5', 'train_loss': '3.292', 'epoch': '1'}\n"
        "DONE in 359.3s\n"
    )
    result = parse_training_progress(log)
    assert result == {"percent": 100, "current_step": 2, "total_steps": 2}


def test_parse_training_progress_no_bar_yet():
    log = "[00:24:08] found 7 document(s) under C:\\opt\\Aprocrifos\n[00:24:08] extracting: doc1.pdf\n"
    assert parse_training_progress(log) is None


def test_parse_training_progress_empty_string():
    assert parse_training_progress("") is None


def test_job_progress_reads_real_file(tmp_path):
    log_file = tmp_path / "job.log"
    log_file.write_text(" 50%|#####     | 1/2 [03:52<03:52, 232.71s/it]", encoding="utf-8")
    assert job_progress(str(log_file)) == {"percent": 50, "current_step": 1, "total_steps": 2}


def test_job_progress_missing_file_returns_none(tmp_path):
    assert job_progress(str(tmp_path / "does-not-exist.log")) is None


def test_job_progress_none_path_returns_none():
    assert job_progress(None) is None


def test_job_progress_only_scans_tail(tmp_path):
    # A huge log with an old 10% bar way back and a real 90% bar near the
    # end should report the recent one, not get confused by the old one --
    # confirms the _TAIL_BYTES truncation doesn't accidentally pick up the
    # wrong (earlier) match after slicing.
    log_file = tmp_path / "big.log"
    padding = "x" * 20000
    log_file.write_text(
        " 10%|#         | 1/10 [00:01<00:09, 1.0s/it]" + padding + " 90%|######### | 9/10 [00:09<00:01, 1.0s/it]",
        encoding="utf-8",
    )
    result = job_progress(str(log_file))
    assert result == {"percent": 90, "current_step": 9, "total_steps": 10}
