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

"""Estimates a running training job's percent-complete by parsing the
tqdm-style progress bar HuggingFace's Trainer already prints to the job's
own log file (e.g. "50%|#####     | 1/2 [03:52<03:52, 232.71s/it]") --
no new instrumentation needed, since a redirected tqdm bar's carriage
returns land in the log file as literal text rather than really
overwriting a line, so every update it ever printed is still readable.

The same log also contains "Loading weights: 100%|...|290/290" and
"Map: 100%|...|201/201" bars from preprocessing, printed with an explicit
tqdm `desc` label right before the percentage -- the real training loop's
own bar has no such label. These must NOT be reported as training
progress: confirmed live, a job still loading its base model showed a
misleading "100%" (from "Loading weights" hitting 290/290) while the
actual training hadn't started. `_LABEL_BEFORE_RE` filters these out by
checking the text immediately preceding each match for a "word:" prefix,
so only the last *unlabeled* bar -- the real one -- is ever reported.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

_PROGRESS_RE = re.compile(r"(\d{1,3})%\|[^|]*\|\s*(\d+)/(\d+)")
_LABEL_BEFORE_RE = re.compile(r"[A-Za-z][A-Za-z \t]*:\s*$")

# How much of the log's tail to scan -- the latest progress bar update is
# always near the end; reading the whole file for a long-running job would
# be wasteful.
_TAIL_BYTES = 8192
_LABEL_LOOKBACK = 40


def parse_training_progress(log_text: str) -> Optional[dict[str, Any]]:
    for m in reversed(list(_PROGRESS_RE.finditer(log_text))):
        preceding = log_text[max(0, m.start() - _LABEL_LOOKBACK):m.start()]
        if _LABEL_BEFORE_RE.search(preceding):
            continue  # a labeled setup bar (e.g. "Loading weights:"), not the real training loop
        percent, current, total = (int(g) for g in m.groups())
        return {"percent": min(percent, 100), "current_step": current, "total_steps": total}
    return None


def job_progress(log_path: Optional[str]) -> Optional[dict[str, Any]]:
    """None if there's no log, the file doesn't exist yet, or no progress
    bar has printed anything yet (e.g. still extracting/pinging nodes)."""
    if not log_path:
        return None
    path = Path(log_path)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[-_TAIL_BYTES:]
    except OSError:
        return None
    return parse_training_progress(text)
