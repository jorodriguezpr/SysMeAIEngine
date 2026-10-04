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

"""Thin HTTP client for master -> slave-node calls against /cluster/* routes."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import requests

from .nodes import base_url


class NodeClient:
    def __init__(self, node: dict[str, Any], timeout: float = 15.0) -> None:
        self.node = node
        self.base = base_url(node)
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        api_key = self.node.get("api_key")
        return {"X-API-Key": api_key} if api_key else {}

    def info(self) -> dict[str, Any]:
        r = requests.get(f"{self.base}/cluster/info", headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def ping(self) -> tuple[bool, str]:
        try:
            self.info()
            return True, "ok"
        except requests.RequestException as e:
            return False, str(e)

    def submit_job(
        self,
        job_spec: dict[str, Any],
        dataset_path: Path,
        base_model_zip: Optional[Path] = None,
        upload_timeout: float = 300.0,
    ) -> dict[str, Any]:
        """POST a distributed training job: rendezvous info + hyperparams as a
        JSON field, dataset file, and (if the base model is a local checkpoint
        rather than an HF hub id) a zipped copy of it — a plain hub id needs no
        transfer since every rank downloads it independently."""
        import json

        files = {"dataset": (dataset_path.name, open(dataset_path, "rb"), "application/octet-stream")}
        if base_model_zip is not None:
            files["base_model_zip"] = (
                base_model_zip.name, open(base_model_zip, "rb"), "application/zip"
            )
        try:
            r = requests.post(
                f"{self.base}/cluster/jobs",
                headers=self._headers(),
                data={"job_spec": json.dumps(job_spec)},
                files=files,
                timeout=upload_timeout,
            )
        finally:
            for _, (_, fh, _) in files.items():
                fh.close()
        r.raise_for_status()
        return r.json()

    def list_jobs(self) -> list[dict[str, Any]]:
        r = requests.get(
            f"{self.base}/cluster/jobs", headers=self._headers(), timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def job_status(self, job_id: str) -> dict[str, Any]:
        r = requests.get(
            f"{self.base}/cluster/jobs/{job_id}", headers=self._headers(), timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def job_log(self, job_id: str) -> str:
        r = requests.get(
            f"{self.base}/cluster/jobs/{job_id}/log", headers=self._headers(), timeout=self.timeout
        )
        r.raise_for_status()
        return r.text

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        r = requests.post(
            f"{self.base}/cluster/jobs/{job_id}/cancel", headers=self._headers(), timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()
