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

"""Multi-node distributed training: master orchestrator + per-node agent.

See docs/CLUSTER.md for setup (firewall, ENGINE_HOST, matching torch versions)
and the honest limitations (heterogeneous CPU+GPU ranks bottleneck to the
slowest rank under synchronous DDP; not benchmarked on real multi-machine
hardware — see src/core/device_select.py for the same caveat on plain CUDA).
"""
