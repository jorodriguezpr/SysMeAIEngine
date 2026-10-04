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

"""Safe arithmetic shortcut for chat.

Small local LLMs (and honestly, LLMs in general) are unreliable at exact
arithmetic — next-token prediction over text patterns isn't computation.
Rather than hope the model gets it right, pure arithmetic expressions are
detected here and evaluated directly with Python, bypassing the model
entirely. Uses an AST walk over a fixed operator whitelist — never `eval()`
on raw text — so it can't execute anything beyond +-*/%**.
"""
from __future__ import annotations

import ast
import operator
import re

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_CHARS_RE = re.compile(r"^[\s\d+\-*/().%]+$")


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression node: {node!r}")


def try_calculate(text: str) -> str | None:
    """Return "<expr> = <result>" if `text` is a pure arithmetic expression
    (optionally trailing '=' or '?'), else None."""
    stripped = text.strip().rstrip("=?").strip()
    if not stripped or not re.search(r"\d", stripped):
        return None
    if not _ALLOWED_CHARS_RE.match(stripped):
        return None
    try:
        tree = ast.parse(stripped, mode="eval")
        result = _eval_node(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError):
        return None
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{stripped} = {result}"
