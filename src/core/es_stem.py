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

"""Light Spanish stemmer for keyword (TF-IDF) retrieval -- no dependencies.

Why: TF-IDF treats "bautizó", "bautismo", "bautizar" as unrelated tokens,
so a question phrased with a different word form than the book misses the
passage. Reducing words to a shared stem ("bautiz") lets keyword search
match across conjugation and derivation.

Method: lowercase, strip accents, then remove the longest matching suffix
from ordered classes (derivational, then verbal, then plural/gender),
keeping a stem of at least 4 letters. Deliberately conservative (a light
Snowball-style stemmer, not a full morphological analyzer): over-stemming
merges unrelated words and hurts precision more than it helps recall.
"""
from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[^\W\d_]{2,}")

# longest first within each class
_DERIVATIONAL = sorted([
    "amientos", "imientos", "amiento", "imiento", "aciones", "uciones", "adoras", "adores",
    "ancias", "encias", "idades", "mente", "ación", "ucion", "ución", "acion", "adora", "ador",
    "ancia", "encia", "anzas", "anza", "istas", "ista", "ismos", "ismo", "ables", "ibles",
    "able", "ible", "osas", "osos", "osa", "oso", "ivas", "ivos", "iva", "ivo", "idad", "ez",
], key=len, reverse=True)
_VERBAL = sorted([
    "aríamos", "eríamos", "iríamos", "ábamos", "iéramos", "iésemos", "aremos", "eremos", "iremos",
    "aban", "ieron", "aron", "ando", "iendo", "yendo", "arán", "erán", "irán", "arían", "erían",
    "irían", "aría", "ería", "iría", "aba", "ían", "ías", "ía", "ados", "idos", "adas", "idas",
    "ado", "ido", "ada", "ida", "aste", "iste", "amos", "emos", "imos", "ar", "er", "ir", "ó", "ió",
], key=len, reverse=True)
_INFLECTIONAL = ["es", "os", "as", "s", "a", "o", "e"]


def _strip_accents(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


_DERIV = [_strip_accents(s) for s in _DERIVATIONAL]
_VERB = sorted({_strip_accents(s) for s in _VERBAL}, key=len, reverse=True)


def stem(word: str) -> str:
    w = _strip_accents(word.lower())
    if len(w) <= 4:
        return w
    for suffixes in (_DERIV, _VERB, _INFLECTIONAL):
        for s in suffixes:
            if w.endswith(s) and len(w) - len(s) >= 4:
                w = w[: -len(s)]
                break
    return w


def analyzer(text: str) -> list[str]:
    """Token analyzer for sklearn's TfidfVectorizer(analyzer=...)."""
    return [stem(t) for t in _TOKEN_RE.findall(text)]
