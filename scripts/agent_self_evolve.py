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

"""Self-evolve: search the web for a topic, scrape pages, write .txt files into raw/.

Respects the domain allowlist in .env (ENGINE_AGENT_ALLOWLIST).

Beyond the initial search-result URLs, this can also follow a few in-page
links per page (--max-depth > 1) to gather more per topic than the exact
search hits — bounded by --max-pages (the total scrape budget across all
depths) and --links-per-page, and restricted to the same domain as the page
a link was found on unless an allowlist is configured (in which case the
allowlist alone governs, same as for the initial search results).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.pipeline import clean_text, extract_any  # noqa: E402

_DOC_EXT_BY_CONTENT_TYPE = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}
_DOC_EXTS = (".pdf", ".docx", ".pptx", ".xlsx")


def _allowlisted(url: str, allowlist: list[str]) -> bool:
    if not allowlist:
        return True
    u = url.lower()
    return any(a in u for a in allowlist)


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _search(topic: str, max_results: int) -> list[str]:
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        print("ddgs not installed — skipping web search", file=sys.stderr)
        return []
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.text(topic, max_results=max_results):
            href = r.get("href") or r.get("url")
            if href:
                urls.append(href)
    return urls


def _doc_ext_from_url(url: str) -> str | None:
    path = url.split("?")[0].split("#")[0].lower()
    for ext in _DOC_EXTS:
        if path.endswith(ext):
            return ext
    return None


def _extract_document(content: bytes, ext: str) -> str:
    """Route a non-HTML response (PDF/Word/PowerPoint/Excel) through the same
    extractors ingestion uses for uploaded files, instead of misreading it as
    HTML."""
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(content)
        tmp_path = Path(f.name)
    try:
        return clean_text(extract_any(tmp_path))
    except Exception:
        return ""
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_article(html: str) -> tuple[str, str]:
    """readability-lxml strips nav/ads/sidebars far better than raw
    full-page text extraction — falls back to the full page if the
    "article" heuristic comes back too short (some reference-doc pages
    don't fit it well). Returns (text, article_html) — the html is reused
    for link discovery so followed links come from the article body, not
    site-wide nav/footer chrome."""
    from bs4 import BeautifulSoup

    article_html = ""
    text = ""
    try:
        from readability import Document as ReadabilityDocument

        article_html = ReadabilityDocument(html).summary()
        soup = BeautifulSoup(article_html, "lxml")
        text = clean_text(soup.get_text("\n"))
    except Exception:
        text = ""

    if len(text) >= 200:
        return text, article_html

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return clean_text(soup.get_text("\n")), html


def _extract_links(html: str, base_url: str) -> list[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        absolute = urljoin(base_url, href)
        if absolute.startswith(("http://", "https://")):
            links.append(absolute.split("#")[0])
    return links


def _scrape(url: str, ua: str, timeout: int = 30) -> tuple[str, list[str]]:
    """Returns (text, links) — links is empty for non-HTML documents."""
    try:
        import requests
    except ImportError:
        print("requests not installed", file=sys.stderr)
        return "", []
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": ua})
        r.raise_for_status()
    except Exception:
        return "", []

    content_type = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
    doc_ext = _DOC_EXT_BY_CONTENT_TYPE.get(content_type) or _doc_ext_from_url(url)
    if doc_ext:
        return _extract_document(r.content, doc_ext), []

    text, article_html = _extract_article(r.text)
    return text, _extract_links(article_html, url)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--topic", required=True)
    p.add_argument("--max-pages", type=int, default=6, help="total scrape budget across all depths")
    p.add_argument("--max-depth", type=int, default=1, help="1 = search results only; 2+ = also follow in-page links")
    p.add_argument("--links-per-page", type=int, default=3, help="links to follow per page when max-depth > 1")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ua", default="SysMeAIEngine/1.0 (+local)")
    args = p.parse_args()

    # load allowlist from env/config
    from src.core.config import Config

    cfg = Config.from_env()
    allowlist = cfg.agent_allowlist
    ua = cfg.agent_user_agent or args.ua

    print(f"SEARCHING DuckDuckGo for '{args.topic}' (up to {args.max_pages * 2} results)...", flush=True)
    seed_urls = [u for u in _search(args.topic, args.max_pages * 2) if _allowlisted(u, allowlist)]
    if not seed_urls:
        print("(no allowlisted urls found)", flush=True)
        return 0

    # Reserve budget for deeper levels: with a pure FIFO queue and no cap,
    # search always returns >= max_pages seeds, so the whole budget gets
    # consumed by depth-1 pages before a single depth-2 link is ever
    # reached (they queue behind every remaining seed). Splitting the
    # budget evenly across depths keeps --max-depth meaningful instead of
    # being a silent no-op.
    if args.max_depth > 1:
        seed_budget = max(1, -(-args.max_pages // args.max_depth))  # ceil
        seed_urls = seed_urls[:seed_budget]
    print(f"found {len(seed_urls)} allowlisted seed URL(s) to start from", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    queue: deque[tuple[str, int]] = deque((u, 1) for u in seed_urls)
    seen: set[str] = set()
    scraped = 0

    while queue and scraped < args.max_pages:
        url, depth = queue.popleft()
        if url in seen:
            continue
        seen.add(url)

        scraped += 1
        print(f"[{scraped}/{args.max_pages}] scraping (depth {depth}): {url}", flush=True)
        text, links = _scrape(url, ua=ua)
        if len(text) < 400:
            print(f"[{scraped}/{args.max_pages}] skip (only {len(text)} chars extracted)", flush=True)
        else:
            stem = (
                url.replace("https://", "").replace("http://", "")
                .replace("/", "_").replace("?", "_").replace("&", "_")[:90]
            )
            digest = hashlib.sha256(text.encode()).hexdigest()[:8]
            out_file = args.out / f"{stem}-{digest}.txt"
            out_file.write_text(text, encoding="utf-8")
            print(f"[{scraped}/{args.max_pages}] WROTE {out_file}", flush=True)

        if depth < args.max_depth and links:
            # same-domain by default (a link-following crawl reaches further
            # than a search result does, so it gets a tighter default leash);
            # an explicit allowlist governs instead, same as for seed URLs
            page_domain = _domain(url)
            candidates = [
                link for link in links
                if link not in seen
                and _allowlisted(link, allowlist)
                and (allowlist or _domain(link) == page_domain)
            ]
            for link in candidates[: args.links_per_page]:
                queue.append((link, depth + 1))

        time.sleep(1.0)  # be polite

    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
