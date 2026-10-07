"""COLLECT: filter, deduplicate and fetch article text for search results."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import aiohttp

from app.research.search import SearchResult, _iso
from app.research_config import SearchConfig

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
MAX_DOWNLOAD_BYTES = 3_000_000
_TRACKING_PARAMS = re.compile(r"^(utm_|fbclid$|gclid$|mc_|ref$|ref_src$|ocid$)")


@dataclass
class CollectedSource:
    title: str
    url: str
    source: str
    published_at: str | None
    text: str           # extracted article text, or the search snippet if fetching failed
    retrieved_at: str
    fetched: bool       # True if `text` came from the page itself


def normalize_url(url: str) -> str:
    """Canonical form for duplicate detection: no scheme/www/fragment/tracking params/trailing slash."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if not _TRACKING_PARAMS.match(k)))
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(("", host, path, query, ""))


def _domain(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def filter_and_dedupe(results: list[SearchResult], cfg: SearchConfig) -> list[SearchResult]:
    """Drop blocked/noisy results and duplicates (same URL, or near-identical headline)."""
    blocked_words = [w.lower() for w in cfg.blocked_title_words]
    seen_urls: set[str] = set()
    kept: list[SearchResult] = []
    kept_titles: list[str] = []

    for r in results:
        domain = _domain(r.url)
        if any(domain == d or domain.endswith("." + d) for d in cfg.blocked_domains):
            continue
        title_words = set(_norm_title(r.title).split())
        if any(w in title_words for w in blocked_words):
            continue
        key = normalize_url(r.url)
        if key in seen_urls:
            continue
        # Same story syndicated across sites / providers.
        nt = _norm_title(r.title)
        if any(SequenceMatcher(None, nt, t).ratio() > 0.9 for t in kept_titles):
            continue
        seen_urls.add(key)
        kept_titles.append(nt)
        kept.append(r)
    return kept


def rank_results(results: list[SearchResult]) -> list[SearchResult]:
    """Interleave providers (so the top N isn't all headline-only news), newest first within each."""
    groups: dict[str, list[SearchResult]] = {}
    for r in results:
        groups.setdefault(r.provider, []).append(r)
    for provider, items in groups.items():
        dated = sorted((r for r in items if r.published_at), key=lambda r: r.published_at, reverse=True)
        groups[provider] = dated + [r for r in items if not r.published_at]

    ranked: list[SearchResult] = []
    queues = list(groups.values())
    while any(queues):
        for q in queues:
            if q:
                ranked.append(q.pop(0))
    return ranked


def _extract(html: str, url: str) -> tuple[str, str | None]:
    import trafilatura

    out = trafilatura.extract(
        html, url=url, output_format="json", with_metadata=True,
        include_comments=False, include_tables=False, favor_precision=True,
    )
    if not out:
        return "", None
    data = json.loads(out)
    return (data.get("text") or "").strip(), data.get("date")


async def _fetch_one(
    session: aiohttp.ClientSession, r: SearchResult, cfg: SearchConfig, sem: asyncio.Semaphore
) -> CollectedSource:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = CollectedSource(
        title=r.title, url=r.url, source=r.source or _domain(r.url), published_at=r.published_at,
        text=r.snippet.strip(), retrieved_at=now, fetched=False,
    )
    if r.provider == "google-news":
        return base  # redirect links; headline + publisher + date is what we have

    async with sem:
        try:
            async with session.get(
                r.url, timeout=aiohttp.ClientTimeout(total=cfg.fetch_timeout),
                headers={"User-Agent": USER_AGENT}, allow_redirects=True,
            ) as resp:
                ctype = resp.headers.get("Content-Type", "")
                if resp.status != 200 or "html" not in ctype:
                    log.debug("Skip fetch %s: HTTP %s %s", r.url, resp.status, ctype)
                    return base
                # content.read(n) returns only what's buffered, so read chunks up to the cap.
                raw = bytearray()
                async for chunk in resp.content.iter_chunked(65536):
                    raw.extend(chunk)
                    if len(raw) >= MAX_DOWNLOAD_BYTES:
                        break
                html = bytes(raw).decode(resp.charset or "utf-8", errors="replace")
        except (aiohttp.ClientError, asyncio.TimeoutError, UnicodeError, ValueError) as e:
            log.debug("Fetch failed %s: %r", r.url, e)
            return base

    try:
        text, date = await asyncio.to_thread(_extract, html, r.url)
    except Exception as e:  # extraction must never break a run
        log.debug("Extraction failed %s: %r", r.url, e)
        return base

    if len(text) > len(base.text):
        base.text = text[: cfg.max_chars_per_source]
        base.fetched = True
    if not base.published_at and date:
        base.published_at = _iso(date)
    return base


async def collect_sources(
    session: aiohttp.ClientSession, results: list[SearchResult], cfg: SearchConfig
) -> list[CollectedSource]:
    """Filter, dedupe and rank results, fetch a wider pool, then keep up to `max_sources`,
    preferring sources with full article text over headline-only ones."""
    candidates = rank_results(filter_and_dedupe(results, cfg))[: cfg.max_sources * 2]
    sem = asyncio.Semaphore(cfg.fetch_concurrency)
    collected = await asyncio.gather(*(_fetch_one(session, r, cfg, sem) for r in candidates))

    full = [s for s in collected if s.fetched]
    snippet = [s for s in collected if not s.fetched and s.text]
    headline = [s for s in collected if not s.fetched and not s.text]
    # Headlines are still useful evidence (fresh, dated), but cap them so text-backed sources dominate.
    chosen = full[: cfg.max_sources]
    chosen += snippet[: cfg.max_sources - len(chosen)]
    chosen += headline[: min(cfg.max_sources - len(chosen), max(1, cfg.max_sources // 3))]
    return chosen
