"""SEARCH: find candidate sources. Providers are swappable; each returns SearchResult items."""

import asyncio
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Protocol
from urllib.parse import urlencode

import aiohttp

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    source: str = ""          # publisher / site name
    published_at: str | None = None  # ISO 8601 when known
    provider: str = ""


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, max_results: int) -> list[SearchResult]: ...


def _iso(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class DuckDuckGoProvider:
    """DuckDuckGo web + news search via the `ddgs` package (no API key)."""

    name = "duckduckgo"

    def __init__(self, news_timelimit: str = "y", include_web: bool = True):
        self.news_timelimit = news_timelimit
        self.include_web = include_web

    def _search_sync(self, query: str, max_results: int) -> list[SearchResult]:
        from ddgs import DDGS

        results: list[SearchResult] = []
        ddgs = DDGS(timeout=15)
        try:
            for item in ddgs.news(query, max_results=max_results, timelimit=self.news_timelimit) or []:
                results.append(SearchResult(
                    title=item.get("title", ""), url=item.get("url", ""),
                    snippet=item.get("body", ""), source=item.get("source", ""),
                    published_at=_iso(item.get("date")), provider="ddg-news",
                ))
        except Exception as e:  # ddgs raises its own exception types for rate limits etc.
            log.warning("DuckDuckGo news search failed for %r: %s", query, e)

        if self.include_web:
            try:
                for item in ddgs.text(query, max_results=max_results) or []:
                    results.append(SearchResult(
                        title=item.get("title", ""), url=item.get("href", ""),
                        snippet=item.get("body", ""), provider="ddg-web",
                    ))
            except Exception as e:
                log.warning("DuckDuckGo web search failed for %r: %s", query, e)
        return results

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        # ddgs is synchronous; keep it off the event loop.
        return await asyncio.to_thread(self._search_sync, query, max_results)


class GoogleNewsRSSProvider:
    """Google News RSS search (no API key). Links are Google redirect URLs, so we don't fetch them."""

    name = "google-news"
    BASE = "https://news.google.com/rss/search"

    def __init__(self, session_factory, when: str = "1y"):
        self._session_factory = session_factory
        self.when = when

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        params = {"q": f"{query} when:{self.when}", "hl": "en", "gl": "ZA", "ceid": "ZA:en"}
        try:
            session = await self._session_factory()
            async with session.get(
                f"{self.BASE}?{urlencode(params)}", timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    log.warning("Google News RSS HTTP %s for %r", resp.status, query)
                    return []
                body = await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("Google News RSS failed for %r: %s", query, e)
            return []
        return parse_google_news_rss(body)[:max_results]


def parse_google_news_rss(xml_text: str) -> list[SearchResult]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        log.warning("Google News RSS returned invalid XML")
        return []
    results = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        # Titles look like "Headline - Publisher"; drop the publisher suffix.
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        results.append(SearchResult(
            title=title,
            url=(item.findtext("link") or "").strip(),
            source=source,
            published_at=_iso(item.findtext("pubDate")),
            provider="google-news",
        ))
    return results


async def search_all(
    providers: list[SearchProvider], queries: list[str], max_results: int
) -> list[SearchResult]:
    """Run every query on every provider concurrently. Failures yield no results, not exceptions."""
    tasks = [p.search(q, max_results) for p in providers for q in queries]
    out: list[SearchResult] = []
    for result in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(result, BaseException):
            log.warning("Search provider error: %r", result)
            continue
        out.extend(r for r in result if r.url and r.title)
    return out
