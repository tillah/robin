import asyncio

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from app.research.collect import collect_sources, filter_and_dedupe, normalize_url, rank_results
from app.research.search import SearchResult, parse_google_news_rss, search_all
from app.research_config import SearchConfig

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>BII commits $65m to Zimbabwe agriculture - Trade Finance Global</title>
<link>https://news.google.com/rss/articles/abc</link>
<pubDate>Tue, 06 Oct 2026 14:30:51 GMT</pubDate><source url="https://tfg.com">Trade Finance Global</source></item>
<item><title>Second story</title><link>https://news.google.com/rss/articles/def</link></item>
</channel></rss>"""

ARTICLE = """<html><head><title>Big news</title></head><body><article>
<h1>Zimbabwe cold chain gap</h1>
""" + "".join(f"<p>Paragraph {i}: horticulture exporters report heavy spoilage because cold storage "
              f"near farms is scarce, and demand from European buyers keeps rising.</p>" for i in range(12)) + """
</article></body></html>"""


def test_parse_google_news_rss():
    items = parse_google_news_rss(RSS)
    assert len(items) == 2
    assert items[0].title == "BII commits $65m to Zimbabwe agriculture"
    assert items[0].source == "Trade Finance Global"
    assert items[0].published_at == "2026-10-06T14:30:51+00:00"
    assert items[1].published_at is None
    assert parse_google_news_rss("<not xml") == []


async def test_search_all_survives_failing_provider():
    class Good:
        name = "good"
        async def search(self, q, n):
            return [SearchResult("T", "https://a.com/x"), SearchResult("", "https://no-title.com")]

    class Bad:
        name = "bad"
        async def search(self, q, n):
            raise aiohttp.ClientConnectionError("network down")

    results = await search_all([Good(), Bad()], ["q"], 5)
    assert [r.url for r in results] == ["https://a.com/x"]


def test_normalize_url():
    assert normalize_url("https://www.Example.com/a/?utm_source=x&id=2#frag") == \
        normalize_url("http://example.com/a?id=2")
    assert normalize_url("https://example.com/a?id=1") != normalize_url("https://example.com/a?id=2")


def test_filter_and_dedupe():
    cfg = SearchConfig(blocked_domains=("indeed.com",), blocked_title_words=("jobs",))
    results = [
        SearchResult("Cold chain gap in Zimbabwe", "https://herald.co.zw/a?utm_source=t"),
        SearchResult("Cold chain gap in Zimbabwe", "https://www.herald.co.zw/a"),           # same URL
        SearchResult("Cold chain gap in Zimbabwe!", "https://other.com/syndicated"),       # same story
        SearchResult("Farm jobs in Harare", "https://jobsite.co.zw/1"),                     # blocked word
        SearchResult("Anything", "https://za.indeed.com/x"),                                # blocked domain
        SearchResult("Telecom tender announced", "https://potraz.gov.zw/t"),
    ]
    kept = filter_and_dedupe(results, cfg)
    assert [r.url for r in kept] == ["https://herald.co.zw/a?utm_source=t", "https://potraz.gov.zw/t"]


def test_rank_interleaves_providers_newest_first():
    rs = [
        SearchResult("g-old", "u1", provider="google-news", published_at="2026-01-01T00:00:00+00:00"),
        SearchResult("g-new", "u2", provider="google-news", published_at="2026-10-01T00:00:00+00:00"),
        SearchResult("web1", "u3", provider="ddg-web"),
        SearchResult("web2", "u4", provider="ddg-web"),
    ]
    assert [r.title for r in rank_results(rs)] == ["g-new", "web1", "g-old", "web2"]


@pytest.fixture
async def site():
    async def article(request):
        return web.Response(text=ARTICLE, content_type="text/html")

    async def missing(request):
        return web.Response(status=404)

    async def slow(request):
        await asyncio.sleep(3)
        return web.Response(text=ARTICLE, content_type="text/html")

    async def pdf(request):
        return web.Response(body=b"%PDF", content_type="application/pdf")

    app = web.Application()
    app.router.add_get("/article", article)
    app.router.add_get("/missing", missing)
    app.router.add_get("/slow", slow)
    app.router.add_get("/doc.pdf", pdf)
    server = TestServer(app)
    await server.start_server()
    yield str(server.make_url("")).rstrip("/")
    await server.close()


async def test_collect_sources_fetch_and_fallbacks(site, closed_port_url):
    cfg = SearchConfig(max_sources=10, fetch_timeout=1, max_chars_per_source=500)
    results = [
        SearchResult("Article", f"{site}/article", snippet="snip a", provider="ddg-web"),
        SearchResult("Missing page", f"{site}/missing", snippet="snip missing", provider="ddg-web"),
        SearchResult("Slow page", f"{site}/slow", snippet="snip slow", provider="ddg-web"),
        SearchResult("A PDF", f"{site}/doc.pdf", snippet="snip pdf", provider="ddg-web"),
        SearchResult("Network down", f"{closed_port_url}/x", snippet="snip down", provider="ddg-web"),
        SearchResult("Headline only", "https://news.google.com/rss/articles/zzz", provider="google-news",
                     published_at="2026-10-01T00:00:00+00:00"),
    ]
    async with aiohttp.ClientSession() as session:
        sources = await collect_sources(session, results, cfg)
    by_title = {s.title: s for s in sources}

    assert by_title["Article"].fetched is True
    assert "cold storage" in by_title["Article"].text
    assert len(by_title["Article"].text) <= 500
    for t in ("Missing page", "Slow page", "A PDF", "Network down"):
        assert by_title[t].fetched is False and by_title[t].text.startswith("snip")
    assert by_title["Headline only"].text == "" and by_title["Headline only"].published_at
    assert sources[0].title == "Article"  # full-text sources come first


async def test_collect_caps_headline_only_sources():
    cfg = SearchConfig(max_sources=6)
    headlines = ["Maize prices surge", "Telco launches 5G", "New tender for ICT", "Drought hits south",
                 "Bank funds SMEs", "Cyber attack on utility", "Mine expands output", "Fibre rollout begins",
                 "Tobacco sales open", "Startup raises seed round"]
    results = [SearchResult(h, f"https://news.google.com/rss/articles/{i}", provider="google-news")
               for i, h in enumerate(headlines)]
    async with aiohttp.ClientSession() as session:
        sources = await collect_sources(session, results, cfg)
    assert len(sources) == 2  # max_sources // 3
