"""Research pipeline with stub search providers and a scripted model (no network)."""

import asyncio

import pytest

from app.db import Database
from app.ollama_client import OllamaTimeout, OllamaUnavailable
from app.research.pipeline import ResearchBusy, ResearchService
from app.research.search import SearchResult
from app.research_config import load_research_config
from tests.helpers import StubOllama, analysis_reply, opp_json

CFG = load_research_config("research.toml")


class StubProvider:
    name = "stub"

    def __init__(self, results=None, delay=0.0):
        headlines = ["Exporters lose millions to spoilage", "Tower vandalism costs telcos",
                     "New horticulture export deal signed"]
        self.results = results if results is not None else [
            SearchResult(h, f"https://news.google.com/rss/articles/{i}", provider="google-news",
                         published_at=f"2026-10-0{i}T00:00:00+00:00", source="Herald")
            for i, h in enumerate(headlines, start=1)
        ]
        self.delay = delay

    async def search(self, query, max_results):
        await asyncio.sleep(self.delay)
        return self.results


@pytest.fixture
async def db():
    d = Database(":memory:")
    await d.connect()
    yield d
    await d.close()


_services: list[ResearchService] = []


@pytest.fixture(autouse=True)
async def _close_services():
    yield
    while _services:
        await _services.pop().close()


def service(db, ollama, provider=None):
    svc = ResearchService(db, ollama, CFG, providers=[provider or StubProvider()])
    _services.append(svc)
    return svc


async def test_successful_run_stores_opportunities_sources_and_run(db):
    ollama = StubOllama(analysis=[analysis_reply(
        opp_json(source_ids=[1]),
        opp_json(title="Telecom tower security monitoring", category="telecommunications",
                 problem="Towers are vandalised for batteries and diesel.",
                 solution="Remote tower sensors with alarm response.", source_ids=[1, 2]),
    )])
    svc = service(db, ollama)
    progress = []

    async def on_progress(msg):
        progress.append(msg)

    report = await svc.run("cold chain Zimbabwe", progress=on_progress)
    assert report.status == "completed", report.error
    assert len(report.outcomes) == 2 and all(o.is_new for o in report.outcomes)
    assert len(progress) == 2

    rows = await db.list_opportunities()
    assert {r["title"] for r in rows} == {"Cold storage for horticulture exporters", "Telecom tower security monitoring"}
    tower = next(r for r in rows if r["title"].startswith("Telecom"))
    assert len(await db.get_sources(tower["id"])) == 2
    run = await db.get_run(report.run_id)
    assert run["status"] == "completed" and run["opportunities_found"] == 2


async def test_repeat_run_updates_instead_of_duplicating(db):
    ollama = StubOllama(analysis=[
        analysis_reply(opp_json(source_ids=[1])),
        analysis_reply(opp_json(source_ids=[1, 2], evidence="New: exports up 35%")),
    ])
    svc = service(db, ollama)
    first = await svc.run("cold chain")
    second = await svc.run("cold chain again")

    assert first.outcomes[0].is_new and not second.outcomes[0].is_new
    assert second.outcomes[0].id == first.outcomes[0].id
    assert second.outcomes[0].new_source_count == 1   # source 1 already linked
    rows = await db.list_opportunities()
    assert len(rows) == 1 and rows[0]["seen_count"] == 2
    assert "exports up 35%" in rows[0]["evidence"]


async def test_distinct_opportunities_in_one_run_never_merge(db):
    # Near-identical titles from the same run: parse_analysis collapses true duplicates,
    # and the store step must not merge distinct ones into each other.
    ollama = StubOllama(
        analysis=[analysis_reply(
            opp_json(title="Agricultural finance for smallholders", source_ids=[1]),
            opp_json(title="Agricultural insurance for smallholders", source_ids=[2]),
        )],
        match_reply='{"match_id": 1}',  # the model would (wrongly) say "same" if asked
    )
    report = await service(db, ollama).run("agri")
    assert [o.is_new for o in report.outcomes] == [True, True]
    assert len(await db.list_opportunities()) == 2


async def test_no_search_results_fails_gracefully(db):
    ollama = StubOllama()
    report = await service(db, ollama, StubProvider(results=[])).run("x")
    assert report.status == "failed" and "No usable search results" in report.error
    assert (await db.get_run(report.run_id))["status"] == "failed"
    assert ollama.calls == []


@pytest.mark.parametrize("error, text", [
    (OllamaUnavailable("Can't reach Ollama."), "reach Ollama"),
    (OllamaTimeout("did not respond"), "did not respond"),
])
async def test_ollama_failures_fail_gracefully(db, error, text):
    report = await service(db, StubOllama(analysis=[error])).run("x")
    assert report.status == "failed" and text in report.error
    assert await db.list_opportunities() == []


async def test_malformed_model_output_fails_gracefully(db):
    report = await service(db, StubOllama(analysis=["nope", "still nope"])).run("x")
    assert report.status == "failed" and "invalid JSON" in report.error


async def test_unexpected_exception_is_contained(db, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("bug")
    monkeypatch.setattr("app.research.pipeline.analyse_sources", boom)
    report = await service(db, StubOllama()).run("x")
    assert report.status == "failed" and "Unexpected" in report.error
    assert (await db.get_run(report.run_id))["error"].startswith("RuntimeError")


async def test_failing_progress_callback_does_not_break_run(db):
    async def bad_progress(msg):
        raise ConnectionError("discord gone")
    report = await service(db, StubOllama(analysis=[analysis_reply(opp_json())])).run("x", progress=bad_progress)
    assert report.status == "completed"


async def test_concurrent_run_is_rejected(db):
    svc = service(db, StubOllama(analysis=[analysis_reply(opp_json())]), StubProvider(delay=0.2))
    first = asyncio.create_task(svc.run("a"))
    await asyncio.sleep(0.05)
    with pytest.raises(ResearchBusy):
        await svc.run("b")
    assert (await first).status == "completed"
