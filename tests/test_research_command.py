import asyncio

import pytest

from app.commands.research import handle_research
from app.db import Database
from app.embeds import (
    EMBED_TOTAL_LIMIT,
    opportunity_detail_embed,
    opportunity_list_embed,
    opportunity_summary_embed,
)
from app.research.pipeline import ResearchService
from app.research_config import load_research_config
from tests.helpers import FakeInteraction, StubOllama, analysis_reply, opp_json
from tests.test_pipeline import StubProvider

CFG = load_research_config("research.toml")


@pytest.fixture
async def db():
    d = Database(":memory:")
    await d.connect()
    yield d
    await d.close()


async def test_research_command_success(db):
    svc = ResearchService(db, StubOllama(analysis=[analysis_reply(opp_json())]), CFG, providers=[StubProvider()])
    inter = FakeInteraction()
    await handle_research(inter, "cold chain Zimbabwe", svc)

    assert inter.deferred
    assert any("Searching" in e for e in inter.edits)
    assert "Found **1** opportunities" in inter.edits[-1]
    [msg] = inter.sent
    embed = msg["embed"]
    assert embed.title.startswith("#1 Cold storage")
    assert "Score" in embed.fields[0].name
    assert "news.google.com" in next(f.value for f in embed.fields if f.name == "Sources")
    assert "new" in embed.footer.text


async def test_research_command_reports_failure(db):
    svc = ResearchService(db, StubOllama(), CFG, providers=[StubProvider(results=[])])
    inter = FakeInteraction()
    await handle_research(inter, "x", svc)
    assert "failed" in inter.edits[-1] and inter.sent == []


async def test_research_command_busy(db):
    svc = ResearchService(db, StubOllama(analysis=[analysis_reply(opp_json())]), CFG,
                          providers=[StubProvider(delay=0.2)])
    running = asyncio.create_task(svc.run("a"))
    await asyncio.sleep(0.05)
    inter = FakeInteraction()
    await handle_research(inter, "b", svc)
    assert "already" in inter.edits[-1] or "in progress" in inter.edits[-1]
    await running


def row(**kw):
    base = dict(id=7, title="T", country="Zimbabwe", category="agriculture", problem="P", solution="S",
                target_customers="C", evidence="E", competitors="X", why_now="W", risks="R", next_step="N",
                demand=8, competition=4, startup_difficulty=6, revenue_potential=7, accessibility=6,
                score=7.2, status="active", seen_count=1, first_seen="2026-10-07T00:00:00+00:00",
                last_seen="2026-10-07T00:00:00+00:00")
    base.update(kw)
    return base


def test_embeds_stay_within_discord_limits():
    huge = "x" * 5000
    r = row(title=huge, problem=huge, solution=huge, evidence=huge, why_now=huge, risks=huge,
            competitors=huge, target_customers=huge, next_step=huge)
    sources = [{"title": "S[1]" + "y" * 300, "url": f"https://a.com/{i}", "source": "Pub",
                "published_at": "2026-10-01T00:00:00+00:00"} for i in range(40)]
    for e in (opportunity_detail_embed(r, sources), opportunity_summary_embed(r, sources, "footer")):
        assert len(e) <= EMBED_TOTAL_LIMIT
        assert len(e.title) <= 256
        assert all(len(f.value) <= 1024 for f in e.fields)
    big_list = opportunity_list_embed([row(id=i, title=huge) for i in range(15)])
    assert len(big_list.description) <= 4096


def test_list_embed_empty():
    assert "No opportunities" in opportunity_list_embed([]).description
