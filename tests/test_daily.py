"""Daily intelligence: thresholds, topic rotation, tracking, delivery and scheduling."""

import asyncio
import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import discord
import pytest

from app.daily import (
    DAILY_JOB_KIND,
    DailyIntelligence,
    DailyScheduler,
    DiscordNotifier,
    classify,
    daily_topics,
)
from app.db import Database
from app.ollama_client import OllamaUnavailable
from app.research.analyse import Opportunity
from app.research.collect import CollectedSource
from app.research.pipeline import OpportunityOutcome, ResearchService
from app.research.tracking import Tracker, parse_verdict
from app.research_config import AlertConfig, load_research_config
from tests.helpers import StubOllama, analysis_reply, opp_json
from tests.test_pipeline import StubProvider

BASE_CFG = load_research_config("research.toml")
CFG = replace(BASE_CFG, schedule=replace(BASE_CFG.schedule, topics_per_day=2, extra_topics=()))
ALERTS = AlertConfig(report_min_score=5.0, alert_min_score=7.0, max_alerts_per_day=2)

HIGH = {"demand": 10, "competition": 2, "startup_difficulty": 2, "revenue_potential": 10, "accessibility": 9}
MID = {"demand": 8, "competition": 4, "startup_difficulty": 6, "revenue_potential": 7, "accessibility": 6}
LOW = {"demand": 3, "competition": 9, "startup_difficulty": 9, "revenue_potential": 3, "accessibility": 2}


class FakeNotifier:
    def __init__(self, ok: bool = True):
        self.ok = ok
        self.sent: list = []

    async def send(self, content=None, embed=None) -> bool:
        self.sent.append(embed if embed is not None else content)
        return self.ok

    @property
    def titles(self) -> list[str]:
        return [e.title for e in self.sent if isinstance(e, discord.Embed)]


def outcome(id, score, is_new=True, previous=None):
    opp = Opportunity.model_validate(opp_json(title=f"Opportunity {id}"))
    opp.overall_score = score
    return OpportunityOutcome(id, opp, is_new, previous, [], 1)


# --- thresholds ---------------------------------------------------------------------

def test_thresholds_bucket_new_opportunities():
    c = classify([outcome(1, 4.9), outcome(2, 5.0), outcome(3, 6.9), outcome(4, 7.0)], ALERTS)
    assert [o.id for o in c.stored_only] == [1]
    assert sorted(o.id for o in c.report) == [2, 3]
    assert [o.id for o in c.alerts] == [4]


def test_known_opportunities_do_not_re_alert():
    c = classify([outcome(1, 8.0, is_new=False, previous=7.5), outcome(2, 6.0, is_new=False, previous=6.0)], ALERTS)
    assert c.alerts == [] and c.report == []
    assert sorted(o.id for o in c.updated) == [1, 2]


def test_crossing_a_threshold_counts_as_news():
    c = classify([outcome(1, 7.4, is_new=False, previous=6.5), outcome(2, 5.5, is_new=False, previous=4.0)], ALERTS)
    assert [o.id for o in c.alerts] == [1]
    assert [o.id for o in c.report] == [2]


def test_alert_cap_moves_overflow_to_report():
    c = classify([outcome(i, 7.0 + i / 10) for i in range(1, 6)], ALERTS)
    assert [o.id for o in c.alerts] == [5, 4]          # best two
    assert [o.id for o in c.report][:3] == [3, 2, 1]


def test_tracked_opportunities_never_alert_from_daily_research():
    c = classify([outcome(1, 9.0)], ALERTS, tracked_ids={1})
    assert c.alerts == [] and [o.id for o in c.updated] == [1]


def test_same_opportunity_twice_in_one_day_counted_once():
    c = classify([outcome(1, 6.0), outcome(1, 7.5, is_new=False, previous=6.0)], ALERTS)
    assert [o.id for o in c.alerts] == [1] and c.report == []


# --- topic rotation -----------------------------------------------------------------

def test_topic_rotation_covers_everything_and_uses_signals():
    cfg = replace(BASE_CFG, schedule=replace(BASE_CFG.schedule, topics_per_day=6, extra_topics=("custom topic",)))
    total = len(cfg.countries) * len(cfg.categories)
    seen, signals = set(), set()
    start = date(2026, 10, 8)
    for d in range(total // 6):
        topics = daily_topics(cfg, start + timedelta(days=d))
        assert topics[-1] == ("custom topic", None)
        seen.update(t for t, _ in topics[:-1])
        signals.update(s for _, s in topics[:-1])
    assert len(seen) == total
    assert signals <= set(cfg.signals) and len(signals) > 1


# --- tracking -----------------------------------------------------------------------

def src(i, url=None):
    return CollectedSource(f"News {i}", url or f"https://news{i}.com/a", "Pub", "2026-10-07T00:00:00+00:00",
                           f"Article text {i}", "2026-10-07T00:00:00+00:00", True)


class StubResearch:
    def __init__(self, sources):
        self.sources = sources
        self.cfg = CFG

    async def search_and_collect(self, queries, news_timelimit=None, max_sources=None):
        return self.sources


@pytest.fixture
async def db():
    d = Database(":memory:")
    await d.connect()
    yield d
    await d.close()


async def tracked_opp(db) -> dict:
    opp_id = await db.insert_opportunity(dict(
        title="Cold storage", country="Zimbabwe", category="agriculture", problem="p" * 20,
        solution="s" * 20, evidence="Old evidence", score=6.8, **MID))
    await db.add_sources(opp_id, [{"title": "Old", "url": "https://old.com/a"}], None)
    await db.set_status(opp_id, "tracked")
    return await db.get_opportunity(opp_id)


def verdict(**kw):
    base = {"meaningful": True, "what_changed": "Government opened a cold-chain tender worth $2m.",
            "why_it_matters": "Direct revenue opportunity.", "source_ids": [1], "scores": HIGH}
    base.update(kw)
    return json.dumps(base)


async def test_tracking_no_new_sources_skips_model(db):
    opp = await tracked_opp(db)
    ollama = StubOllama()
    tracker = Tracker(StubResearch([src(0, "https://www.old.com/a/?utm_source=x")]), db, ollama)
    assert await tracker.check(opp) is None
    assert ollama.calls == []


async def test_tracking_meaningful_update_is_stored(db):
    opp = await tracked_opp(db)
    tracker = Tracker(StubResearch([src(1)]), db, StubOllama(analysis=[verdict()]))
    update = await tracker.check(opp)
    assert update and update.old_score == 6.8 and update.new_score > 9
    row = await db.get_opportunity(opp["id"])
    assert "tender worth $2m" in row["evidence"] and row["evidence"].startswith("Old evidence")
    assert len(await db.get_sources(opp["id"])) == 2


@pytest.mark.parametrize("reply", [
    verdict(meaningful=False),
    verdict(meaningful="false"),
    verdict(what_changed=""),
    verdict(source_ids=[7]),            # cites a source that doesn't exist
    "not json",
    json.dumps({"what_changed": "x"}),   # missing 'meaningful'
])
async def test_tracking_ignores_unmeaningful_or_malformed(db, reply):
    opp = await tracked_opp(db)
    tracker = Tracker(StubResearch([src(1)]), db, StubOllama(analysis=[reply]))
    assert await tracker.check(opp) is None
    assert (await db.get_opportunity(opp["id"]))["score"] == 6.8


async def test_tracking_partial_scores_keep_old_score(db):
    opp = await tracked_opp(db)
    tracker = Tracker(StubResearch([src(1)]), db, StubOllama(analysis=[verdict(scores={"demand": 9})]))
    update = await tracker.check(opp)
    assert update and update.new_score == 6.8


async def test_tracking_ollama_down_returns_none(db):
    opp = await tracked_opp(db)
    tracker = Tracker(StubResearch([src(1)]), db, StubOllama(analysis=[OllamaUnavailable("down")]))
    assert await tracker.check(opp) is None


def test_parse_verdict_string_true_accepted():
    assert parse_verdict(verdict(meaningful="true"), 1) is not None


# --- full daily flow -----------------------------------------------------------------

def daily(db, ollama, notifier, provider=None):
    research = ResearchService(db, ollama, CFG, providers=[provider or StubProvider()])
    return DailyIntelligence(research, Tracker(research, db, ollama), db, notifier), research


async def test_daily_flow_report_then_alerts(db):
    ollama = StubOllama(analysis=[
        analysis_reply(opp_json(title="Cold storage hubs", scores=HIGH),
                       opp_json(title="Drone crop spraying", scores=MID),
                       opp_json(title="Fax machine repair", scores=LOW)),
        analysis_reply(),  # second topic finds nothing
    ])
    notifier = FakeNotifier()
    job, research = daily(db, ollama, notifier)
    result = await job.run(date(2026, 10, 8))
    await research.close()

    assert [s for _, s, _ in result.topics] == ["completed", "completed"]
    assert notifier.titles[0].startswith("📊 Daily intelligence · 2026-10-08")
    assert len(notifier.titles) == 2 and "High-priority" in notifier.titles[1] and "Cold storage" in notifier.titles[1]
    report = notifier.sent[0]
    worth = next(f for f in report.fields if "Worth a look" in f.name)
    assert "Drone crop spraying" in worth.value and "Fax" not in str([f.value for f in report.fields])
    assert len(await db.list_opportunities()) == 3   # low score still stored
    assert await db.has_run_between(DAILY_JOB_KIND, datetime(2026, 1, 1).astimezone(),
                                    datetime(2030, 1, 1).astimezone())


async def test_daily_second_day_does_not_repeat_alerts(db):
    reply = analysis_reply(opp_json(title="Cold storage hubs", scores=HIGH))
    notifier = FakeNotifier()
    job, research = daily(db, StubOllama(analysis=[reply, analysis_reply(), reply, analysis_reply()]), notifier)
    await job.run(date(2026, 10, 8))
    notifier.sent.clear()
    await job.run(date(2026, 10, 9))
    await research.close()
    assert len(notifier.sent) == 1   # just the report; no repeated alert
    assert "No new opportunities" in notifier.sent[0].description


async def test_daily_survives_failures_and_sends_tracked_update(db):
    opp = await tracked_opp(db)
    ollama = StubOllama(analysis=[
        OllamaUnavailable("Can't reach Ollama."),   # topic 1 fails
        analysis_reply(),                           # topic 2 ok, nothing found
        verdict(),                                  # tracked opportunity check
    ])
    notifier = FakeNotifier()
    provider = StubProvider()
    job, research = daily(db, ollama, notifier, provider)
    result = await job.run(date(2026, 10, 8))
    await research.close()

    assert [s for _, s, _ in result.topics] == ["failed", "completed"]
    assert "reach Ollama" in result.topics[0][2]
    assert notifier.titles[0].startswith("📊")
    assert notifier.titles[1].startswith("🚨 Opportunity Update · #%d" % opp["id"])
    update = notifier.sent[1]
    assert "tender" in update.fields[0].value and "→" in update.fields[2].value
    assert result.tracked_updates == 1


async def test_daily_discord_failure_does_not_crash(db):
    notifier = FakeNotifier(ok=False)
    job, research = daily(db, StubOllama(analysis=[analysis_reply(opp_json(scores=HIGH)), analysis_reply()]), notifier)
    result = await job.run(date(2026, 10, 8))
    await research.close()
    assert result.report_sent is False and len(notifier.sent) == 2


async def test_daily_runs_are_not_concurrent(db):
    job, research = daily(db, StubOllama(analysis=[analysis_reply()] * 2), FakeNotifier(), StubProvider(delay=0.1))
    first = asyncio.create_task(job.run(date(2026, 10, 8)))
    await asyncio.sleep(0.02)
    assert await job.run(date(2026, 10, 8)) is None
    assert (await first) is not None
    await research.close()


# --- Discord delivery ---------------------------------------------------------------

class FakeChannel:
    def __init__(self, errors):
        self.errors = list(errors)
        self.sent = []

    async def send(self, content=None, embed=None):
        if self.errors:
            raise self.errors.pop(0)
        self.sent.append(content)


def fake_client(channel):
    return SimpleNamespace(get_channel=lambda _id: channel)


def http_error(status=500):
    return discord.HTTPException(SimpleNamespace(status=status, reason="err"), "boom")


async def test_notifier_retries_transient_errors():
    ch = FakeChannel([http_error()])
    assert await DiscordNotifier(fake_client(ch), 1, retry_delays=(0, 0)).send("hi") is True
    assert ch.sent == ["hi"]


async def test_notifier_recovers_on_third_attempt():
    ch = FakeChannel([http_error(), OSError("network down")])
    assert await DiscordNotifier(fake_client(ch), 1, retry_delays=(0, 0)).send("hi") is True


async def test_notifier_gives_up_after_three_failures():
    ch = FakeChannel([http_error(), http_error(), http_error()])
    assert await DiscordNotifier(fake_client(ch), 1, retry_delays=(0, 0)).send("hi") is False


async def test_notifier_forbidden_does_not_retry():
    ch = FakeChannel([discord.Forbidden(SimpleNamespace(status=403, reason="no"), "missing access"), None])
    assert await DiscordNotifier(fake_client(ch), 1, retry_delays=(0, 0)).send("hi") is False
    assert len(ch.errors) == 1


async def test_notifier_without_channel():
    assert await DiscordNotifier(fake_client(None), None).send("hi") is False


# --- scheduling ---------------------------------------------------------------------

async def test_scheduler_due_logic(db):
    job = SimpleNamespace(running=False)
    cfg = replace(CFG, schedule=replace(CFG.schedule, time="08:00"))
    sched = DailyScheduler(job, db, cfg)
    today = date.today()
    assert await sched.due(datetime(today.year, today.month, today.day, 7, 59)) is False
    assert await sched.due(datetime(today.year, today.month, today.day, 8, 0)) is True
    job.running = True
    assert await sched.due(datetime(today.year, today.month, today.day, 9, 0)) is False
    job.running = False
    connected = [False]
    sched.is_ready = lambda: connected[0]
    assert await sched.due(datetime(today.year, today.month, today.day, 9, 0)) is False  # Discord not back yet
    connected[0] = True
    assert await sched.due(datetime(today.year, today.month, today.day, 9, 0)) is True
    run_id = await db.start_run("daily", DAILY_JOB_KIND)
    await db.finish_run(run_id, "completed")
    assert await sched.due(datetime(today.year, today.month, today.day, 9, 0)) is False


async def test_scheduler_loop_runs_once_per_day(db):
    runs = []

    class Job:
        running = False

        async def run(self):
            runs.append(1)
            run_id = await db.start_run("daily", DAILY_JOB_KIND)
            await db.finish_run(run_id, "completed")

    cfg = replace(CFG, schedule=replace(CFG.schedule, time="00:00", catch_up=True))
    sched = DailyScheduler(Job(), db, cfg, poll_seconds=0.01)
    sched.start()
    await asyncio.sleep(0.1)
    await sched.stop()
    assert runs == [1]


async def test_scheduler_without_catch_up_waits_for_tomorrow(db):
    runs = []

    class Job:
        running = False

        async def run(self):
            runs.append(1)

    cfg = replace(CFG, schedule=replace(CFG.schedule, time="00:00", catch_up=False))
    sched = DailyScheduler(Job(), db, cfg, poll_seconds=0.01)
    if datetime.now() < datetime.combine(date.today(), datetime.min.time()) + timedelta(minutes=6):
        pytest.skip("too close to midnight for this test")
    sched.start()
    await asyncio.sleep(0.05)
    await sched.stop()
    assert runs == []


async def test_scheduler_survives_job_exceptions(db):
    calls = []

    class Job:
        running = False

        async def run(self):
            calls.append(1)
            raise RuntimeError("boom")

    cfg = replace(CFG, schedule=replace(CFG.schedule, time="00:00"))
    sched = DailyScheduler(Job(), db, cfg, poll_seconds=0.01)
    sched.start()
    await asyncio.sleep(0.05)
    assert sched._task is not None and not sched._task.done()
    await sched.stop()
    assert len(calls) >= 2   # kept retrying rather than dying
