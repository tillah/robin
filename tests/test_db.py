import pytest

from app.db import Database


@pytest.fixture
async def db(tmp_path):
    d = Database(str(tmp_path / "sub" / "t.db"))
    await d.connect()
    yield d
    await d.close()


def fields(title="Cold storage", score=6.5, **kw):
    base = dict(title=title, country="Zimbabwe", category="agriculture", problem="p" * 20,
                solution="s" * 20, evidence="Original evidence", demand=8, competition=4,
                startup_difficulty=6, revenue_potential=7, accessibility=6, score=score)
    base.update(kw)
    return base


async def test_persistence_across_connections(tmp_path):
    path = str(tmp_path / "persist.db")
    d = Database(path)
    await d.connect()
    opp_id = await d.insert_opportunity(fields())
    await d.close()

    d2 = Database(path)
    await d2.connect()
    row = await d2.get_opportunity(opp_id)
    await d2.close()
    assert row["title"] == "Cold storage" and row["status"] == "active" and row["seen_count"] == 1


async def test_sources_are_unique_per_opportunity(db):
    opp_id = await db.insert_opportunity(fields())
    s = {"title": "T", "url": "https://a.com/1", "source": "A", "published_at": None}
    assert await db.add_sources(opp_id, [s, s], run_id=None) == 1
    assert await db.add_sources(opp_id, [s, {**s, "url": "https://a.com/2"}], run_id=None) == 1
    assert len(await db.get_sources(opp_id)) == 2


async def test_update_keeps_text_appends_evidence_and_updates_scores(db):
    opp_id = await db.insert_opportunity(fields())
    await db.update_opportunity_seen(opp_id, fields(title="Renamed", solution="new solution text",
                                                    evidence="New fact: exports up 20%", score=8.1, demand=10))
    row = await db.get_opportunity(opp_id)
    assert row["title"] == "Cold storage"            # stable
    assert row["solution"] == "s" * 20                # stable
    assert row["evidence"].startswith("Original evidence")
    assert "New fact: exports up 20%" in row["evidence"]
    assert row["score"] == 8.1 and row["demand"] == 10
    assert row["seen_count"] == 2

    # The same evidence again is not duplicated.
    await db.update_opportunity_seen(opp_id, fields(evidence="New fact: exports up 20%"))
    row = await db.get_opportunity(opp_id)
    assert row["evidence"].count("New fact") == 1


async def test_list_filters_and_order(db):
    await db.insert_opportunity(fields("A", 5.0))
    b = await db.insert_opportunity(fields("B", 8.0, country="Zambia"))
    c = await db.insert_opportunity(fields("C", 7.0, category="AI"))
    await db.set_status(c, "tracked")
    assert [r["title"] for r in await db.list_opportunities()] == ["B", "C", "A"]
    assert [r["title"] for r in await db.list_opportunities(country="zambia")] == ["B"]
    assert [r["title"] for r in await db.list_opportunities(min_score=6)] == ["B", "C"]
    assert [r["title"] for r in await db.list_opportunities(status="tracked")] == ["C"]
    await db.set_status(b, "archived")
    assert "B" not in [r["title"] for r in await db.list_opportunities()]
    assert await db.set_status(9999, "tracked") is False


async def test_research_runs(db):
    run_id = await db.start_run("topic")
    assert (await db.get_run(run_id))["status"] == "running"
    await db.finish_run(run_id, "failed", 3, 0, "boom")
    run = await db.get_run(run_id)
    assert run["status"] == "failed" and run["error"] == "boom" and run["completed_at"]
