import json

import pytest

from app.ollama_client import OllamaUnavailable
from app.research.analyse import (
    AnalysisError,
    Scores,
    analyse_sources,
    build_prompt,
    overall_score,
    parse_analysis,
)
from app.research.collect import CollectedSource
from app.research_config import load_research_config
from tests.helpers import StubOllama, analysis_reply, opp_json

CFG = load_research_config("research.toml")


def src(i):
    return CollectedSource(f"Title {i}", f"https://s{i}.com", "Pub", "2026-10-01T00:00:00+00:00",
                           f"text {i}", "2026-10-07T00:00:00+00:00", True)


def test_valid_output_is_parsed_and_normalised():
    [o] = parse_analysis(analysis_reply(opp_json()), 3, CFG)
    assert o.country == "Zimbabwe"          # case-normalised to configured name
    assert o.category == "agriculture"
    assert o.competitors == "Big Co; Other Co"  # list coerced to text
    assert 1 <= o.overall_score <= 10


@pytest.mark.parametrize("raw", ["not json at all", "", '{"foo": []}', '"just a string"', "null"])
def test_unusable_output_raises(raw):
    with pytest.raises(AnalysisError):
        parse_analysis(raw, 3, CFG)


def test_bare_list_is_accepted():
    assert len(parse_analysis(json.dumps([opp_json()]), 3, CFG)) == 1


def test_bad_items_dropped_good_kept():
    good = opp_json()
    no_title = opp_json(title="")
    no_problem = {k: v for k, v in opp_json(title="Other idea entirely").items() if k != "problem"}
    bad_scores = opp_json(title="Bad scores idea", scores={"demand": "high", "competition": 1,
                          "startup_difficulty": 1, "revenue_potential": 1, "accessibility": 1})
    missing_scores = {k: v for k, v in opp_json(title="No scores idea").items() if k != "scores"}
    hallucinated_source = opp_json(title="Cites nothing real", source_ids=[42, 99])
    no_sources = opp_json(title="No sources at all", source_ids=[])
    raw = analysis_reply(good, no_title, no_problem, bad_scores, missing_scores,
                         hallucinated_source, no_sources, "not a dict")
    result = parse_analysis(raw, 3, CFG)
    assert [o.title for o in result] == [good["title"]]


def test_scores_coerced_and_clamped():
    s = Scores(demand="7", competition=7.6, startup_difficulty=0, revenue_potential=15, accessibility=5)
    assert (s.demand, s.competition, s.startup_difficulty, s.revenue_potential) == (7, 8, 1, 10)


def test_invalid_source_ids_filtered():
    [o] = parse_analysis(analysis_reply(opp_json(source_ids=[0, 2, "3", 7, "x"])), 3, CFG)
    assert o.source_ids == [2, 3]


def test_duplicate_items_in_one_response_collapsed():
    raw = analysis_reply(opp_json(), opp_json(title="Cold storage for horticulture exporters."))
    assert len(parse_analysis(raw, 3, CFG)) == 1


def test_max_opportunities_enforced():
    titles = ["Cold storage hubs", "Drone crop spraying", "Tower security sensors", "SME cyber audits",
              "Solar irrigation leasing", "Fibre installer training", "Grain moisture meters",
              "Tender alert service", "Mobile vet clinics"]
    raw = analysis_reply(*[opp_json(title=t) for t in titles])
    assert len(parse_analysis(raw, 3, CFG)) == CFG.analysis.max_opportunities


def test_overall_score_inverts_competition_and_difficulty():
    weights = {k: 1.0 for k in ("demand", "competition", "startup_difficulty", "revenue_potential", "accessibility")}
    best = Scores(demand=10, competition=1, startup_difficulty=1, revenue_potential=10, accessibility=10)
    worst = Scores(demand=1, competition=10, startup_difficulty=10, revenue_potential=1, accessibility=1)
    assert overall_score(best, weights) == 10.0
    assert overall_score(worst, weights) == 1.0


def test_prompt_contains_sources_and_known_titles():
    prompt = build_prompt("agri Zimbabwe", [src(1), src(2)], CFG, ["Known opp"])
    assert "[1] Title 1" in prompt and "[2] Title 2" in prompt
    assert "Known opp" in prompt and "Zimbabwe" in prompt


async def test_analyse_retries_once_after_malformed_output():
    ollama = StubOllama(analysis=["garbage {", analysis_reply(opp_json())])
    result = await analyse_sources(ollama, "t", [src(1)], CFG)
    assert len(result) == 1 and ollama.analysis_calls == 2


async def test_analyse_gives_up_after_two_malformed_outputs():
    ollama = StubOllama(analysis=["garbage", '{"wrong": 1}'])
    with pytest.raises(AnalysisError):
        await analyse_sources(ollama, "t", [src(1)], CFG)


async def test_analyse_ollama_unavailable():
    ollama = StubOllama(analysis=[OllamaUnavailable("Can't reach Ollama.")])
    with pytest.raises(AnalysisError, match="reach Ollama"):
        await analyse_sources(ollama, "t", [src(1)], CFG)


async def test_analyse_no_sources_skips_model():
    ollama = StubOllama()
    assert await analyse_sources(ollama, "t", [], CFG) == []
    assert ollama.calls == []


def test_country_normalisation():
    from app.research.analyse import normalise_country
    allowed = CFG.countries
    assert normalise_country("zimbabwe", allowed) == "Zimbabwe"
    assert normalise_country("Zimbabwe, Zambia, South Africa, Botswana", allowed) == "Regional"
    assert normalise_country("Southern Africa", allowed) == "Regional"
    assert normalise_country("Kenya", allowed) == "Kenya"
