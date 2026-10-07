from app.ollama_client import OllamaUnavailable
from app.research.matching import find_match, find_match_with_model, similarity
from tests.helpers import StubOllama


def o(id, title, problem="", solution="", category="agriculture", country="Zimbabwe"):
    return dict(id=id, title=title, problem=problem, solution=solution, category=category, country=country)


MACHINERY = o(1, "AI-Driven Agricultural Machinery Solutions for Zimbabwean Farmers",
              "Only 25,000 functional tractors of 40,000 needed", "Tools to optimise tractor usage")
FINANCE = o(2, "AI-Driven Agricultural Finance Platforms for Zimbabwean Farmers",
            "Farmers lack long-term finance; pension funds could provide capital", "Platform for 3-7 year loans")
COLD = o(3, "Cold storage for horticulture exporters", "Exporters lose produce without cold chain",
         "Shared cold rooms near farms")


def test_templated_titles_are_not_merged():
    assert similarity(MACHINERY, FINANCE) < 0.5
    assert find_match(FINANCE, [MACHINERY]) is None


def test_exact_and_near_identical_titles_match():
    assert find_match(o(0, COLD["title"]), [MACHINERY, COLD])["id"] == 3
    assert find_match(o(0, "Cold storage for horticultural exporters"), [COLD])["id"] == 3


def test_same_idea_with_overlapping_wording_matches():
    new = o(0, "Shared cold storage rooms for horticulture exporters",
            "Horticulture exporters lose produce without cold chain", "Pay-per-use shared cold rooms")
    assert find_match(new, [MACHINERY, FINANCE, COLD])["id"] == 3


async def test_model_confirms_borderline_match():
    new = o(0, "Long-term agricultural financing marketplace",
            "Farmers lack long-term finance; pension funds could provide patient capital",
            "Platform matching pension fund capital with farm projects")
    ollama = StubOllama(match_reply='{"match_id": 2}')
    match = await find_match_with_model(new, [MACHINERY, FINANCE], ollama)
    assert match["id"] == 2


async def test_model_cannot_pick_an_id_it_was_not_offered():
    new = o(0, "Long-term agricultural financing marketplace", "Farmers lack long-term finance", "Loans")
    ollama = StubOllama(match_reply='{"match_id": 999}')
    assert await find_match_with_model(new, [FINANCE], ollama) is None


async def test_model_failure_or_garbage_means_no_match():
    new = o(0, "Long-term agricultural financing marketplace", "Farmers lack long-term finance", "Loans")
    for reply in (OllamaUnavailable("down"), "not json", '{"match_id": "abc"}', "[]"):
        assert await find_match_with_model(new, [FINANCE], StubOllama(match_reply=reply)) is None


async def test_unrelated_opportunity_skips_model_call():
    ollama = StubOllama(match_reply='{"match_id": 3}')
    new = o(0, "Telecom tower security monitoring", "Vandalism of towers", "Remote sensors", "telecommunications")
    assert await find_match_with_model(new, [COLD], ollama) is None
    assert ollama.calls == []
