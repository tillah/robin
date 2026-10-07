import pytest

from app.config import ConfigError
from app.research_config import load_research_config


def write(tmp_path, text):
    p = tmp_path / "r.toml"
    p.write_text(text)
    return str(p)


def test_default_config_loads():
    cfg = load_research_config("research.toml")
    assert "Zimbabwe" in cfg.countries and "agriculture" in cfg.categories
    assert "tenders" in cfg.signals
    assert cfg.search.max_sources > 0
    assert set(cfg.weights) == {"demand", "competition", "startup_difficulty", "revenue_potential", "accessibility"}


def test_minimal_config_uses_defaults(tmp_path):
    cfg = load_research_config(write(tmp_path, '[areas]\ncountries=["Kenya"]\ncategories=["fintech"]\n'))
    assert cfg.countries == ("Kenya",)
    assert cfg.signals == ()
    assert cfg.analysis.max_opportunities == 5


@pytest.mark.parametrize("text, msg", [
    ("not = [valid", "Invalid TOML"),
    ('[areas]\ncountries=[]\ncategories=["x"]', "at least one country"),
    ('[areas]\ncountries="Zimbabwe"\ncategories=["x"]', "list of strings"),
    ('[areas]\ncountries=["Z"]\ncategories=["x"]\n[search]\nbogus=1', "Unknown setting"),
    ('[areas]\ncountries=["Z"]\ncategories=["x"]\n[search]\nnews_timelimit="year"', "news_timelimit"),
    ('[areas]\ncountries=["Z"]\ncategories=["x"]\n[scoring.weights]\ndemand=-1', "non-negative"),
])
def test_invalid_configs(tmp_path, text, msg):
    with pytest.raises(ConfigError, match=msg):
        load_research_config(write(tmp_path, text))


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_research_config("/nonexistent/research.toml")
