"""Loads research areas, search, analysis and scoring settings from research.toml."""

import os
import tomllib
from dataclasses import dataclass, field

from app.config import ConfigError

SCORE_KEYS = ("demand", "competition", "startup_difficulty", "revenue_potential", "accessibility")


@dataclass(frozen=True)
class SearchConfig:
    max_results_per_query: int = 8
    news_timelimit: str = "y"
    max_sources: int = 12
    fetch_timeout: float = 15
    fetch_concurrency: int = 4
    max_chars_per_source: int = 2500
    blocked_domains: tuple[str, ...] = ()
    blocked_title_words: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisConfig:
    max_opportunities: int = 5
    founder_profile: str = "A small, tech-capable founder in Southern Africa with limited starting capital."
    num_ctx: int = 16384
    timeout: float = 600


@dataclass(frozen=True)
class ScheduleConfig:
    enabled: bool = False
    time: str = "08:00"
    catch_up: bool = True
    topics_per_day: int = 6
    extra_topics: tuple[str, ...] = ()
    news_timelimit: str = "w"

    @property
    def hour_minute(self) -> tuple[int, int]:
        h, m = self.time.split(":")
        return int(h), int(m)


@dataclass(frozen=True)
class AlertConfig:
    report_min_score: float = 5.0
    alert_min_score: float = 7.0
    max_alerts_per_day: int = 3


@dataclass(frozen=True)
class ResearchConfig:
    countries: tuple[str, ...]
    categories: tuple[str, ...]
    signals: tuple[str, ...]
    search: SearchConfig = field(default_factory=SearchConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    weights: dict[str, float] = field(default_factory=lambda: {k: 0.2 for k in SCORE_KEYS})
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)


def _section(data: dict, name: str) -> dict:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a table in research config")
    return value


def _str_list(section: dict, key: str, where: str) -> tuple[str, ...]:
    value = section.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where}.{key} must be a list of strings")
    return tuple(v.strip() for v in value if v.strip())


def _known_fields(cls, section: dict, where: str) -> dict:
    allowed = cls.__dataclass_fields__.keys()
    unknown = set(section) - set(allowed)
    if unknown:
        raise ConfigError(f"Unknown setting(s) in [{where}]: {', '.join(sorted(unknown))}")
    out = {}
    for key, value in section.items():
        out[key] = tuple(value) if isinstance(value, list) else value
    return out


def load_research_config(path: str | None = None) -> ResearchConfig:
    path = path or os.getenv("RESEARCH_CONFIG", "research.toml")
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"Research config not found: {path}")
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {path}: {e}")

    areas = _section(data, "areas")
    countries = _str_list(areas, "countries", "areas")
    categories = _str_list(areas, "categories", "areas")
    if not countries or not categories:
        raise ConfigError("research config needs at least one country and one category")

    search = SearchConfig(**_known_fields(SearchConfig, _section(data, "search"), "search"))
    if search.news_timelimit not in ("d", "w", "m", "y"):
        raise ConfigError("search.news_timelimit must be one of d, w, m, y")
    analysis = AnalysisConfig(**_known_fields(AnalysisConfig, _section(data, "analysis"), "analysis"))

    raw_weights = _section(data, "scoring").get("weights", {})
    weights = {}
    for key in SCORE_KEYS:
        value = raw_weights.get(key, 0.2)
        if not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"scoring.weights.{key} must be a non-negative number")
        weights[key] = float(value)
    if sum(weights.values()) <= 0:
        raise ConfigError("scoring weights must not all be zero")

    schedule = ScheduleConfig(**_known_fields(ScheduleConfig, _section(data, "schedule"), "schedule"))
    try:
        h, m = schedule.hour_minute
        if not (0 <= h < 24 and 0 <= m < 60):
            raise ValueError
    except ValueError:
        raise ConfigError(f'schedule.time must be "HH:MM" (24h), got {schedule.time!r}')
    if schedule.news_timelimit not in ("d", "w", "m", "y"):
        raise ConfigError("schedule.news_timelimit must be one of d, w, m, y")
    if schedule.topics_per_day < 0:
        raise ConfigError("schedule.topics_per_day must be >= 0")

    alerts = AlertConfig(**_known_fields(AlertConfig, _section(data, "alerts"), "alerts"))
    if not alerts.report_min_score <= alerts.alert_min_score:
        raise ConfigError("alerts.report_min_score must be <= alerts.alert_min_score")

    return ResearchConfig(
        countries=countries,
        categories=categories,
        signals=_str_list(areas, "signals", "areas"),
        search=search,
        analysis=analysis,
        weights=weights,
        schedule=schedule,
        alerts=alerts,
    )
