"""Check tracked opportunities for meaningful new information."""

import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.db import Database
from app.ollama_client import OllamaClient, OllamaError
from app.research.analyse import Scores, overall_score
from app.research.collect import CollectedSource, normalize_url
from app.research.pipeline import ResearchService

log = logging.getLogger(__name__)

TRACK_SYSTEM = """You monitor a business opportunity for meaningful new developments.
You are given the opportunity as currently recorded and some NEW numbered sources.
"Meaningful" means something that changes the opportunity: a new tender, regulation, funding,
competitor, partnership, price/demand shift, or a concrete figure. Generic or repeated coverage is NOT meaningful.
If nothing meaningful is new, set meaningful=false and leave the other fields empty.
Never invent facts not in the sources. Reply with JSON only."""

TRACK_SCHEMA = {
    "type": "object",
    "properties": {
        "meaningful": {"type": "boolean"},
        "what_changed": {"type": "string"},
        "why_it_matters": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "integer"}},
        "scores": {
            "type": "object",
            "properties": {k: {"type": "integer"} for k in (
                "demand", "competition", "startup_difficulty", "revenue_potential", "accessibility")},
        },
    },
    "required": ["meaningful", "what_changed", "why_it_matters", "source_ids"],
}


class TrackVerdict(BaseModel):
    meaningful: bool
    what_changed: str = ""
    why_it_matters: str = ""
    source_ids: list[int] = Field(default_factory=list)
    scores: Scores | None = None

    @field_validator("meaningful", mode="before")
    @classmethod
    def _strict_bool(cls, v):
        # Don't let "false"/"no" strings be read as truthy.
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            return v.strip().lower() == "true"
        raise ValueError("meaningful must be a boolean")

    @field_validator("scores", mode="before")
    @classmethod
    def _optional_scores(cls, v):
        # Partial or empty score objects are ignored rather than failing the whole verdict.
        try:
            return Scores.model_validate(v) if v else None
        except ValidationError:
            return None


@dataclass
class TrackedUpdate:
    opportunity: dict          # row after the update
    what_changed: str
    why_it_matters: str
    sources: list[CollectedSource]
    old_score: float
    new_score: float


def parse_verdict(raw: str, num_sources: int) -> TrackVerdict | None:
    """Validated verdict, or None if not meaningful / malformed / unsupported by real sources."""
    try:
        verdict = TrackVerdict.model_validate(json.loads(raw))
    except (json.JSONDecodeError, TypeError, ValidationError) as e:
        log.info("Tracking verdict malformed: %s", e)
        return None
    if not verdict.meaningful or len(verdict.what_changed.strip()) < 10:
        return None
    verdict.source_ids = sorted({i for i in verdict.source_ids if 1 <= i <= num_sources})
    if not verdict.source_ids:
        log.info("Tracking verdict cited no valid sources; ignoring")
        return None
    return verdict


class Tracker:
    def __init__(self, research: ResearchService, db: Database, ollama: OllamaClient, max_sources: int = 6):
        self.research = research
        self.db = db
        self.ollama = ollama
        self.max_sources = max_sources

    async def check(self, opp: dict, news_timelimit: str = "m") -> TrackedUpdate | None:
        """Search for news about one opportunity. Returns an update only if something meaningful is new.
        Never raises for network/model problems (logs and returns None)."""
        try:
            return await self._check(opp, news_timelimit)
        except Exception:
            log.exception("Tracking check failed for #%s", opp.get("id"))
            return None

    async def _check(self, opp: dict, news_timelimit: str) -> TrackedUpdate | None:
        query = f"{opp['title']} {opp['country']}"
        found = await self.research.search_and_collect([query], news_timelimit, self.max_sources * 2)

        known = {normalize_url(s["url"]) for s in await self.db.get_sources(opp["id"])}
        new_sources = [s for s in found if normalize_url(s.url) not in known][: self.max_sources]
        if not new_sources:
            log.info("Tracked #%d: no new sources", opp["id"])
            return None

        lines = [
            f"OPPORTUNITY #{opp['id']}: {opp['title']} ({opp['country']}, {opp['category']})",
            f"Problem: {opp['problem']}", f"Solution: {opp['solution']}",
            f"Evidence so far: {(opp.get('evidence') or '')[-1500:]}",
            f"Current scores: demand {opp['demand']}, competition {opp['competition']}, "
            f"startup_difficulty {opp['startup_difficulty']}, revenue_potential {opp['revenue_potential']}, "
            f"accessibility {opp['accessibility']}",
            "\nNEW SOURCES:",
        ]
        for i, s in enumerate(new_sources, start=1):
            date = s.published_at[:10] if s.published_at else "undated"
            lines.append(f"\n[{i}] {s.title} ({s.source}, {date})\n{s.text or '(headline only)'}")

        try:
            raw = await self.ollama.chat(
                "\n".join(lines), system=TRACK_SYSTEM, format=TRACK_SCHEMA,
                timeout=self.research.cfg.analysis.timeout,
                options={"num_ctx": self.research.cfg.analysis.num_ctx, "temperature": 0.1},
            )
        except OllamaError as e:
            log.warning("Tracked #%d: model unavailable: %s", opp["id"], e)
            return None

        verdict = parse_verdict(raw, len(new_sources))
        if verdict is None:
            log.info("Tracked #%d: nothing meaningful in %d new sources", opp["id"], len(new_sources))
            return None

        cited = [new_sources[i - 1] for i in verdict.source_ids]
        old_score = opp["score"]
        fields = {"evidence": verdict.what_changed}
        if verdict.scores:
            fields.update(verdict.scores.model_dump())
            fields["score"] = overall_score(verdict.scores, self.research.cfg.weights)
        await self.db.update_opportunity_seen(opp["id"], fields)
        await self.db.add_sources(
            opp["id"],
            [{"title": s.title, "url": s.url, "source": s.source, "published_at": s.published_at,
              "retrieved_at": s.retrieved_at} for s in cited],
            run_id=None,
        )
        updated = await self.db.get_opportunity(opp["id"])
        log.info("Tracked #%d: meaningful update (score %s -> %s)", opp["id"], old_score, updated["score"])
        return TrackedUpdate(updated, verdict.what_changed, verdict.why_it_matters, cited,
                             old_score, updated["score"])
