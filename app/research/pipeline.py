"""Research pipeline: SEARCH -> COLLECT -> ANALYSE -> match/store in SQLite."""

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import Awaitable, Callable

import aiohttp

from app.db import Database
from app.ollama_client import OllamaClient
from app.research.analyse import AnalysisError, Opportunity, analyse_sources
from app.research.collect import CollectedSource, collect_sources
from app.research.matching import find_match_with_model
from app.research.search import (
    DuckDuckGoProvider,
    GoogleNewsRSSProvider,
    SearchProvider,
    search_all,
)
from app.research_config import ResearchConfig

log = logging.getLogger(__name__)

ProgressFn = Callable[[str], Awaitable[None]]


@dataclass
class OpportunityOutcome:
    id: int
    opportunity: Opportunity
    is_new: bool
    previous_score: float | None
    sources: list[CollectedSource]
    new_source_count: int


@dataclass
class ResearchReport:
    run_id: int
    query: str
    status: str = "running"
    sources: list[CollectedSource] = field(default_factory=list)
    outcomes: list[OpportunityOutcome] = field(default_factory=list)
    error: str | None = None


class ResearchBusy(Exception):
    pass


class ResearchService:
    def __init__(
        self,
        db: Database,
        ollama: OllamaClient,
        cfg: ResearchConfig,
        providers: list[SearchProvider] | None = None,
    ):
        self.db = db
        self.ollama = ollama
        self.cfg = cfg
        self._session: aiohttp.ClientSession | None = None
        self._fixed_providers = providers  # injected in tests
        self._providers: dict[str, list[SearchProvider]] = {}
        # One research run at a time: a local 8B model can't usefully do two at once.
        self._lock = asyncio.Lock()

    def providers_for(self, news_timelimit: str | None = None) -> list[SearchProvider]:
        if self._fixed_providers is not None:
            return self._fixed_providers
        limit = news_timelimit or self.cfg.search.news_timelimit
        if limit not in self._providers:
            when = {"d": "1d", "w": "7d", "m": "30d", "y": "1y"}[limit]
            self._providers[limit] = [
                DuckDuckGoProvider(news_timelimit=limit),
                GoogleNewsRSSProvider(self._get_session, when=when),
            ]
        return self._providers[limit]

    async def search_and_collect(
        self, queries: list[str], news_timelimit: str | None = None, max_sources: int | None = None,
    ) -> list[CollectedSource]:
        """SEARCH + COLLECT only (no analysis). Returns [] if nothing was found."""
        results = await search_all(
            self.providers_for(news_timelimit), queries, self.cfg.search.max_results_per_query
        )
        if not results:
            return []
        cfg = self.cfg.search
        if max_sources is not None:
            cfg = replace(cfg, max_sources=max_sources)
        return await collect_sources(await self._get_session(), results, cfg)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    @staticmethod
    def build_queries(topic: str, signal: str | None = None) -> list[str]:
        second = f"{topic} {signal}" if signal else f"{topic} investment market growth"
        return [topic, second]

    async def run(
        self, topic: str, kind: str = "manual", progress: ProgressFn | None = None,
        wait: bool = False, news_timelimit: str | None = None, signal: str | None = None,
    ) -> ResearchReport:
        """Run one research cycle. Never raises for research failures; see report.status/error.
        Raises ResearchBusy if another run is in progress and wait=False."""
        if self.busy and not wait:
            raise ResearchBusy()
        async with self._lock:
            return await self._run(topic, kind, progress, news_timelimit, signal)

    async def _run(
        self, topic: str, kind: str, progress: ProgressFn | None,
        news_timelimit: str | None, signal: str | None,
    ) -> ResearchReport:
        async def say(msg: str) -> None:
            if progress:
                try:
                    await progress(msg)
                except Exception as e:  # a progress update failing must not kill the run
                    log.debug("Progress callback failed: %r", e)

        run_id = await self.db.start_run(topic, kind)
        report = ResearchReport(run_id=run_id, query=topic)
        log.info("Research run %d started (%s): %r", run_id, kind, topic)

        try:
            await say("🔎 Searching the web…")
            report.sources = await self.search_and_collect(self.build_queries(topic, signal), news_timelimit)
            if not report.sources:
                raise AnalysisError(
                    "No usable search results. The network may be down, the search providers may be "
                    "rate-limiting, or everything was filtered out as noise."
                )
            fetched = sum(s.fetched for s in report.sources)
            log.info("Run %d: %d sources (%d with full text)", run_id, len(report.sources), fetched)

            await say(f"🧠 Analysing {len(report.sources)} sources with {self.ollama.model} "
                      "(this can take a few minutes)…")
            existing = await self.db.recent_titles(list(self.cfg.countries))
            opportunities = await analyse_sources(self.ollama, topic, report.sources, self.cfg, existing)

            report.outcomes = await self._store(opportunities, report.sources, run_id)
            report.status = "completed"
            await self.db.finish_run(run_id, "completed", len(report.sources), len(report.outcomes))
            log.info("Run %d completed: %d opportunities (%d new)", run_id, len(report.outcomes),
                     sum(o.is_new for o in report.outcomes))
        except AnalysisError as e:
            report.status, report.error = "failed", str(e)
            log.warning("Run %d failed: %s", run_id, e)
            await self.db.finish_run(run_id, "failed", len(report.sources), 0, str(e))
        except Exception as e:
            report.status, report.error = "failed", "Unexpected error during research (see logs)."
            log.exception("Run %d crashed", run_id)
            try:
                await self.db.finish_run(run_id, "failed", len(report.sources), 0, repr(e)[:500])
            except Exception:
                log.exception("Could not record failed run %d", run_id)
        return report

    async def _store(
        self, opportunities: list[Opportunity], sources: list[CollectedSource], run_id: int
    ) -> list[OpportunityOutcome]:
        outcomes = []
        created_this_run: set[int] = set()
        for opp in opportunities:
            fields = opp.model_dump(exclude={"source_ids", "scores", "overall_score", "score_rationale"})
            fields.update(opp.scores.model_dump())
            fields["score"] = opp.overall_score
            cited = [sources[i - 1] for i in opp.source_ids]

            # The model already separated this run's opportunities; only match against older ones.
            candidates = [c for c in await self.db.candidates_for_match(opp.country)
                          if c["id"] not in created_this_run]
            match = await find_match_with_model(fields, candidates, self.ollama)
            if match:
                opp_id, previous = match["id"], match["score"]
                await self.db.update_opportunity_seen(opp_id, fields)
                log.info("Matched %r to existing #%d %r", opp.title, opp_id, match["title"])
            else:
                opp_id, previous = await self.db.insert_opportunity(fields), None
                created_this_run.add(opp_id)

            added = await self.db.add_sources(
                opp_id,
                [{"title": s.title, "url": s.url, "source": s.source,
                  "published_at": s.published_at, "retrieved_at": s.retrieved_at} for s in cited],
                run_id,
            )
            outcomes.append(OpportunityOutcome(opp_id, opp, match is None, previous, cited, added))
        return outcomes
