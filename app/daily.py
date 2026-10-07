"""Daily intelligence: scheduled research, alert classification, tracking checks and reporting."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Protocol

import discord

from app.db import Database
from app.embeds import daily_report_embed, high_priority_alert_embed, tracked_update_embed
from app.research.pipeline import OpportunityOutcome, ResearchService
from app.research.tracking import Tracker
from app.research_config import AlertConfig, ResearchConfig

log = logging.getLogger(__name__)

DAILY_KIND = "daily"            # research runs made by the daily job
DAILY_JOB_KIND = "daily-job"    # one row per daily job, used to know whether today's ran


# --- Discord delivery ----------------------------------------------------------------

class Notifier(Protocol):
    async def send(self, content: str | None = None, embed: discord.Embed | None = None) -> bool: ...


class DiscordNotifier:
    """Sends to one channel. Never raises: failures are logged and reported as False."""

    def __init__(self, client: discord.Client, channel_id: int | None, retry_delays: tuple[float, ...] = (5, 30)):
        self.client = client
        self.channel_id = channel_id
        # Waits between attempts; generous because the Mac may have just woken and be reconnecting.
        self.retry_delays = retry_delays

    async def _channel(self):
        channel = self.client.get_channel(self.channel_id)
        if channel is None:
            channel = await self.client.fetch_channel(self.channel_id)
        return channel

    async def send(self, content: str | None = None, embed: discord.Embed | None = None) -> bool:
        if not self.channel_id:
            log.warning("DISCORD_REPORT_CHANNEL_ID not set; dropping notification")
            return False
        attempts = len(self.retry_delays) + 1
        for attempt in range(1, attempts + 1):
            try:
                channel = await self._channel()
                await channel.send(content=content, embed=embed)
                return True
            except (discord.Forbidden, discord.NotFound) as e:
                log.error("Cannot post to channel %s (%s). Check the ID and the bot's permissions.",
                          self.channel_id, e.__class__.__name__)
                return False
            except (discord.HTTPException, discord.InvalidData, OSError, asyncio.TimeoutError) as e:
                log.warning("Discord send failed (attempt %d/%d): %r", attempt, attempts, e)
                if attempt < attempts:
                    await asyncio.sleep(self.retry_delays[attempt - 1])
        return False


# --- Classification ------------------------------------------------------------------

@dataclass
class Classified:
    alerts: list[OpportunityOutcome] = field(default_factory=list)
    report: list[OpportunityOutcome] = field(default_factory=list)
    updated: list[OpportunityOutcome] = field(default_factory=list)
    stored_only: list[OpportunityOutcome] = field(default_factory=list)


def classify(outcomes: list[OpportunityOutcome], cfg: AlertConfig, tracked_ids: set[int] = frozenset()) -> Classified:
    """Sort outcomes into alert / report / re-confirmed / stored-only buckets.

    Only things that are new to you trigger anything: a brand-new opportunity, or a known one whose
    score has newly crossed a threshold. Tracked opportunities are reported via tracking updates instead.
    """
    latest: dict[int, OpportunityOutcome] = {}
    for o in outcomes:  # the same opportunity can come up under several topics in one day
        prev = latest.get(o.id)
        if prev is None:
            latest[o.id] = o
        else:
            # keep the most recent assessment, but remember if it was new earlier today
            o.is_new = o.is_new or prev.is_new
            o.previous_score = prev.previous_score if prev.previous_score is not None else o.previous_score
            latest[o.id] = o

    result = Classified()
    for o in latest.values():
        score = o.opportunity.overall_score
        crossed_alert = o.previous_score is not None and o.previous_score < cfg.alert_min_score
        crossed_report = o.previous_score is not None and o.previous_score < cfg.report_min_score
        if o.id in tracked_ids:
            result.updated.append(o)
        elif score >= cfg.alert_min_score and (o.is_new or crossed_alert):
            result.alerts.append(o)
        elif score >= cfg.report_min_score and (o.is_new or crossed_report):
            result.report.append(o)
        elif not o.is_new:
            result.updated.append(o)
        else:
            result.stored_only.append(o)

    result.alerts.sort(key=lambda o: o.opportunity.overall_score, reverse=True)
    # Avoid flooding: overflow alerts go into the report instead.
    overflow = result.alerts[cfg.max_alerts_per_day:]
    result.alerts = result.alerts[: cfg.max_alerts_per_day]
    result.report = sorted(overflow + result.report, key=lambda o: o.opportunity.overall_score, reverse=True)
    return result


# --- Topic selection -----------------------------------------------------------------

def daily_topics(cfg: ResearchConfig, day: date) -> list[tuple[str, str | None]]:
    """(topic, signal) pairs for `day`. Rotates through every country x category combination
    and through the signals, so coverage spreads over successive days."""
    combos = [f"{category} {country}" for country in cfg.countries for category in cfg.categories]
    n = min(cfg.schedule.topics_per_day, len(combos))
    start = (day.toordinal() * n) % len(combos)
    topics = []
    for k in range(n):
        signal = cfg.signals[(day.toordinal() * n + k) % len(cfg.signals)] if cfg.signals else None
        topics.append((combos[(start + k) % len(combos)], signal))
    topics.extend((t, None) for t in cfg.schedule.extra_topics)
    return topics


# --- The daily job -------------------------------------------------------------------

@dataclass
class DailyResult:
    day: str
    topics: list[tuple[str, str, str | None]] = field(default_factory=list)  # (topic, status, error)
    classified: Classified = field(default_factory=Classified)
    tracked_updates: int = 0
    report_sent: bool = False


class DailyIntelligence:
    def __init__(self, research: ResearchService, tracker: Tracker, db: Database, notifier: Notifier):
        self.research = research
        self.tracker = tracker
        self.db = db
        self.notifier = notifier
        self.cfg = research.cfg
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    async def run(self, day: date | None = None) -> DailyResult | None:
        """Run the full daily cycle. Returns None if a daily run is already in progress.
        Never raises: every stage is isolated so one failure can't stop the rest."""
        if self._running:
            return None
        self._running = True
        day = day or date.today()
        result = DailyResult(day=day.isoformat())
        job_id = await self.db.start_run(f"daily intelligence {day.isoformat()}", DAILY_JOB_KIND)
        try:
            await self._run(day, result)
            await self.db.finish_run(job_id, "completed",
                                     opportunities_found=len(result.classified.alerts) + len(result.classified.report))
        except Exception as e:
            log.exception("Daily job crashed")
            await self.db.finish_run(job_id, "failed", error=repr(e)[:500])
        finally:
            self._running = False
        return result

    async def _run(self, day: date, result: DailyResult) -> None:
        topics = daily_topics(self.cfg, day)
        log.info("Daily job %s: %d topics", day, len(topics))

        outcomes: list[OpportunityOutcome] = []
        for topic, signal in topics:
            report = await self.research.run(
                topic, kind=DAILY_KIND, wait=True,
                news_timelimit=self.cfg.schedule.news_timelimit, signal=signal,
            )
            result.topics.append((topic, report.status, report.error))
            outcomes.extend(report.outcomes)

        tracked = await self.db.list_opportunities(status="tracked", limit=50)
        tracked_ids = {o["id"] for o in tracked}
        result.classified = classify(outcomes, self.cfg.alerts, tracked_ids)
        c = result.classified
        log.info("Daily job %s: %d alerts, %d report, %d re-confirmed, %d stored only",
                 day, len(c.alerts), len(c.report), len(c.updated), len(c.stored_only))

        updates = []
        for opp in tracked:
            update = await self.tracker.check(opp, news_timelimit=self.cfg.schedule.news_timelimit)
            if update is not None:
                updates.append(update)

        # Report first, so the alerts and updates below have context.
        report_embed = daily_report_embed(
            day.isoformat(), result.topics,
            [await self.db.get_opportunity(o.id) for o in c.alerts],
            [await self.db.get_opportunity(o.id) for o in c.report],
            len(c.updated), len(c.stored_only), len(updates),
        )
        result.report_sent = await self.notifier.send(embed=report_embed)

        for o in c.alerts:
            row = await self.db.get_opportunity(o.id)
            await self.notifier.send(embed=high_priority_alert_embed(row, await self.db.get_sources(o.id)))

        for update in updates:
            sources = [{"title": s.title, "url": s.url, "source": s.source, "published_at": s.published_at}
                       for s in update.sources]
            sent = await self.notifier.send(embed=tracked_update_embed(
                update.opportunity, update.what_changed, update.why_it_matters, sources,
                update.old_score, update.new_score))
            result.tracked_updates += int(sent)


# --- Scheduler -----------------------------------------------------------------------

class DailyScheduler:
    """Wakes every `poll_seconds` and starts the daily job once per local day after the configured
    time. Polling (rather than one long sleep) copes with the Mac sleeping or the clock changing."""

    def __init__(self, daily: DailyIntelligence, db: Database, cfg: ResearchConfig, poll_seconds: float = 30,
                 is_ready=lambda: True):
        self.daily = daily
        # Gate on the Discord connection: after the Mac wakes, the network (and Discord) may still be
        # reconnecting, and starting then would fail every search and lose the report.
        self.is_ready = is_ready
        self.db = db
        self.cfg = cfg
        self.poll_seconds = poll_seconds
        self._task: asyncio.Task | None = None
        self._skip_day: date | None = None

    def scheduled_time(self, day: date) -> datetime:
        h, m = self.cfg.schedule.hour_minute
        return datetime(day.year, day.month, day.day, h, m)

    async def ran_on(self, day: date) -> bool:
        """True if a daily job started on this local date."""
        start = datetime(day.year, day.month, day.day).astimezone()
        end = start + timedelta(days=1)
        return await self.db.has_run_between(DAILY_JOB_KIND, start, end)

    async def due(self, now: datetime) -> bool:
        today = now.date()
        if now < self.scheduled_time(today) or self._skip_day == today or self.daily.running:
            return False
        if not self.is_ready():
            return False
        return not await self.ran_on(today)

    def start(self) -> None:
        if self._task is None:
            now = datetime.now()
            if not self.cfg.schedule.catch_up and now > self.scheduled_time(now.date()) + timedelta(minutes=5):
                self._skip_day = now.date()  # started late and catch-up is off: wait for tomorrow
            self._task = asyncio.create_task(self._loop(), name="daily-scheduler")
            log.info("Daily scheduler started: %s local time (catch_up=%s)",
                     self.cfg.schedule.time, self.cfg.schedule.catch_up)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                if await self.due(datetime.now()):
                    log.info("Daily job due; starting")
                    await self.daily.run()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Daily scheduler iteration failed")
            await asyncio.sleep(self.poll_seconds)
