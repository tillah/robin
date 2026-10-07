"""/research, /opportunities, /opportunity."""

import logging

import discord
from discord import app_commands
from discord.ext import commands

from app.db import Database
from app.embeds import opportunity_detail_embed, opportunity_list_embed, opportunity_summary_embed
from app.research.pipeline import ResearchBusy, ResearchReport, ResearchService

log = logging.getLogger(__name__)


def outcome_footer(o) -> str:
    if o.is_new:
        return "🆕 new"
    change = ""
    if o.previous_score is not None and o.previous_score != o.opportunity.overall_score:
        change = f", score was {o.previous_score}"
    return f"🔁 seen before (+{o.new_source_count} sources{change})"


def report_summary(report: ResearchReport) -> str:
    if report.status != "completed":
        return f"⚠️ Research on **{report.query}** failed: {report.error}"
    fetched = sum(s.fetched for s in report.sources)
    if not report.outcomes:
        return (f"🔎 **{report.query}** — analysed {len(report.sources)} sources ({fetched} full articles) "
                "but found no concrete opportunities. Try a more specific topic.")
    new = sum(o.is_new for o in report.outcomes)
    return (f"🔎 **{report.query}** — analysed {len(report.sources)} sources ({fetched} full articles). "
            f"Found **{len(report.outcomes)}** opportunities ({new} new, "
            f"{len(report.outcomes) - new} already known). Run #{report.run_id}.")


async def handle_research(interaction: discord.Interaction, topic: str, service: ResearchService) -> None:
    await interaction.response.defer(thinking=True)
    log.info("/research from user=%s topic=%r", interaction.user.id, topic)

    async def progress(msg: str) -> None:
        await interaction.edit_original_response(content=msg)

    try:
        report = await service.run(topic, progress=progress)
    except ResearchBusy:
        await interaction.edit_original_response(
            content="⏳ Another research run is in progress. Try again in a few minutes.")
        return

    await interaction.edit_original_response(content=report_summary(report))
    for o in report.outcomes:
        row = await service.db.get_opportunity(o.id)
        sources = await service.db.get_sources(o.id)
        await interaction.followup.send(embed=opportunity_summary_embed(row, sources, outcome_footer(o)))


class ResearchCog(commands.Cog):
    def __init__(self, bot: commands.Bot, service: ResearchService, db: Database):
        self.bot = bot
        self.service = service
        self.db = db

    async def _choices(self, values, current: str) -> list[app_commands.Choice[str]]:
        return [app_commands.Choice(name=v, value=v) for v in values if current.lower() in v.lower()][:25]

    @app_commands.command(name="research", description="Research a topic on the web and extract business opportunities")
    @app_commands.describe(topic="Sector + country works best, e.g. agriculture opportunities Zimbabwe")
    async def research(self, interaction: discord.Interaction, topic: app_commands.Range[str, 3, 200]):
        await handle_research(interaction, topic, self.service)

    @app_commands.command(name="opportunities", description="List stored opportunities")
    @app_commands.describe(
        country="Filter by country", category="Filter by category",
        min_score="Only show scores at or above this", tracked="Only tracked opportunities",
    )
    async def opportunities(
        self, interaction: discord.Interaction, country: str | None = None, category: str | None = None,
        min_score: app_commands.Range[float, 0, 10] | None = None, tracked: bool = False,
    ):
        rows = await self.db.list_opportunities(
            country=country, category=category, min_score=min_score,
            status="tracked" if tracked else None, limit=15,
        )
        filters = ", ".join(x for x in (country, category, f"≥{min_score}" if min_score else None,
                                        "tracked" if tracked else None) if x)
        title = "Opportunities" + (f" ({filters})" if filters else "")
        await interaction.response.send_message(embed=opportunity_list_embed(rows, title))

    @opportunities.autocomplete("country")
    async def _country_ac(self, interaction: discord.Interaction, current: str):
        return await self._choices(self.service.cfg.countries, current)

    @opportunities.autocomplete("category")
    async def _category_ac(self, interaction: discord.Interaction, current: str):
        return await self._choices(self.service.cfg.categories, current)

    @app_commands.command(name="opportunity", description="Show an opportunity's details and sources")
    @app_commands.describe(id="Opportunity ID (from /opportunities)")
    async def opportunity(self, interaction: discord.Interaction, id: app_commands.Range[int, 1]):
        row = await self.db.get_opportunity(id)
        if not row:
            await interaction.response.send_message(f"No opportunity with ID {id}.", ephemeral=True)
            return
        sources = await self.db.get_sources(id)
        await interaction.response.send_message(embed=opportunity_detail_embed(row, sources))
