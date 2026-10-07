"""/track, /untrack and /daily."""

import asyncio
import logging

import discord
from discord import app_commands
from discord.ext import commands

from app.daily import DailyIntelligence
from app.db import Database

log = logging.getLogger(__name__)


async def handle_track(interaction: discord.Interaction, opp_id: int, db: Database, track: bool) -> None:
    row = await db.get_opportunity(opp_id)
    if not row:
        await interaction.response.send_message(f"No opportunity with ID {opp_id}.", ephemeral=True)
        return
    if track:
        if row["status"] == "tracked":
            msg = f"📌 #{opp_id} **{row['title']}** is already tracked."
        else:
            await db.set_status(opp_id, "tracked")
            msg = (f"📌 Now tracking #{opp_id} **{row['title']}**. Each daily run will check it for news, "
                   "and you'll get an 🚨 Opportunity Update only when something meaningful changes.")
    else:
        if row["status"] != "tracked":
            msg = f"#{opp_id} isn't tracked."
        else:
            await db.set_status(opp_id, "active")
            msg = f"Stopped tracking #{opp_id} **{row['title']}**."
    await interaction.response.send_message(msg)


class TrackingCog(commands.Cog):
    def __init__(self, bot: commands.Bot, db: Database, daily: DailyIntelligence, report_channel_id: int | None):
        self.bot = bot
        self.db = db
        self.daily = daily
        self.report_channel_id = report_channel_id
        self._daily_task: asyncio.Task | None = None

    @app_commands.command(name="track", description="Track an opportunity and get alerts when it changes")
    @app_commands.describe(id="Opportunity ID (from /opportunities)")
    async def track(self, interaction: discord.Interaction, id: app_commands.Range[int, 1]):
        await handle_track(interaction, id, self.db, track=True)

    @app_commands.command(name="untrack", description="Stop tracking an opportunity")
    @app_commands.describe(id="Opportunity ID")
    async def untrack(self, interaction: discord.Interaction, id: app_commands.Range[int, 1]):
        await handle_track(interaction, id, self.db, track=False)

    @app_commands.command(name="daily", description="Run the daily intelligence report now")
    async def daily_now(self, interaction: discord.Interaction):
        if not self.report_channel_id:
            await interaction.response.send_message(
                "Set DISCORD_REPORT_CHANNEL_ID in .env first so I know where to post the report.", ephemeral=True)
            return
        if self.daily.running:
            await interaction.response.send_message("⏳ The daily run is already in progress.", ephemeral=True)
            return
        await interaction.response.send_message(
            f"🗓️ Starting the daily intelligence run. This takes a while (about 2 minutes per topic); "
            f"the report will be posted in <#{self.report_channel_id}>.")
        log.info("/daily triggered by user=%s", interaction.user.id)
        self._daily_task = asyncio.create_task(self.daily.run(), name="daily-manual")
