"""Discord bot setup: wires services into command cogs and syncs slash commands."""

import logging
import math

import discord
from discord import app_commands
from discord.ext import commands

from app.commands.ask import AskCog
from app.commands.research import ResearchCog
from app.commands.tracking import TrackingCog
from app.config import Settings
from app.daily import DailyIntelligence, DailyScheduler, DiscordNotifier
from app.db import Database
from app.ollama_client import OllamaClient
from app.research.pipeline import ResearchService
from app.research.tracking import Tracker
from app.research_config import ResearchConfig

log = logging.getLogger(__name__)


class RobinBot(commands.Bot):
    def __init__(self, settings: Settings, research_config: ResearchConfig):
        # Slash commands need no privileged intents.
        super().__init__(command_prefix=commands.when_mentioned, intents=discord.Intents.default())
        self.settings = settings
        self.ollama = OllamaClient(
            base_url=settings.ollama_url,
            model=settings.ollama_model,
            timeout=settings.ollama_timeout,
            think=settings.ollama_think,
        )
        self.db = Database(settings.db_path)
        self.research = ResearchService(self.db, self.ollama, research_config)
        self.notifier = DiscordNotifier(self, settings.report_channel_id)
        self.daily = DailyIntelligence(
            self.research, Tracker(self.research, self.db, self.ollama), self.db, self.notifier
        )
        self.scheduler = DailyScheduler(self.daily, self.db, research_config, is_ready=self._discord_connected)
        self.tree.on_error = self.on_app_command_error

    def _discord_connected(self) -> bool:
        # latency is inf until a heartbeat has been acknowledged on the current connection.
        return self.is_ready() and not self.is_closed() and math.isfinite(self.latency)

    async def setup_hook(self) -> None:
        await self.db.connect()
        log.info("Database ready at %s", self.settings.db_path)
        await self.add_cog(AskCog(self, self.ollama))
        await self.add_cog(ResearchCog(self, self.research, self.db))
        await self.add_cog(TrackingCog(self, self.db, self.daily, self.settings.report_channel_id))

        if await self.ollama.is_available():
            log.info("Ollama reachable at %s with model %s", self.settings.ollama_url, self.settings.ollama_model)
        else:
            log.warning(
                "Ollama not reachable or model %s missing at %s — /ask will fail until it is.",
                self.settings.ollama_model, self.settings.ollama_url,
            )

        if self.settings.discord_guild_id:
            guild = discord.Object(id=self.settings.discord_guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            log.info("Synced %d commands to guild %s", len(synced), self.settings.discord_guild_id)
        else:
            synced = await self.tree.sync()
            log.info("Synced %d global commands", len(synced))

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s) in %d server(s)", self.user, self.user.id, len(self.guilds))
        # on_ready can fire again after reconnects; start() is idempotent.
        schedule = self.research.cfg.schedule
        if not schedule.enabled:
            log.info("Daily schedule disabled in research config")
        elif not self.settings.report_channel_id:
            log.warning("Daily schedule enabled but DISCORD_REPORT_CHANNEL_ID is not set; not scheduling")
        else:
            self.scheduler.start()

    async def close(self) -> None:
        await self.scheduler.stop()
        await super().close()
        await self.research.close()
        await self.ollama.close()
        await self.db.close()

    async def on_app_command_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        log.exception("Unhandled error in /%s", interaction.command.name if interaction.command else "?",
                      exc_info=error)
        message = "⚠️ Something went wrong handling that command. Check the bot logs."
        try:
            if interaction.response.is_done():
                await interaction.followup.send(message, ephemeral=True)
            else:
                await interaction.response.send_message(message, ephemeral=True)
        except discord.HTTPException:
            log.warning("Could not deliver error message to Discord")
