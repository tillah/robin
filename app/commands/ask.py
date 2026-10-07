"""/ask — send a question straight to the local model."""

import logging

import discord
from discord import app_commands
from discord.ext import commands

from app.discord_utils import split_message
from app.ollama_client import OllamaClient, OllamaError

log = logging.getLogger(__name__)

ASK_SYSTEM_PROMPT = (
    "You are Robin, a concise business and technology research assistant. "
    "Answer clearly using Discord-friendly markdown. Keep answers focused."
)


async def handle_ask(interaction: discord.Interaction, question: str, ollama: OllamaClient) -> None:
    # Discord requires an acknowledgement within 3s; defer gives us 15 minutes.
    await interaction.response.defer(thinking=True)
    log.info("/ask from user=%s chars=%d", interaction.user.id, len(question))

    try:
        answer = await ollama.chat(question, system=ASK_SYSTEM_PROMPT)
    except OllamaError as e:
        log.warning("/ask failed: %s", e)
        await interaction.followup.send(f"⚠️ {e}")
        return

    if not answer:
        await interaction.followup.send("⚠️ The model returned an empty response. Try rephrasing.")
        return

    for chunk in split_message(answer):
        await interaction.followup.send(chunk)


class AskCog(commands.Cog):
    def __init__(self, bot: commands.Bot, ollama: OllamaClient):
        self.bot = bot
        self.ollama = ollama

    @app_commands.command(name="ask", description="Ask the local AI model a question")
    @app_commands.describe(question="Your question")
    async def ask(self, interaction: discord.Interaction, question: app_commands.Range[str, 1, 4000]):
        await handle_ask(interaction, question, self.ollama)
