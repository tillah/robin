"""Tests the /ask handler with a fake Discord interaction."""

import discord
import pytest

from app.commands.ask import handle_ask
from app.ollama_client import OllamaClient, OllamaUnavailable
from tests.helpers import FakeInteraction


class StubOllama:
    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error

    async def chat(self, prompt, **kwargs):
        if self.error:
            raise self.error
        return self.reply


async def test_ask_success():
    inter = FakeInteraction()
    await handle_ask(inter, "hello", StubOllama(reply="Hi there"))
    assert inter.deferred
    assert inter.texts == ["Hi there"]


async def test_ask_long_reply_is_split():
    inter = FakeInteraction()
    await handle_ask(inter, "hello", StubOllama(reply=("word " * 1000).strip()))
    assert len(inter.sent) == 3
    assert all(len(m) <= 2000 for m in inter.texts)


async def test_ask_ollama_unavailable_reports_error():
    inter = FakeInteraction()
    await handle_ask(inter, "hello", StubOllama(error=OllamaUnavailable("Can't reach Ollama.")))
    assert inter.texts == ["⚠️ Can't reach Ollama."]


async def test_ask_empty_reply():
    inter = FakeInteraction()
    await handle_ask(inter, "hello", StubOllama(reply=""))
    assert "empty response" in inter.texts[0]


async def test_ask_discord_failure_propagates_to_global_handler():
    # Discord send failures bubble up to the tree's on_error, which logs them.
    inter = FakeInteraction(fail_send=True)
    with pytest.raises(discord.HTTPException):
        await handle_ask(inter, "hello", StubOllama(reply="Hi"))


async def test_ask_end_to_end_with_fake_ollama_server(fake_ollama):
    client = OllamaClient(base_url=fake_ollama["url"])
    inter = FakeInteraction()
    try:
        await handle_ask(inter, "hello", client)
    finally:
        await client.close()
    assert inter.texts == ["Hello from fake"]
