import pytest
from aiohttp import web

from app.ollama_client import (
    OllamaClient,
    OllamaResponseError,
    OllamaTimeout,
    OllamaUnavailable,
    strip_thinking,
)
from tests.conftest import slow_handler


async def test_chat_success_sends_model_and_think_flag(fake_ollama):
    client = OllamaClient(base_url=fake_ollama["url"], model="qwen3:8b")
    try:
        assert await client.chat("hello", system="sys") == "Hello from fake"
    finally:
        await client.close()
    req = fake_ollama["requests"][0]
    assert req["model"] == "qwen3:8b"
    assert req["stream"] is False
    assert req["think"] is False
    assert req["messages"][0] == {"role": "system", "content": "sys"}
    assert req["messages"][1] == {"role": "user", "content": "hello"}


async def test_chat_strips_think_tags(fake_ollama):
    async def handler(request):
        return web.json_response({"message": {"content": "<think>hmm</think>\n\nAnswer"}})
    fake_ollama["handler"] = handler
    client = OllamaClient(base_url=fake_ollama["url"])
    try:
        assert await client.chat("q") == "Answer"
    finally:
        await client.close()


async def test_ollama_unavailable(closed_port_url):
    client = OllamaClient(base_url=closed_port_url)
    try:
        with pytest.raises(OllamaUnavailable):
            await client.chat("hello")
        assert await client.is_available() is False
    finally:
        await client.close()


async def test_timeout(fake_ollama):
    fake_ollama["handler"] = slow_handler
    client = OllamaClient(base_url=fake_ollama["url"], timeout=0.5)
    try:
        with pytest.raises(OllamaTimeout):
            await client.chat("hello")
    finally:
        await client.close()


async def test_model_missing_404(fake_ollama):
    async def handler(request):
        return web.json_response({"error": "model not found"}, status=404)
    fake_ollama["handler"] = handler
    client = OllamaClient(base_url=fake_ollama["url"], model="nope")
    try:
        with pytest.raises(OllamaResponseError, match="ollama pull nope"):
            await client.chat("hello")
    finally:
        await client.close()


@pytest.mark.parametrize("body", ['{"unexpected": true}', "not json", '{"message": {"content": 5}}'])
async def test_malformed_response(fake_ollama, body):
    async def handler(request):
        return web.Response(text=body, content_type="application/json")
    fake_ollama["handler"] = handler
    client = OllamaClient(base_url=fake_ollama["url"])
    try:
        with pytest.raises(OllamaResponseError):
            await client.chat("hello")
    finally:
        await client.close()


async def test_server_error(fake_ollama):
    async def handler(request):
        return web.Response(status=500, text="boom")
    fake_ollama["handler"] = handler
    client = OllamaClient(base_url=fake_ollama["url"])
    try:
        with pytest.raises(OllamaResponseError, match="500"):
            await client.chat("hello")
    finally:
        await client.close()


async def test_is_available(fake_ollama):
    client = OllamaClient(base_url=fake_ollama["url"], model="qwen3:8b")
    try:
        assert await client.is_available() is True
        client.model = "other:1b"
        assert await client.is_available() is False
    finally:
        await client.close()


def test_strip_thinking_variants():
    assert strip_thinking("plain") == "plain"
    assert strip_thinking("<think>\na\nb\n</think>result") == "result"
    assert strip_thinking("<think>never closed") == ""
