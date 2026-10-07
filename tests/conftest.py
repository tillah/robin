import asyncio
import socket

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer


@pytest.fixture
async def fake_ollama():
    """A local HTTP server imitating Ollama. Set `state["handler"]` to customise /api/chat."""
    state = {"requests": []}

    async def default_chat(request):
        return web.json_response({"message": {"role": "assistant", "content": "Hello from fake"}})

    async def chat(request):
        state["requests"].append(await request.json())
        return await state.get("handler", default_chat)(request)

    async def tags(request):
        return web.json_response({"models": [{"name": "qwen3:8b"}]})

    app = web.Application()
    app.router.add_post("/api/chat", chat)
    app.router.add_get("/api/tags", tags)
    server = TestServer(app)
    await server.start_server()
    state["url"] = str(server.make_url("")).rstrip("/")
    yield state
    await server.close()


@pytest.fixture
def closed_port_url():
    """URL on a port with nothing listening."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


async def slow_handler(request):
    await asyncio.sleep(5)
    return web.json_response({"message": {"content": "too late"}})
