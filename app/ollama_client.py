"""Minimal async client for a local Ollama server. No Discord code here."""

import asyncio
import logging
import re
from typing import Any

import aiohttp

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaError(Exception):
    """Base error for Ollama failures. str(e) is safe to show to users."""


class OllamaUnavailable(OllamaError):
    pass


class OllamaTimeout(OllamaError):
    pass


class OllamaResponseError(OllamaError):
    pass


def strip_thinking(text: str) -> str:
    """Remove qwen3 <think>...</think> blocks (and an unterminated leading one)."""
    text = _THINK_RE.sub("", text)
    if text.lstrip().lower().startswith("<think>"):
        # Unterminated reasoning block: nothing usable after it.
        return ""
    return text.strip()


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        timeout: float = 300.0,
        think: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.think = think
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def is_available(self) -> bool:
        """True if the server is reachable and the configured model is installed."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    return False
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return False
        names = {m.get("name") for m in data.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    async def chat(
        self,
        prompt: str,
        system: str | None = None,
        format: str | dict | None = None,
        timeout: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Send a single-turn chat and return the cleaned assistant text.

        `format` may be "json" or a JSON schema dict for structured output.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
        }
        if format is not None:
            payload["format"] = format
        if options:
            payload["options"] = options

        total = timeout or self.timeout
        session = await self._get_session()
        log.debug("Ollama request: model=%s prompt_chars=%d", self.model, len(prompt))
        try:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=total),
            ) as resp:
                if resp.status == 404:
                    raise OllamaResponseError(
                        f"Model '{self.model}' not found in Ollama. Run: ollama pull {self.model}"
                    )
                if resp.status != 200:
                    body = (await resp.text())[:300]
                    log.error("Ollama HTTP %s: %s", resp.status, body)
                    raise OllamaResponseError(f"Ollama returned HTTP {resp.status}.")
                data = await resp.json(content_type=None)
        except asyncio.TimeoutError:
            raise OllamaTimeout(f"The local model did not respond within {total:.0f}s.")
        except aiohttp.ClientConnectionError as e:
            log.warning("Ollama connection failed: %s", e)
            raise OllamaUnavailable(
                "Can't reach Ollama. Is it running? (start it with `ollama serve`)"
            )
        except aiohttp.ClientError as e:
            log.warning("Ollama client error: %s", e)
            raise OllamaUnavailable(f"Error talking to Ollama: {e.__class__.__name__}")
        except ValueError:
            raise OllamaResponseError("Ollama returned invalid JSON.")

        try:
            content = data["message"]["content"]
        except (KeyError, TypeError):
            raise OllamaResponseError("Unexpected response format from Ollama.")
        if not isinstance(content, str):
            raise OllamaResponseError("Unexpected response format from Ollama.")

        log.debug(
            "Ollama response: chars=%d eval_count=%s duration_ns=%s",
            len(content), data.get("eval_count"), data.get("total_duration"),
        )
        return strip_thinking(content)
