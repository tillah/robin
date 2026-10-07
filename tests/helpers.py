"""Shared fakes for tests."""

import json
from types import SimpleNamespace

import discord

from app.ollama_client import OllamaError
from app.research.matching import CONFIRM_SYSTEM


class FakeInteraction:
    def __init__(self, fail_send: bool = False):
        self.user = SimpleNamespace(id=123)
        self.deferred = False
        self.sent: list[dict] = []        # followup messages: {"content":..., "embed":...}
        self.edits: list[str] = []        # edits of the original (deferred) response
        self.fail_send = fail_send
        self.response = SimpleNamespace(defer=self._defer, send_message=self._send_message,
                                        is_done=lambda: self.deferred)
        self.followup = SimpleNamespace(send=self._send)

    async def _defer(self, thinking=False):
        self.deferred = True

    async def _send(self, content=None, embed=None, **kwargs):
        if self.fail_send:
            raise discord.HTTPException(SimpleNamespace(status=500, reason="err"), "discord down")
        self.sent.append({"content": content, "embed": embed})

    async def _send_message(self, content=None, embed=None, **kwargs):
        self.deferred = True
        self.sent.append({"content": content, "embed": embed, **kwargs})

    async def edit_original_response(self, content=None, **kwargs):
        if self.fail_send:
            raise discord.HTTPException(SimpleNamespace(status=500, reason="err"), "discord down")
        self.edits.append(content)

    @property
    def texts(self) -> list[str]:
        return [m["content"] for m in self.sent if m["content"]]


class StubOllama:
    """Scripted Ollama. `analysis` is a list of replies (str or Exception) for analysis calls;
    match-confirmation calls return `match_reply`."""

    model = "stub-model"

    def __init__(self, analysis=None, match_reply='{"match_id": 0}', reply="stub reply"):
        self.analysis = list(analysis or [])
        self.match_reply = match_reply
        self.reply = reply
        self.calls: list[dict] = []

    async def chat(self, prompt, system=None, format=None, timeout=None, options=None):
        self.calls.append({"prompt": prompt, "system": system, "format": format})
        if system == CONFIRM_SYSTEM:
            if isinstance(self.match_reply, Exception):
                raise self.match_reply
            return self.match_reply
        if format is None:
            return self.reply
        if not self.analysis:
            raise OllamaError("no scripted reply")
        item = self.analysis.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def analysis_calls(self) -> int:
        return sum(1 for c in self.calls if c["format"] is not None and c["system"] != CONFIRM_SYSTEM)


def opp_json(title="Cold storage for horticulture exporters", source_ids=(1,), **overrides) -> dict:
    data = {
        "title": title,
        "country": "zimbabwe",
        "category": "Agriculture",
        "problem": "Exporters lose produce because there is no cold chain near farms.",
        "solution": "Shared, pay-per-use cold rooms near major farming areas.",
        "target_customers": "Horticulture exporters",
        "evidence": "Source says exports grew 20% but spoilage is high.",
        "competitors": ["Big Co", "Other Co"],
        "why_now": "Export demand rising",
        "risks": "Power outages",
        "next_step": "Interview 10 exporters",
        "source_ids": list(source_ids),
        "scores": {"demand": 8, "competition": 4, "startup_difficulty": 6,
                   "revenue_potential": 7, "accessibility": 6},
    }
    data.update(overrides)
    return data


def analysis_reply(*opps) -> str:
    return json.dumps({"opportunities": list(opps)})
