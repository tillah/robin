"""Decide whether a newly found opportunity is the same as one already stored."""

import json
import logging
import re
from difflib import SequenceMatcher

from app.ollama_client import OllamaClient, OllamaError

log = logging.getLogger(__name__)

# Generic or "template" words that models put in many titles ("AI-Driven X Platform for Y").
_STOPWORDS = {
    "a", "an", "and", "the", "of", "for", "in", "on", "to", "with", "by", "at", "from", "via",
    "or", "as", "is", "be", "its", "their", "new", "local", "based", "platform", "service",
    "services", "solution", "solutions", "opportunity", "opportunities", "business", "market",
    "sector", "africa", "african", "sme", "smes", "small", "medium", "enterprise", "enterprises",
    "driven", "powered", "enhanced", "enabled", "smart", "digital", "tech", "technology",
    "system", "systems", "tool", "tools", "app", "provider", "company", "companies",
    "zimbabwean", "zambian", "botswanan", "motswana", "south", "southern",
}


def keywords(text: str, ignore: set[str] = frozenset()) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for w in words:
        if w in _STOPWORDS or w in ignore or len(w) < 3:
            continue
        # crude stemming so "farmers"/"farmer", "drones"/"drone" match
        for suffix in ("ing", "es", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 4:
                w = w[: -len(suffix)]
                break
        out.add(w)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(new: dict, existing: dict) -> float:
    """0..1 similarity based on distinctive title keywords plus title+problem+solution keywords.

    Character-level title similarity is only trusted when near-identical: model titles are
    templated ("AI-Driven Finance Platform for Farmers" vs "AI-Driven Machinery Platform for
    Farmers"), so moderate character overlap says little.
    """
    t1, t2 = new["title"].lower().strip(), existing["title"].lower().strip()
    if t1 == t2 or SequenceMatcher(None, t1, t2).ratio() >= 0.92:
        return 1.0
    # Country and category words appear in both texts and would inflate overlap.
    context = " ".join(str(x.get(k, "")) for x in (new, existing) for k in ("country", "category"))
    ignore = keywords(context) | set(re.findall(r"[a-z]+", context.lower()))
    title_kw = _jaccard(keywords(t1, ignore), keywords(t2, ignore))
    body_kw = _jaccard(
        keywords(f"{new['title']} {new['problem']} {new['solution']}", ignore),
        keywords(f"{existing['title']} {existing['problem']} {existing['solution']}", ignore),
    )
    return 0.6 * title_kw + 0.4 * body_kw


MATCH_THRESHOLD = 0.5       # at or above: same opportunity without asking the model
CONFIRM_THRESHOLD = 0.12    # between this and MATCH_THRESHOLD: ask the model
MAX_CONFIRM_CANDIDATES = 5


def find_match(new: dict, candidates: list[dict], threshold: float = MATCH_THRESHOLD) -> dict | None:
    best, best_score = None, 0.0
    for c in candidates:
        s = similarity(new, c)
        if s > best_score:
            best, best_score = c, s
    return best if best_score >= threshold else None


CONFIRM_SYSTEM = (
    "You decide whether a newly found business opportunity is the SAME opportunity as one already "
    "on file (same problem for the same customers, even if worded differently). Related but distinct "
    'ideas are NOT the same. Reply with JSON: {"match_id": <id or 0>}'
)
CONFIRM_SCHEMA = {
    "type": "object",
    "properties": {"match_id": {"type": "integer"}},
    "required": ["match_id"],
}


async def find_match_with_model(
    new: dict, candidates: list[dict], ollama: OllamaClient | None
) -> dict | None:
    """Lexical match first; for borderline candidates ask the model. Any model failure = no match."""
    direct = find_match(new, candidates)
    if direct or ollama is None:
        return direct

    scored = sorted(((similarity(new, c), c) for c in candidates), key=lambda x: x[0], reverse=True)
    shortlist = [c for s, c in scored if s >= CONFIRM_THRESHOLD][:MAX_CONFIRM_CANDIDATES]
    if not shortlist:
        return None

    lines = [f"NEW: {new['title']}\nProblem: {new['problem']}\nSolution: {new['solution']}\n", "ON FILE:"]
    for c in shortlist:
        lines.append(f"[{c['id']}] {c['title']}\nProblem: {c['problem']}\nSolution: {c['solution']}\n")
    try:
        raw = await ollama.chat("\n".join(lines), system=CONFIRM_SYSTEM, format=CONFIRM_SCHEMA,
                                timeout=60, options={"temperature": 0})
        match_id = int(json.loads(raw).get("match_id", 0))
    except (OllamaError, ValueError, TypeError, AttributeError) as e:
        log.info("Match confirmation unavailable (%s); treating as new", e)
        return None
    # Only accept an id we actually offered.
    return next((c for c in shortlist if c["id"] == match_id), None)
