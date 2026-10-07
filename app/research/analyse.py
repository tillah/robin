"""ANALYSE: turn collected sources into validated, scored opportunities using Ollama."""

import json
import logging
from datetime import date
from difflib import SequenceMatcher

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.ollama_client import OllamaClient, OllamaError
from app.research.collect import CollectedSource
from app.research_config import SCORE_KEYS, ResearchConfig

log = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Analysis could not produce usable output. str(e) is safe to show users."""


class Scores(BaseModel):
    demand: int
    competition: int
    startup_difficulty: int
    revenue_potential: int
    accessibility: int

    @field_validator("*", mode="before")
    @classmethod
    def _to_int_1_10(cls, v):
        if isinstance(v, bool):
            raise ValueError("score must be a number")
        if isinstance(v, str):
            v = float(v.strip())  # ValueError -> validation error
        if not isinstance(v, (int, float)):
            raise ValueError("score must be a number")
        return max(1, min(10, round(v)))


class Opportunity(BaseModel):
    title: str = Field(min_length=3, max_length=150)
    country: str = "Unknown"
    category: str = "Unknown"
    problem: str = Field(min_length=10)
    solution: str = Field(min_length=10)
    target_customers: str = "Unknown"
    evidence: str = "Unknown"
    competitors: str = "Unknown"
    why_now: str = "Unknown"
    risks: str = "Unknown"
    next_step: str = "Unknown"
    source_ids: list[int] = Field(min_length=1)
    score_rationale: str = ""
    scores: Scores
    overall_score: float = 0.0

    @field_validator("title", "problem", "solution", mode="before")
    @classmethod
    def _required_text(cls, v):
        if not isinstance(v, str) or not v.strip() or v.strip().lower() == "unknown":
            raise ValueError("required text missing")
        return v.strip()

    @field_validator(
        "country", "category", "target_customers", "evidence", "competitors", "why_now",
        "risks", "next_step", "score_rationale", mode="before",
    )
    @classmethod
    def _text(cls, v):
        # Models often return lists for fields like competitors/risks.
        if isinstance(v, list):
            v = "; ".join(str(x).strip() for x in v if str(x).strip())
        if v is None:
            return "Unknown"
        if not isinstance(v, str):
            raise ValueError("expected text")
        return v.strip() or "Unknown"

    @field_validator("source_ids", mode="before")
    @classmethod
    def _ids(cls, v):
        if not isinstance(v, list):
            v = [v]
        out = []
        for x in v:
            try:
                out.append(int(str(x).strip().strip("[]")))
            except ValueError:
                continue
        return out


# Schema given to Ollama's structured output. Validation above is the real gatekeeper.
_TEXT = {"type": "string"}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "opportunities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **{k: _TEXT for k in (
                        "title", "country", "category", "problem", "solution", "target_customers",
                        "evidence", "competitors", "why_now", "risks", "next_step")},
                    "source_ids": {"type": "array", "items": {"type": "integer"}},
                    "score_rationale": _TEXT,
                    "scores": {
                        "type": "object",
                        "properties": {k: {"type": "integer"} for k in SCORE_KEYS},
                        "required": list(SCORE_KEYS),
                    },
                },
                "required": [
                    "title", "country", "category", "problem", "solution", "target_customers",
                    "evidence", "competitors", "why_now", "risks", "next_step", "source_ids",
                    "score_rationale", "scores",
                ],
            },
        }
    },
    "required": ["opportunities"],
}

SYSTEM_PROMPT = """You are a business opportunity analyst focused on Southern Africa.
You read numbered news/web sources and identify concrete, actionable business opportunities.

Rules:
- Base every opportunity on the provided sources. Cite them in source_ids using their [numbers].
  Cite only sources that specifically support that opportunity, not generic background.
- Do NOT invent facts, figures, companies or tenders that are not in the sources. If unsure, say "Unknown".
- Prefer specific opportunities (a product/service for a defined customer) over vague themes.
  Each opportunity must address a clearly different problem; do not return variations of one idea.
- Opportunities do NOT need to involve AI or software. Only propose technology where the sources justify it.
- "country" is ONE country from the list, or "Regional" if the opportunity spans several.
- "category" is the market sector the customers are in (e.g. agriculture), not the technology used.
- Write the solution as the product/service itself; do not mention "the founder".
- Skip sources that are irrelevant (job ads, unrelated news). It is fine to return fewer opportunities, or none.
- "evidence" must summarise what the cited sources actually say, with concrete facts/figures from them.
- Score each opportunity on its own merits; scores should differ between opportunities.
- First write score_rationale (one or two sentences citing the facts behind the scores), then the scores.
- Scores are integers 1-10. Be strict and calibrated; most real opportunities score 4-6 on most dimensions.
  A score of 8 or more must be justified by specific facts in the sources, not general optimism.
  demand: 9-10 = sources show paying customers, a tender, a budget or quantified unmet demand;
          6-7 = a clear problem is described but no evidence anyone will pay; 3-5 = plausible but speculative.
  competition: 10 = very crowded / dominated by big players (bad); 1-3 = sources show almost no providers.
               If unknown, assume 5-6, not low.
  startup_difficulty: 10 = needs large capital, licences, hardware or infrastructure (bad);
                      1-3 = can start with a laptop, little money and no licence.
  revenue_potential: 9-10 = large and recurring revenue evidenced; 4-6 = modest or unclear.
  accessibility: for the founder described below. 8+ only if they could reach first paying customers
                 within ~3 months with their skills; 1-3 if it needs government/telco/bank-level access.
Respond with JSON only, matching the required schema."""


def build_prompt(
    topic: str,
    sources: list[CollectedSource],
    cfg: ResearchConfig,
    existing_titles: list[str],
) -> str:
    lines = [
        f"Today's date: {date.today().isoformat()}",
        f"Research topic: {topic}",
        f"Founder profile (use ONLY for accessibility/startup_difficulty scores): {cfg.analysis.founder_profile}",
        f"Countries of interest: {', '.join(cfg.countries)}",
        f"Categories: use one of these exactly if it genuinely fits, otherwise a short sector name: {', '.join(cfg.categories)}",
        f"Signals to look for: {', '.join(cfg.signals)}",
        f"Return at most {cfg.analysis.max_opportunities} opportunities, best first.",
    ]
    if existing_titles:
        lines.append(
            "\nAlready-known opportunities (for reference only). Prefer finding NEW opportunities. "
            "Only include a known one if these sources add specific new evidence about it, "
            "and then reuse its EXACT title:"
        )
        lines.extend(f"- {t}" for t in existing_titles[:40])

    lines.append("\nSOURCES:")
    for i, s in enumerate(sources, start=1):
        meta = " | ".join(x for x in (s.source, s.published_at[:10] if s.published_at else "") if x)
        body = s.text or "(headline only)"
        lines.append(f"\n[{i}] {s.title}\n({meta})\n{body}")
    return "\n".join(lines)


def overall_score(scores: Scores, weights: dict[str, float]) -> float:
    total = 0.0
    for key in SCORE_KEYS:
        value = getattr(scores, key)
        if key in ("competition", "startup_difficulty"):
            value = 11 - value  # lower competition/difficulty is better
        total += weights[key] * value
    return round(total / sum(weights.values()), 1)


def _match_name(value: str, allowed: tuple[str, ...]) -> str:
    for name in allowed:
        if value.lower() == name.lower():
            return name
    return value


def normalise_country(value: str, allowed: tuple[str, ...]) -> str:
    """One configured country, 'Regional' when several are named, else the model's value."""
    mentioned = [c for c in allowed if c.lower() in value.lower()]
    if len(mentioned) > 1 or value.strip().lower() in ("regional", "southern africa", "africa", "multiple"):
        return "Regional"
    if len(mentioned) == 1 and len(value) <= len(mentioned[0]) + 2:
        return mentioned[0]
    return _match_name(value, allowed)


def parse_analysis(raw: str, num_sources: int, cfg: ResearchConfig) -> list[Opportunity]:
    """Validate model output. Raises AnalysisError if the overall shape is unusable;
    individual bad opportunities are dropped and logged."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise AnalysisError("The model returned invalid JSON.")
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("opportunities"), list):
        items = data["opportunities"]
    else:
        raise AnalysisError("The model's JSON did not contain an opportunities list.")

    valid: list[Opportunity] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            opp = Opportunity.model_validate(item)
        except ValidationError as e:
            log.info("Dropped malformed opportunity %r: %s", item.get("title"), e.errors()[:3])
            continue
        opp.source_ids = sorted({i for i in opp.source_ids if 1 <= i <= num_sources})
        if not opp.source_ids:
            log.info("Dropped opportunity without valid sources: %r", opp.title)
            continue
        # Only collapse near-identical titles; templated titles ("X for smallholders") differ by one word.
        if any(SequenceMatcher(None, opp.title.lower(), v.title.lower()).ratio() >= 0.95 for v in valid):
            continue
        opp.country = normalise_country(opp.country, cfg.countries)
        opp.category = _match_name(opp.category, cfg.categories)
        opp.overall_score = overall_score(opp.scores, cfg.weights)
        valid.append(opp)
    return valid[: cfg.analysis.max_opportunities]


async def analyse_sources(
    ollama: OllamaClient,
    topic: str,
    sources: list[CollectedSource],
    cfg: ResearchConfig,
    existing_titles: list[str] | None = None,
) -> list[Opportunity]:
    if not sources:
        return []
    prompt = build_prompt(topic, sources, cfg, existing_titles or [])
    options = {"num_ctx": cfg.analysis.num_ctx, "temperature": 0.2}

    last_error: AnalysisError | None = None
    for attempt in (1, 2):
        try:
            raw = await ollama.chat(
                prompt, system=SYSTEM_PROMPT, format=RESPONSE_SCHEMA,
                timeout=cfg.analysis.timeout, options=options,
            )
        except OllamaError as e:
            raise AnalysisError(str(e))
        try:
            return parse_analysis(raw, len(sources), cfg)
        except AnalysisError as e:
            log.warning("Analysis attempt %d unusable: %s (raw starts %r)", attempt, e, raw[:200])
            last_error = e
    raise last_error
