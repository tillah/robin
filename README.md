# Robin — local AI business research agent

A Discord bot that runs entirely on your Mac, using a local Ollama model (`qwen3:8b`).

## Setup

1. Make sure Ollama is running and the model is installed:
   ```bash
   ollama pull qwen3:8b
   ```
2. Create the virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and set `DISCORD_TOKEN` and `DISCORD_REPORT_CHANNEL_ID`.
   `.env` is git-ignored — never commit it.
   - Optional: set `DISCORD_GUILD_ID` to your server ID so slash-command changes appear instantly.

## Run

```bash
.venv/bin/python main.py
```

Logs go to the console and `logs/robin.log`.

## Commands

| Command | What it does |
|---|---|
| `/ask <question>` | Answers using the local model |
| `/research <topic>` | Searches the web, analyses sources with the local model, stores and shows opportunities |
| `/opportunities [country] [category] [min_score] [tracked]` | Lists stored opportunities, best first |
| `/opportunity <id>` | Full details and supporting sources |
| `/track <id>` / `/untrack <id>` | Follow an opportunity; daily runs check it for meaningful news |
| `/daily` | Run the daily intelligence job now (posts to the report channel) |

## How research works

```
/research topic
  → SEARCH   DuckDuckGo (web + news) and Google News RSS — no API keys
  → COLLECT  filter noise (job boards, social media), dedupe, fetch article text (trafilatura)
  → ANALYSE  qwen3:8b returns structured JSON; every item is validated (pydantic) and must cite real sources
  → SCORE    overall = weighted average of demand, competition, difficulty, revenue, accessibility
  → MATCH    same opportunity seen before? update it (new evidence + sources appended) instead of duplicating
  → SQLite   data/robin.db
```

Ollama never touches the web: search and collection are separate modules (`app/research/search.py`,
`app/research/collect.py`), so the search provider can be swapped without touching analysis.

A research run takes roughly 1.5–3 minutes on an M-series Mac. Only one runs at a time.

## Daily intelligence

Every day at `[schedule].time` (local Mac time) the bot:

1. picks today's topics — a rotation through every country × category (`topics_per_day` per day), each paired with one of your signals, plus any `extra_topics`
2. researches each (search → collect → analyse → score → match against SQLite)
3. checks every tracked opportunity for new sources, and asks the model whether anything *meaningful* changed
4. posts to `DISCORD_REPORT_CHANNEL_ID`:
   - **📊 Daily intelligence** report — always one message, even on quiet days, with research coverage and any failures
   - **🔥 High-priority** alert per opportunity scoring ≥ `alert_min_score` (capped by `max_alerts_per_day`)
   - **🚨 Opportunity Update** per tracked opportunity with meaningful new information — nothing if nothing changed

Thresholds (`[alerts]` in `research.toml`):

| Score | Behaviour |
|---|---|
| below 5 | stored only |
| 5 – 7 | listed in the daily report |
| 7 + | high-priority alert |

Only *news* triggers anything: a brand-new opportunity, or a known one whose score newly crosses a threshold.
Re-seeing a known opportunity just adds its sources/evidence quietly.

The bot must be running and the Mac awake. If the bot starts after the scheduled time and today's job hasn't run,
it runs straight away (`catch_up = true`). To keep the Mac awake while the bot runs:

```bash
caffeinate -i .venv/bin/python main.py
```

## Configuring research

Edit `research.toml` and restart the bot:

- `[areas]` — countries, categories and signals to look for
- `[search]` — results per query, how far back news goes, blocked domains / title words
- `[analysis]` — max opportunities per run, your founder profile (affects accessibility/difficulty scores)
- `[scoring.weights]` — how much each sub-score counts towards the overall score
- `[schedule]` — daily time, catch-up, topics per day, extra topics, news recency
- `[alerts]` — report/alert thresholds and the daily alert cap

## Project layout

```
main.py                    entry point, logging
research.toml              research areas, schedule, thresholds (no secrets)
app/config.py              .env settings
app/ollama_client.py       local Ollama client (no Discord code)
app/bot.py                 Discord bot wiring
app/commands/              slash commands (/ask, /research, /opportunities, /opportunity, /track, /daily)
app/research/search.py     SEARCH providers (DuckDuckGo, Google News RSS)
app/research/collect.py    COLLECT: filter, dedupe, fetch article text
app/research/analyse.py    ANALYSE: prompt, JSON schema, validation, scoring
app/research/matching.py   is this the same opportunity as one already stored?
app/research/pipeline.py   one research run end-to-end
app/research/tracking.py   tracked-opportunity checks
app/daily.py               daily job, alert thresholds, Discord delivery, scheduler
app/db.py                  SQLite (data/robin.db)
```

## Troubleshooting

- **`CERTIFICATE_VERIFY_FAILED`** — run via `main.py` (it enables the macOS Keychain via `truststore`).
- **"Can't reach Ollama"** — start Ollama (`ollama serve` or the Ollama app) and check `ollama list` shows `qwen3:8b`.
- **No search results** — DuckDuckGo occasionally rate-limits; the run is recorded as failed and the next one usually works.
- **Daily report not appearing** — check `DISCORD_REPORT_CHANNEL_ID` and that the bot can view/send messages
  in that channel; errors are in `logs/robin.log`.
- **New slash commands not showing** — restart Discord, or set `DISCORD_GUILD_ID` for instant sync.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

## Notes

- HTTPS uses the macOS Keychain via `truststore`, because the MacPorts Python this venv is built on has no CA bundle.
- qwen3's "thinking" mode is off by default (`OLLAMA_THINK=false`) for faster replies.
