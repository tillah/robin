"""Local SQLite storage for opportunities, their sources, and research runs."""

import logging
import os
from datetime import datetime, timezone

import aiosqlite

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS opportunities (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    title               TEXT NOT NULL,
    country             TEXT NOT NULL,
    category            TEXT NOT NULL,
    problem             TEXT NOT NULL,
    solution            TEXT NOT NULL,
    target_customers    TEXT,
    evidence            TEXT,
    competitors         TEXT,
    why_now             TEXT,
    risks               TEXT,
    next_step           TEXT,
    demand              INTEGER,
    competition         INTEGER,
    startup_difficulty  INTEGER,
    revenue_potential   INTEGER,
    accessibility       INTEGER,
    score               REAL NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active',   -- active | tracked | archived
    seen_count          INTEGER NOT NULL DEFAULT 1,
    first_seen          TEXT NOT NULL,
    last_seen           TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id  INTEGER NOT NULL REFERENCES opportunities(id) ON DELETE CASCADE,
    title           TEXT NOT NULL,
    url             TEXT NOT NULL,
    source          TEXT,
    published_at    TEXT,
    retrieved_at    TEXT NOT NULL,
    run_id          INTEGER REFERENCES research_runs(id),
    UNIQUE (opportunity_id, url)
);

CREATE TABLE IF NOT EXISTS research_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    query               TEXT NOT NULL,
    kind                TEXT NOT NULL DEFAULT 'manual',   -- manual | daily
    started_at          TEXT NOT NULL,
    completed_at        TEXT,
    status              TEXT NOT NULL,                    -- running | completed | failed
    sources_found       INTEGER DEFAULT 0,
    opportunities_found INTEGER DEFAULT 0,
    error               TEXT
);

CREATE INDEX IF NOT EXISTS idx_opp_country ON opportunities(country);
CREATE INDEX IF NOT EXISTS idx_opp_score ON opportunities(score);
CREATE INDEX IF NOT EXISTS idx_sources_opp ON sources(opportunity_id);
"""

OPPORTUNITY_TEXT_FIELDS = (
    "title", "country", "category", "problem", "solution", "target_customers",
    "evidence", "competitors", "why_now", "risks", "next_step",
)
SCORE_FIELDS = ("demand", "competition", "startup_difficulty", "revenue_potential", "accessibility")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str):
        self.path = path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._conn.execute("PRAGMA journal_mode = WAL")
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected")
        return self._conn

    # --- research runs -------------------------------------------------

    async def start_run(self, query: str, kind: str = "manual") -> int:
        cur = await self.conn.execute(
            "INSERT INTO research_runs (query, kind, started_at, status) VALUES (?, ?, ?, 'running')",
            (query, kind, now_iso()),
        )
        await self.conn.commit()
        return cur.lastrowid

    async def finish_run(
        self, run_id: int, status: str, sources_found: int = 0,
        opportunities_found: int = 0, error: str | None = None,
    ) -> None:
        await self.conn.execute(
            "UPDATE research_runs SET completed_at=?, status=?, sources_found=?, "
            "opportunities_found=?, error=? WHERE id=?",
            (now_iso(), status, sources_found, opportunities_found, error, run_id),
        )
        await self.conn.commit()

    async def has_run_between(self, kind: str, start: datetime, end: datetime) -> bool:
        """True if a run of `kind` started in [start, end). Datetimes must be timezone-aware."""
        async with self.conn.execute(
            "SELECT 1 FROM research_runs WHERE kind=? AND started_at >= ? AND started_at < ? LIMIT 1",
            (kind, start.astimezone(timezone.utc).isoformat(timespec="seconds"),
             end.astimezone(timezone.utc).isoformat(timespec="seconds")),
        ) as cur:
            return await cur.fetchone() is not None

    async def get_run(self, run_id: int) -> dict | None:
        async with self.conn.execute("SELECT * FROM research_runs WHERE id=?", (run_id,)) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    # --- opportunities -------------------------------------------------

    async def insert_opportunity(self, fields: dict) -> int:
        ts = now_iso()
        cols = [*OPPORTUNITY_TEXT_FIELDS, *SCORE_FIELDS, "score"]
        values = [fields.get(c) for c in cols]
        cur = await self.conn.execute(
            f"INSERT INTO opportunities ({', '.join(cols)}, first_seen, last_seen, created_at, updated_at) "
            f"VALUES ({', '.join('?' * len(cols))}, ?, ?, ?, ?)",
            (*values, ts, ts, ts, ts),
        )
        await self.conn.commit()
        return cur.lastrowid

    async def update_opportunity_seen(self, opp_id: int, fields: dict, max_evidence_chars: int = 3000) -> None:
        """Record a re-sighting. Descriptive text stays stable; new evidence is appended
        (dated) and scores are replaced with the latest assessment."""
        existing = await self.get_opportunity(opp_id)
        if not existing:
            return
        ts = now_iso()
        evidence = existing.get("evidence") or ""
        new_evidence = (fields.get("evidence") or "").strip()
        if new_evidence and new_evidence != "Unknown" and new_evidence not in evidence:
            evidence = f"{evidence}\n\n[{ts[:10]}] {new_evidence}".strip()
            if len(evidence) > max_evidence_chars:
                evidence = "…" + evidence[-max_evidence_chars:]
        scores = [fields.get(c, existing[c]) for c in (*SCORE_FIELDS, "score")]
        await self.conn.execute(
            f"UPDATE opportunities SET {', '.join(f'{c}=?' for c in (*SCORE_FIELDS, 'score'))}, "
            "evidence=?, last_seen=?, updated_at=?, seen_count = seen_count + 1 WHERE id=?",
            (*scores, evidence, ts, ts, opp_id),
        )
        await self.conn.commit()

    async def get_opportunity(self, opp_id: int) -> dict | None:
        async with self.conn.execute("SELECT * FROM opportunities WHERE id=?", (opp_id,)) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def list_opportunities(
        self, country: str | None = None, category: str | None = None,
        min_score: float | None = None, status: str | None = None, limit: int = 20,
    ) -> list[dict]:
        where, params = ["status != 'archived'"], []
        if country:
            where.append("country = ? COLLATE NOCASE")
            params.append(country)
        if category:
            where.append("category = ? COLLATE NOCASE")
            params.append(category)
        if min_score is not None:
            where.append("score >= ?")
            params.append(min_score)
        if status:
            where.append("status = ?")
            params.append(status)
        sql = (f"SELECT * FROM opportunities WHERE {' AND '.join(where)} "
               "ORDER BY score DESC, last_seen DESC LIMIT ?")
        async with self.conn.execute(sql, (*params, limit)) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def candidates_for_match(self, country: str) -> list[dict]:
        """Opportunities that a new finding in `country` might duplicate."""
        async with self.conn.execute(
            "SELECT id, title, country, category, problem, solution, score FROM opportunities "
            "WHERE status != 'archived' AND (country = ? COLLATE NOCASE OR ? = 'Unknown')",
            (country, country),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def recent_titles(self, countries: list[str], limit: int = 40) -> list[str]:
        if not countries:
            return []
        marks = ", ".join("?" * len(countries))
        async with self.conn.execute(
            f"SELECT title FROM opportunities WHERE status != 'archived' AND country IN ({marks}) "
            "ORDER BY last_seen DESC LIMIT ?",
            (*countries, limit),
        ) as cur:
            return [r["title"] for r in await cur.fetchall()]

    async def set_status(self, opp_id: int, status: str) -> bool:
        cur = await self.conn.execute(
            "UPDATE opportunities SET status=?, updated_at=? WHERE id=?", (status, now_iso(), opp_id)
        )
        await self.conn.commit()
        return cur.rowcount > 0

    # --- sources -------------------------------------------------------

    async def add_sources(self, opp_id: int, sources: list[dict], run_id: int | None) -> int:
        """Attach sources; URLs already linked to this opportunity are skipped. Returns # added."""
        added = 0
        for s in sources:
            cur = await self.conn.execute(
                "INSERT OR IGNORE INTO sources (opportunity_id, title, url, source, published_at, "
                "retrieved_at, run_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (opp_id, s["title"], s["url"], s.get("source"), s.get("published_at"),
                 s.get("retrieved_at") or now_iso(), run_id),
            )
            added += cur.rowcount
        await self.conn.commit()
        return added

    async def get_sources(self, opp_id: int) -> list[dict]:
        async with self.conn.execute(
            "SELECT * FROM sources WHERE opportunity_id=? "
            "ORDER BY COALESCE(published_at, retrieved_at) DESC",
            (opp_id,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]
