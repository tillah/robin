"""Discord embed builders for opportunities. Pure functions: no I/O."""

import discord

from app.discord_utils import truncate

EMBED_TOTAL_LIMIT = 6000


def score_colour(score: float) -> discord.Colour:
    if score >= 7:
        return discord.Colour.green()
    if score >= 5:
        return discord.Colour.gold()
    return discord.Colour.light_grey()


def score_breakdown(o: dict) -> str:
    return (f"Demand {o['demand']} · Competition {o['competition']} · Difficulty {o['startup_difficulty']}"
            f" · Revenue {o['revenue_potential']} · Access {o['accessibility']}")


def _link(title: str, url: str, limit: int = 80) -> str:
    title = truncate(title.replace("[", "(").replace("]", ")"), limit)
    return f"[{title}]({url})"


def source_lines(sources: list[dict], limit: int = 1024) -> str:
    lines, used = [], 0
    for s in sources:
        date = f" · {s['published_at'][:10]}" if s.get("published_at") else ""
        publisher = f" — {s['source']}" if s.get("source") else ""
        line = f"• {_link(s['title'], s['url'])}{publisher}{date}"
        if used + len(line) + 1 > limit:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) or "None recorded"


def opportunity_summary_embed(o: dict, sources: list[dict], footer: str = "") -> discord.Embed:
    """Compact card used in /research results and the daily report."""
    e = discord.Embed(
        title=truncate(f"#{o['id']} {o['title']}", 256),
        description=truncate(o["problem"], 400),
        colour=score_colour(o["score"]),
    )
    e.add_field(name=f"Score {o['score']}/10", value=score_breakdown(o), inline=False)
    e.add_field(name="Opportunity", value=truncate(o["solution"], 400), inline=False)
    e.add_field(name="Why now", value=truncate(o.get("why_now") or "Unknown", 300), inline=False)
    e.add_field(name="Sources", value=source_lines(sources[:3], 800), inline=False)
    e.set_footer(text=truncate(f"{o['country']} · {o['category']}" + (f" · {footer}" if footer else ""), 2048))
    return e


def opportunity_detail_embed(o: dict, sources: list[dict]) -> discord.Embed:
    e = discord.Embed(
        title=truncate(f"#{o['id']} {o['title']}", 256),
        description=f"**{o['country']} · {o['category']}** · status: {o['status']}",
        colour=score_colour(o["score"]),
    )
    e.add_field(name=f"Score {o['score']}/10", value=score_breakdown(o), inline=False)
    for name, key in (
        ("Problem", "problem"), ("Potential solution", "solution"),
        ("Target customers", "target_customers"), ("Evidence", "evidence"),
        ("Competitors", "competitors"), ("Why now", "why_now"), ("Risks", "risks"),
        ("Recommended next step", "next_step"),
    ):
        e.add_field(name=name, value=truncate(o.get(key) or "Unknown", 450), inline=False)
    e.add_field(name=f"Sources ({len(sources)})", value=source_lines(sources, 1024), inline=False)
    e.set_footer(text=f"First seen {o['first_seen'][:10]} · last seen {o['last_seen'][:10]} · "
                      f"seen {o['seen_count']}×")
    # Safety net for Discord's 6000-char total embed limit.
    while len(e) > EMBED_TOTAL_LIMIT and len(e.fields) > 1:
        e.remove_field(len(e.fields) - 2)
    return e


def opportunity_list_embed(rows: list[dict], title: str = "Opportunities") -> discord.Embed:
    e = discord.Embed(title=title, colour=discord.Colour.blurple())
    if not rows:
        e.description = "No opportunities found yet. Try `/research <topic>`."
        return e
    lines = []
    for o in rows:
        flag = " 📌" if o["status"] == "tracked" else ""
        lines.append(f"`#{o['id']}` **{o['score']}** · {truncate(o['title'], 90)}{flag}\n"
                     f"   ↳ {o['country']} · {o['category']}")
    e.description = truncate("\n".join(lines), 4000)
    e.set_footer(text="Use /opportunity <id> for details")
    return e


def high_priority_alert_embed(o: dict, sources: list[dict]) -> discord.Embed:
    e = discord.Embed(
        title=truncate(f"🔥 High-priority opportunity · #{o['id']} {o['title']}", 256),
        description=f"**{o['country']} · {o['category']}** — {truncate(o['problem'], 350)}",
        colour=discord.Colour.red(),
    )
    e.add_field(name=f"Score {o['score']}/10", value=score_breakdown(o), inline=False)
    e.add_field(name="Opportunity", value=truncate(o["solution"], 400), inline=False)
    e.add_field(name="Why it matters", value=truncate(o.get("why_now") or "Unknown", 400), inline=False)
    e.add_field(name="Evidence", value=truncate(o.get("evidence") or "Unknown", 600), inline=False)
    e.add_field(name="Sources", value=source_lines(sources[:4], 900), inline=False)
    e.add_field(name="Recommended next action", value=truncate(o.get("next_step") or "Unknown", 300), inline=False)
    e.set_footer(text="/track " + str(o["id"]) + " to follow this opportunity")
    return e


def tracked_update_embed(o: dict, what_changed: str, why_it_matters: str, sources: list[dict],
                         old_score: float, new_score: float) -> discord.Embed:
    delta = round(new_score - old_score, 1)
    score_text = (f"{old_score} → **{new_score}** ({'+' if delta > 0 else ''}{delta})"
                  if delta else f"{new_score} (unchanged)")
    e = discord.Embed(
        title=truncate(f"🚨 Opportunity Update · #{o['id']} {o['title']}", 256),
        colour=discord.Colour.orange(),
    )
    e.add_field(name="What changed", value=truncate(what_changed, 700), inline=False)
    e.add_field(name="Why it matters", value=truncate(why_it_matters or "Unknown", 500), inline=False)
    e.add_field(name="Score", value=score_text, inline=False)
    e.add_field(name="Source", value=source_lines(sources[:3], 800), inline=False)
    e.set_footer(text=f"{o['country']} · {o['category']} · /opportunity {o['id']} for full details")
    return e


def _opp_lines(rows: list[dict], limit: int = 1000) -> str:
    lines, used = [], 0
    for o in rows:
        line = f"`#{o['id']}` **{o['score']}** · {truncate(o['title'], 80)} ({o['country']})"
        if used + len(line) + 1 > limit:
            lines.append(f"…and {len(rows) - len(lines)} more")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) or "None"


def daily_report_embed(day: str, topics: list[tuple[str, str, str | None]], alerts: list[dict],
                       report_items: list[dict], updated: int, stored_only: int,
                       tracked_updates: int) -> discord.Embed:
    ok = sum(1 for _, status, _ in topics if status == "completed")
    e = discord.Embed(title=f"📊 Daily intelligence · {day}", colour=discord.Colour.blurple())
    if not alerts and not report_items:
        e.description = "No new opportunities worth your attention today."
    else:
        e.description = (f"**{len(alerts)}** high-priority · **{len(report_items)}** worth a look · "
                         f"{updated} known opportunities re-confirmed · {stored_only} low-score stored")
    if alerts:
        e.add_field(name="🔥 High priority (alerts sent)", value=_opp_lines(alerts), inline=False)
    if report_items:
        e.add_field(name="👀 Worth a look", value=_opp_lines(report_items), inline=False)
    if tracked_updates:
        e.add_field(name="📌 Tracked", value=f"{tracked_updates} tracked opportunity update(s) sent", inline=False)
    coverage = "\n".join(f"{'✅' if status == 'completed' else '⚠️'} {truncate(t, 60)}"
                         + (f" — {truncate(err, 80)}" if err else "") for t, status, err in topics)
    e.add_field(name=f"Research coverage ({ok}/{len(topics)} ok)", value=truncate(coverage or "None", 1024),
                inline=False)
    e.set_footer(text="/opportunity <id> for details · /track <id> to follow")
    return e
