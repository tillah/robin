"""Application settings loaded from environment variables / .env."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(Exception):
    pass


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {value!r}")


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {value!r}")


@dataclass(frozen=True)
class Settings:
    discord_token: str
    # If set, slash commands are synced to this server only (instant updates).
    discord_guild_id: int | None
    # Channel for daily reports, alerts and tracked-opportunity updates.
    report_channel_id: int | None
    ollama_url: str
    ollama_model: str
    ollama_timeout: float
    # qwen3 "thinking" mode: slower, and the reasoning is stripped anyway.
    ollama_think: bool
    log_level: str
    log_dir: str
    db_path: str


def load_settings() -> Settings:
    load_dotenv()

    token = os.getenv("DISCORD_TOKEN", "").strip()
    if not token:
        raise ConfigError("DISCORD_TOKEN is not set. Add it to your .env file.")

    return Settings(
        discord_token=token,
        discord_guild_id=_get_int("DISCORD_GUILD_ID", None),
        report_channel_id=_get_int("DISCORD_REPORT_CHANNEL_ID", None),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        ollama_timeout=_get_float("OLLAMA_TIMEOUT", 300.0),
        ollama_think=_get_bool("OLLAMA_THINK", False),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_dir=os.getenv("LOG_DIR", "logs"),
        db_path=os.getenv("DB_PATH", "data/robin.db"),
    )
