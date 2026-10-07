import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import truststore

# Verify HTTPS certificates against the macOS Keychain. The venv's Python
# (MacPorts) ships without a CA bundle, so HTTPS fails without this.
truststore.inject_into_ssl()

import discord  # noqa: E402

from app.bot import RobinBot
from app.config import ConfigError, load_settings
from app.research_config import load_research_config


def setup_logging(level: str, log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)-7s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "robin.log"), maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console)
    # discord.py is very chatty at DEBUG; keep it at INFO unless debugging it specifically.
    logging.getLogger("discord").setLevel(max(logging.INFO, root.level))
    # Third-party libraries that log every HTTP request / extraction miss.
    for noisy in ("primp", "ddgs", "trafilatura", "httpx"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def main() -> None:
    try:
        settings = load_settings()
        research_config = load_research_config()
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    setup_logging(settings.log_level, settings.log_dir)
    bot = RobinBot(settings, research_config)
    try:
        # log_handler=None: we've configured logging ourselves.
        bot.run(settings.discord_token, log_handler=None)
    except discord.LoginFailure:
        logging.getLogger(__name__).error("Discord rejected the token. Check DISCORD_TOKEN in .env.")
        sys.exit(1)
    except discord.PrivilegedIntentsRequired:
        logging.getLogger(__name__).error("Privileged intents not enabled in the Discord developer portal.")
        sys.exit(1)


if __name__ == "__main__":
    main()
