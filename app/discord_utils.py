"""Helpers for fitting text into Discord's limits."""

DISCORD_MESSAGE_LIMIT = 2000


def split_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split text into chunks of at most `limit` chars.

    Prefers paragraph, then line, then word boundaries. If a chunk ends inside a
    ``` code block, the block is closed and reopened in the next chunk.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    reopen = ""  # fence opener (e.g. "```python") to prepend to the next chunk
    # Leave room for a reopened fence at the start and a closing fence at the end.
    fence_room = 20

    while text:
        text = reopen + text if reopen else text
        reopen = ""
        if len(text) <= limit:
            chunks.append(text)
            break

        budget = limit - fence_room
        window = text[:budget]
        cut = window.rfind("\n\n")
        if cut < budget // 2:
            cut = window.rfind("\n")
        if cut < budget // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = budget

        chunk = text[:cut].rstrip()
        text = text[cut:].lstrip("\n ")

        # Odd number of fences means we're inside a code block.
        fences = [line for line in chunk.split("\n") if line.strip().startswith("```")]
        if len(fences) % 2 == 1:
            chunk += "\n```"
            reopen = fences[-1].strip() + "\n"

        chunks.append(chunk)

    return chunks


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
