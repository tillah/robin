from app.discord_utils import split_message, truncate


def test_short_message_single_chunk():
    assert split_message("hello") == ["hello"]


def test_empty_message():
    assert split_message("   ") == []


def test_long_message_respects_limit_and_keeps_content():
    paragraphs = [f"Paragraph {i} " + "word " * 80 for i in range(30)]
    text = "\n\n".join(paragraphs)
    chunks = split_message(text)
    assert len(chunks) > 1
    assert all(len(c) <= 2000 for c in chunks)
    # Every paragraph start survives the split.
    joined = "\n".join(chunks)
    for i in range(30):
        assert f"Paragraph {i} " in joined


def test_unbroken_text_hard_split():
    chunks = split_message("x" * 5000)
    assert all(len(c) <= 2000 for c in chunks)
    assert sum(len(c) for c in chunks) == 5000


def test_code_block_closed_and_reopened():
    code = "```python\n" + "\n".join(f"print({i})" for i in range(400)) + "\n```"
    chunks = split_message(code)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 2000
        assert c.count("```") % 2 == 0, "each chunk must have balanced fences"
    assert chunks[1].startswith("```python")


def test_truncate():
    assert truncate("abc", 5) == "abc"
    assert truncate("abcdefgh", 5) == "abcd…"
    assert len(truncate("a" * 100, 10)) == 10
