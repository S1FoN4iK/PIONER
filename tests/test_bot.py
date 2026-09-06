"""Мелкие решения бота: нарезка текста, триггеры, обращения, склад промптов."""

from __future__ import annotations

import pytest
from conftest import PNG, bot_user, message

from ttbot.ai import Image
from ttbot.bot import (
    _TEXT_LIMIT,
    _Vault,
    _addressed,
    _has_image,
    _match_trigger,
    _split,
    _strip_mention,
)


# --- нарезка длинных ответов --------------------------------------------


def test_short_text_stays_one_chunk():
    assert _split("привет") == ["привет"]


def test_long_text_is_split_within_limit():
    text = "абзац\n" * 2000
    chunks = _split(text)
    assert len(chunks) > 1
    assert all(len(c) <= _TEXT_LIMIT for c in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_split_prefers_line_breaks():
    text = "a" * 4000 + "\n" + "b" * 200
    assert _split(text)[0] == "a" * 4000


def test_split_handles_text_without_separators():
    solid = "я" * 5000
    chunks = _split(solid)
    assert len(chunks) == 2 and "".join(chunks) == solid


# --- триггерные слова ----------------------------------------------------

WORDS = ("нарисуй картинку", "нарисуй", "ответь")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Нарисуй кота", "кота"),
        ("нарисуй кота", "кота"),
        ("Нарисуй картинку кота", "кота"), 
        ("Ответь, что такое Python", "что такое Python"),
        ("Ответь — почему небо синее", "почему небо синее"),
        ("Нарисуй", ""),    
        ("Ответьте пожалуйста", None),  
        ("просто текст", None),
        ("а нарисуй кота", None),        
    ],
)
def test_match_trigger(text, expected):
    assert _match_trigger(text, WORDS) == expected


def test_empty_trigger_list_matches_nothing():
    assert _match_trigger("Нарисуй кота", ()) is None


# --- обращения в группе --------------------------------------------------


def test_private_chat_is_always_addressed(bot):
    assert _addressed(message(bot, "привет"), "MyBot")


def test_group_ignores_plain_text(bot):
    assert not _addressed(message(bot, "ребята привет", chat_type="supergroup"), "MyBot")


def test_group_catches_mention(bot):
    assert _addressed(message(bot, "эй @MyBot", chat_type="supergroup"), "MyBot")
    assert _addressed(message(bot, "эй @mybot", chat_type="supergroup"), "MyBot")


def test_group_catches_reply_to_bot(bot):
    reply = message(bot, "мой ответ", chat_type="supergroup", from_bot=True)
    assert _addressed(message(bot, "а подробнее?", chat_type="supergroup", reply=reply), "MyBot")


def test_group_ignores_reply_to_human(bot):
    reply = message(bot, "чужое сообщение", chat_type="supergroup")
    assert not _addressed(message(bot, "ага", chat_type="supergroup", reply=reply), "MyBot")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("@MyBot привет", "привет"),
        ("эй @mybot!", "эй !"),
        ("@MyBot", ""),
        ("без упоминания", "без упоминания"),
    ],
)
def test_strip_mention(text, expected):
    assert _strip_mention(text, "MyBot") == expected


def test_strip_mention_without_username_is_noop():
    assert _strip_mention("@MyBot привет", None) == "@MyBot привет"


# --- есть ли картинка ----------------------------------------------------


def test_has_image(bot):
    from conftest import photo

    assert _has_image(message(bot, photo_sizes=photo()))
    assert not _has_image(message(bot, "просто текст"))
    assert not _has_image(None)


# --- склад промптов под кнопкой «ещё вариант» ---------------------------


def test_vault_round_trip():
    vault = _Vault()
    token = vault.put("рыжий кот", [])
    shot = vault.get(token)
    assert shot is not None and shot.prompt == "рыжий кот"


def test_vault_keeps_sources_for_repeat_edit():
    vault = _Vault()
    token = vault.put("добавь снег", [Image.of(PNG)])
    assert vault.get(token).sources[0].data == PNG


def test_vault_forgets_expired():
    vault = _Vault(ttl=0)
    assert vault.get(vault.put("кот", [])) is None


def test_vault_token_fits_callback_data():
    """У Telegram на callback_data всего 64 байта."""
    token = _Vault().put("к" * 500, [])
    assert len(f"again:{token}".encode()) <= 64


def test_vault_does_not_grow_forever():
    vault = _Vault(limit=5)
    tokens = [vault.put(f"промпт {i}", []) for i in range(20)]
    alive = [t for t in tokens if vault.get(t) is not None]
    assert len(alive) <= 5
    assert tokens[-1] in alive 
