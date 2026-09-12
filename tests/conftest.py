"""Общие заготовки для тестов: настройки без чтения .env, фейковый бот, картинка."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from aiogram import Bot
from aiogram.types import Chat, Document, Message, PhotoSize, User, Video, Voice

from ttbot.config import Settings

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40

CHAT_ID = -100123
BOT_TOKEN = "123456789:AABBCCDDEEFFgghhiijjkkllmmnnooppqqr"


def run(coro):
    """Мелкая обёртка вместо pytest-asyncio — лишняя зависимость тут не нужна."""
    return asyncio.run(coro)


def settings(**overrides: Any) -> Settings:
    """Настройки строго из аргументов: боевой .env в тесты попадать не должен."""
    base: dict[str, Any] = {"bot_token": "test"}
    base.update(overrides)
    return Settings(_env_file=None, **base)


def ai_settings(**overrides: Any) -> Settings:
    """Настройки с включённым AI — база для большинства тестов."""
    base: dict[str, Any] = {
        "ai_api_base": "https://gw.example",
        "ai_api_key": "k",
        "ai_chat_model": "chat-model",
        "ai_image_model": "image-model",
    }
    base.update(overrides)
    return settings(**base)


_DUMMY: httpx.AsyncClient | None = None


def dummy_client() -> httpx.AsyncClient:
    """Клиент-пустышка там, где запросов не будет.

    Создание обычного AsyncClient стоит ~0.25 с (SSL-контекст), а тестов много.
    MockTransport и быстрее, и гарантирует, что наружу ничего не уйдёт.
    """
    global _DUMMY
    if _DUMMY is None:
        _DUMMY = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(599, json={"error": "тест не ходит в сеть"})
            )
        )
    return _DUMMY


@pytest.fixture
def bot() -> Bot:
    """Бот, который никуда не ходит: транспорт подменён, профиль подставлен."""
    instance = Bot(token=BOT_TOKEN)
    instance._me = bot_user()
    return instance


def bot_user() -> User:
    return User(id=99, is_bot=True, first_name="Bot", username="MyBot")


def human() -> User:
    return User(id=7, is_bot=False, first_name="Даня", username="daniil")


def photo() -> list[PhotoSize]:
    return [PhotoSize(file_id="f", file_unique_id="u", width=100, height=100)]


def voice() -> Voice:
    return Voice(file_id="v", file_unique_id="vu", duration=3)


def video(size: int = 1024, mime: str | None = "video/mp4") -> Video:
    return Video(
        file_id="vid",
        file_unique_id="vidu",
        width=320,
        height=240,
        duration=5,
        mime_type=mime,
        file_size=size,
    )


def document(name: str = "note.txt", mime: str = "text/plain", size: int = 512) -> Document:
    return Document(
        file_id="doc", file_unique_id="docu", file_name=name, mime_type=mime, file_size=size
    )


def message(
    bot: Bot,
    text: str | None = None,
    *,
    caption: str | None = None,
    photo_sizes: list[PhotoSize] | None = None,
    voice_note: Voice | None = None,
    video_file: Video | None = None,
    doc: Document | None = None,
    chat_type: str = "private",
    reply: Message | None = None,
    from_bot: bool = False,
) -> Message:
    return Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=Chat(id=CHAT_ID, type=chat_type),
        from_user=bot_user() if from_bot else human(),
        text=text,
        caption=caption,
        photo=photo_sizes,
        voice=voice_note,
        video=video_file,
        document=doc,
        reply_to_message=reply,
    ).as_(bot)


@pytest.fixture
def no_feed_pause(monkeypatch):
    """Пауза между попытками нужна в бою, но не в тестах — они и так гоняются при старте."""
    monkeypatch.setattr("ttbot.media._FEED_RETRY_PAUSE", 0)
