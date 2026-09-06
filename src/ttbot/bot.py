from __future__ import annotations

import asyncio
import io
import logging
import re
import secrets
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from aiogram import F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    InputMediaVideo,
    Message,
)

from .ai import AiClient, AiError, ChatMemory, ChatMode, Image, Overrides
from .config import Settings
from .media import (
    Downloader,
    DownloadResult,
    Media,
    find_media_url,
    looks_like_profile_url,
)
from .state import State

logger = logging.getLogger(__name__)

_TEXT_LIMIT = 4096 
_CAPTION_LIMIT = 1024  
_PHOTO_LIMIT_MB = 10 
_TTS_LIMIT = 4000 
_AGAIN = "again"  

_PREF_KEYS = {
    "chat_model": "модель для текста",
    "image_model": "модель для картинок",
    "prompt": "системный промпт",
    "size": "размер картинки",
}


@dataclass
class _Shot:

    prompt: str
    sources: list[Image] = field(default_factory=list)
    born: float = 0.0


class _Vault:

    def __init__(self, limit: int = 200, ttl: int = 3600) -> None:
        self._limit = limit
        self._ttl = ttl
        self._shots: dict[str, _Shot] = {}

    def put(self, prompt: str, sources: list[Image]) -> str:
        self._evict()
        token = secrets.token_urlsafe(8)
        self._shots[token] = _Shot(prompt, sources, time.monotonic())
        return token

    def get(self, token: str) -> _Shot | None:
        shot = self._shots.get(token)
        if shot is None:
            return None
        if time.monotonic() - shot.born > self._ttl:
            del self._shots[token]
            return None
        return shot

    def _evict(self) -> None:
        now = time.monotonic()
        for token in [t for t, s in self._shots.items() if now - s.born > self._ttl]:
            del self._shots[token]
        while len(self._shots) >= self._limit:
            self._shots.pop(next(iter(self._shots)))


class VideoSender:
    def __init__(self, bot, settings: Settings, downloader: Downloader) -> None:
        self._bot = bot
        self._settings = settings
        self._downloader = downloader

    @property
    def _size_limit(self) -> int:
        return self._settings.max_file_size_mb

    def _caption(self, media: Media) -> str | None:
        if not self._settings.send_caption:
            return None
        parts: list[str] = []
        if media.author:
            parts.append(f"@{media.author}")
        if media.page_url:
            parts.append(media.page_url)
        return "\n".join(parts) or None

    @staticmethod
    def _size_mb(path: Path) -> float:
        return path.stat().st_size / (1024 * 1024)

    async def send_from_url(
        self, chat_id: int, url: str, reply_to: int | None = None
    ) -> bool:
        with tempfile.TemporaryDirectory(prefix="dl_") as tmp:
            result = await self._downloader.download(url, Path(tmp))
            return await self._send_result(chat_id, result, reply_to)

    async def _send_result(
        self, chat_id: int, result: DownloadResult, reply_to: int | None
    ) -> bool:
        if result.is_single_video:
            return await self._send_single_video(chat_id, result, reply_to)
        sent = await self._send_album(chat_id, result, reply_to)
        await self._send_audio(chat_id, result)
        return sent

    async def _send_single_video(
        self, chat_id: int, result: DownloadResult, reply_to: int | None
    ) -> bool:
        media = result.media
        path = result.videos[0]
        size_mb = self._size_mb(path)
        if size_mb > self._size_limit:
            link = media.direct_url or media.page_url
            await self._bot.send_message(
                chat_id,
                f"Файл весит {size_mb:.0f} МБ — больше лимита Bot API "
                f"({self._size_limit} МБ).\nЗабрать оригинал: {link}",
                reply_to_message_id=reply_to,
            )
            return False
        await self._bot.send_video(
            chat_id,
            video=FSInputFile(path),
            caption=self._caption(media),
            width=media.width,
            height=media.height,
            duration=int(media.duration) if media.duration else None,
            supports_streaming=True,
            reply_to_message_id=reply_to,
        )
        return True

    def _build_album(
        self, result: DownloadResult, caption: str | None
    ) -> tuple[list, int]:

        entries: list[tuple[Path, bool]] = [] 
        skipped = 0
        for path in result.videos:
            if self._size_mb(path) > self._size_limit:
                skipped += 1
                continue
            entries.append((path, True))
        for path in result.images:
            entries.append((path, False))

        items: list = []
        for i, (path, is_video) in enumerate(entries):
            cap = caption if i == 0 else None
            if is_video:
                items.append(
                    InputMediaVideo(
                        media=FSInputFile(path), caption=cap, supports_streaming=True
                    )
                )
            else:
                items.append(InputMediaPhoto(media=FSInputFile(path), caption=cap))
        return items, skipped

    async def _send_album(
        self, chat_id: int, result: DownloadResult, reply_to: int | None
    ) -> bool:
        caption = self._caption(result.media)
        items, skipped = self._build_album(result, caption)

        sent = False
        first = True
        for i in range(0, len(items), 10):
            batch = items[i : i + 10]
            rt = reply_to if first else None
            if len(batch) == 1:
                await self._send_one(chat_id, batch[0], rt)
            else:
                await self._bot.send_media_group(chat_id, media=batch, reply_to_message_id=rt)
            sent = True
            first = False
            if i + 10 < len(items):
                await asyncio.sleep(1)

        if skipped:
            await self._bot.send_message(
                chat_id,
                f"{skipped} файл(ов) превысили лимит {self._size_limit} МБ "
                f"и не отправлены. Оригинал: {result.media.page_url}",
                reply_to_message_id=reply_to if not sent else None,
            )
            sent = True
        return sent

    async def _send_one(self, chat_id: int, item, reply_to: int | None) -> None:
        if isinstance(item, InputMediaVideo):
            await self._bot.send_video(
                chat_id,
                video=item.media,
                caption=item.caption,
                supports_streaming=True,
                reply_to_message_id=reply_to,
            )
        else:
            await self._bot.send_photo(
                chat_id,
                photo=item.media,
                caption=item.caption,
                reply_to_message_id=reply_to,
            )

    async def _send_audio(self, chat_id: int, result: DownloadResult) -> None:
        if result.audio is None:
            return
        media = result.media
        try:
            await self._bot.send_audio(
                chat_id,
                audio=FSInputFile(result.audio),
                title=(media.description[:60] if media.description else None),
                performer=(media.author or None),
            )
        except Exception as exc:
            logger.warning("failed to send audio for %s: %s", media.page_url, exc)


class AiResponder:
    """Мост между Telegram и моделью: текст, генерация и правка картинок."""

    def __init__(
        self,
        bot,
        settings: Settings,
        ai: AiClient,
        state: State | None = None,
        downloader: Downloader | None = None,
    ) -> None:
        self._bot = bot
        self._settings = settings
        self._ai = ai
        self._state = state
        self._downloader = downloader
        self._memory = ChatMemory(
            settings.ai_history_turns, settings.ai_history_ttl_minutes * 60
        )
        self._vault = _Vault()

    @property
    def enabled(self) -> bool:
        return self._ai.enabled

    @property
    def can_chat(self) -> bool:
        return self._ai.can_chat

    @property
    def can_draw(self) -> bool:
        return self._ai.can_draw

    @property
    def can_edit(self) -> bool:
        return self._ai.can_edit

    @property
    def can_see(self) -> bool:
        return self._ai.can_see

    @property
    def can_hear(self) -> bool:
        return self._ai.can_hear

    @property
    def can_speak(self) -> bool:
        return self._ai.can_speak

    @property
    def can_sum(self) -> bool:
        return self._ai.can_hear and self._ai.can_chat and self._downloader is not None

    @property
    def can_tune(self) -> bool:
        return self._settings.ai_chat_settings and self._state is not None

    @staticmethod
    def _key(message: Message) -> tuple[int, int]:
        return message.chat.id, message.from_user.id if message.from_user else 0

    def forget(self, message: Message) -> bool:
        return self._memory.forget(self._key(message))

    async def _over(self, chat_id: int) -> Overrides:
        """Переопределения этого чата из /set, если они вообще разрешены."""
        if not self.can_tune:
            return Overrides()
        raw = await self._state.prefs(chat_id)
        return Overrides(
            chat_model=raw.get("chat_model", ""),
            image_model=raw.get("image_model", ""),
            system_prompt=raw.get("prompt", ""),
            image_size=raw.get("size", ""),
        )

    async def ask(self, message: Message, prompt: str) -> None:
        """Вопрос к модели: с картинкой, если она рядом, иначе обычный текст."""
        images = await self.images_of(message) if self.can_see else []
        await self.answer(message, prompt, images or None)

    async def answer(
        self, message: Message, prompt: str, images: list[Image] | None = None
    ) -> None:
        status = await message.reply("👀 Смотрю…" if images else "💭 Думаю…")
        await self._answer_into(message, status, prompt, images, voice=False)

    async def _answer_into(
        self,
        message: Message,
        status: Message,
        prompt: str,
        images: list[Image] | None,
        voice: bool,
    ) -> None:
        key = self._key(message)
        try:
            over = await self._over(message.chat.id)
            history = None if images else self._memory.history(key)
            text = await self._ai.chat(prompt, history, over, images)
        except Exception as exc:
            await _edit(status, _fail_text(exc))
            return
        if not images:
            self._memory.remember(key, prompt, text)
        await self._deliver(message, status, text, voice)

    async def _deliver(
        self, message: Message, status: Message, text: str, voice: bool
    ) -> None:
        if voice and self._ai.can_speak:
            try:
                audio = await self._ai.speak(text[:_TTS_LIMIT])
            except AiError as exc:
                logger.warning("не озвучили ответ, шлём текстом: %s", exc)
            else:
                await _delete(status)
                await message.reply_voice(BufferedInputFile(audio, "voice.ogg"))
                return
        chunks = _split(text)
        if not await _edit(status, chunks[0]):
            await message.reply(chunks[0])
        for chunk in chunks[1:]:
            await message.answer(chunk)

    async def draw(self, message: Message, prompt: str) -> None:
        status = await message.reply("🎨 Рисую…")
        try:
            over = await self._over(message.chat.id)
            images = await self._ai.draw(prompt, over)
            await self._send_images(message, images, prompt, [])
        except Exception as exc:
            await _edit(status, _fail_text(exc))
            return
        await _delete(status)

    async def redraw(self, message: Message, prompt: str) -> None:
        status = await message.reply("🖌 Правлю…")
        try:
            sources = await self.images_of(message)
            if not sources:
                await _edit(
                    status,
                    "Не вижу картинку. Пришлите фото с подписью «/edit что поменять» "
                    "или ответьте так на уже отправленное фото.",
                )
                return
            over = await self._over(message.chat.id)
            images = await self._ai.redraw(prompt, sources, over)
            await self._send_images(message, images, prompt, sources)
        except Exception as exc:
            await _edit(status, _fail_text(exc))
            return
        await _delete(status)

    async def again(self, callback: CallbackQuery, token: str) -> None:
        """Кнопка «ещё вариант»: повторяем тот же запрос к модели."""
        shot = self._vault.get(token)
        message = callback.message
        if shot is None or message is None:
            await callback.answer(
                "Запрос уже забыт — отправьте его заново.", show_alert=True
            )
            return
        await callback.answer("Делаю ещё вариант…")
        status = await message.reply("🎨 Рисую…")
        try:
            over = await self._over(message.chat.id)
            if shot.sources:
                images = await self._ai.redraw(shot.prompt, shot.sources, over)
            else:
                images = await self._ai.draw(shot.prompt, over)
            await self._send_images(message, images, shot.prompt, shot.sources)
        except Exception as exc:
            await _edit(status, _fail_text(exc))
            return
        await _delete(status)

    async def images_of(self, message: Message) -> list[Image]:
        """Берём фото и из ответа-на-сообщение, и из самого сообщения."""
        images: list[Image] = []
        for source in (message.reply_to_message, message):
            image = await self._download_photo(source)
            if image is not None:
                images.append(image)
        return images

    async def _download_photo(self, message: Message | None) -> Image | None:
        if not _has_image(message):
            return None
        target = message.photo[-1] if message.photo else message.document
        buf = io.BytesIO()
        await self._bot.download(target, destination=buf)
        data = buf.getvalue()
        return Image.of(data) if data else None

    async def listen(self, message: Message) -> None:
        """Голосовое: расшифровываем, отвечаем — текстом или голосом."""
        status = await message.reply("🎤 Слушаю…")
        try:
            audio, filename = await self._download_audio(message)
            heard = await self._ai.transcribe(audio, filename)
        except Exception as exc:
            await _edit(status, _fail_text(exc))
            return
        await _edit(status, f"🎤 «{heard}»\n💭 Думаю…")
        await self._answer_into(
            message, status, heard, None, voice=self._settings.ai_voice_reply
        )

    async def say(self, message: Message, text: str) -> None:
        status = await message.reply("🔊 Озвучиваю…")
        try:
            audio = await self._ai.speak(text[:_TTS_LIMIT])
        except Exception as exc:
            await _edit(status, _fail_text(exc))
            return
        await _delete(status)
        await message.reply_voice(BufferedInputFile(audio, "voice.ogg"))

    async def summarize(self, message: Message, url: str) -> None:
        """Скачиваем ролик, расшифровываем дорожку и пересказываем."""
        status = await message.reply("⬇️ Забираю ролик…")
        try:
            with tempfile.TemporaryDirectory(prefix="sum_") as tmp:
                result = await self._downloader.download(url, Path(tmp))
                path = result.audio or (result.videos[0] if result.videos else None)
                if path is None:
                    await _edit(status, "В этом посте нет дорожки, пересказывать нечего.")
                    return
                size_mb = path.stat().st_size / (1024 * 1024)
                limit = self._settings.ai_transcribe_max_mb
                if size_mb > limit:
                    await _edit(
                        status,
                        f"Дорожка весит {size_mb:.0f} МБ — больше лимита расшифровки "
                        f"({limit} МБ). Поднимите AI_TRANSCRIBE_MAX_MB, если шлюз позволяет.",
                    )
                    return
                await _edit(status, "🎧 Расшифровываю…")
                heard = await self._ai.transcribe(path.read_bytes(), path.name)

            await _edit(status, "💭 Пересказываю…")
            over = await self._over(message.chat.id)
            summary = await self._ai.chat(
                f"{self._settings.ai_summary_prompt}\n\n{heard}", None, over
            )
        except Exception as exc:
            await _edit(status, _fail_text(exc))
            return
        await self._deliver(message, status, summary, voice=False)

    async def show_prefs(self, message: Message) -> None:
        raw = await self._state.prefs(message.chat.id)
        lines = ["Настройки этого чата:"]
        for key, label in _PREF_KEYS.items():
            value = raw.get(key) or "— из .env"
            lines.append(f"• {key} ({label}): {value}")
        lines.append("")
        lines.append("Поменять: /set prompt Ты отвечаешь как пират")
        lines.append("Сбросить одну: /set prompt -   Сбросить все: /set reset")
        await message.reply("\n".join(lines))

    async def set_pref(self, message: Message, args: str) -> None:
        if args.strip().lower() in ("reset", "сброс"):
            dropped = await self._state.clear_prefs(message.chat.id)
            await message.reply(
                f"Сброшено настроек: {dropped}." if dropped else "Настроек и так нет."
            )
            return

        key, _, value = args.strip().partition(" ")
        key = key.lower()
        if key not in _PREF_KEYS:
            await message.reply(
                "Не знаю такого ключа. Доступны: " + ", ".join(_PREF_KEYS) + "."
            )
            return
        value = value.strip()
        if value == "-":
            value = ""
        await self._state.set_pref(message.chat.id, key, value)
        await message.reply(
            f"{key}: {value}" if value else f"{key} сброшен — снова берётся из .env."
        )

    async def _download_audio(self, message: Message) -> tuple[bytes, str]:
        source = message.voice or message.audio or message.video_note
        if source is None and message.document:
            source = message.document
        if source is None:
            raise AiError("не нашёл аудио в сообщении")

        limit = self._settings.ai_transcribe_max_mb
        size_mb = (source.file_size or 0) / (1024 * 1024)
        if size_mb > limit:
            raise AiError(f"аудио весит {size_mb:.0f} МБ — больше лимита {limit} МБ")

        buf = io.BytesIO()
        await self._bot.download(source, destination=buf)
        data = buf.getvalue()
        if not data:
            raise AiError("не получилось скачать аудио из Telegram")
        name = getattr(source, "file_name", None) or (
            "voice.ogg" if message.voice or message.video_note else "audio.mp3"
        )
        return data, name

    def _keyboard(self, prompt: str, sources: list[Image]) -> InlineKeyboardMarkup | None:
        if not self._settings.ai_regen_button:
            return None
        token = self._vault.put(prompt, sources)
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🔁 Ещё вариант", callback_data=f"{_AGAIN}:{token}")]
            ]
        )

    async def _send_images(
        self, message: Message, images: list[Image], prompt: str, sources: list[Image]
    ) -> None:
        caption = prompt[:_CAPTION_LIMIT] if self._settings.send_caption else None
        photos = [img for img in images if img.size_mb <= _PHOTO_LIMIT_MB][:10]
        heavy = [img for img in images if img.size_mb > _PHOTO_LIMIT_MB]

        if len(photos) == 1:
            await message.reply_photo(
                _as_file(photos[0]),
                caption=caption,
                reply_markup=self._keyboard(prompt, sources),
            )
            caption = None
        elif photos:
            group = [
                InputMediaPhoto(media=_as_file(img), caption=caption if i == 0 else None)
                for i, img in enumerate(photos)
            ]
            await self._bot.send_media_group(
                message.chat.id, media=group, reply_to_message_id=message.message_id
            )
            caption = None

        for img in heavy:
            await message.reply_document(_as_file(img), caption=caption)
            caption = None


def _as_file(image: Image) -> BufferedInputFile:
    return BufferedInputFile(image.data, filename=image.filename)


def _has_image(message: Message | None) -> bool:
    if message is None:
        return False
    document = message.document
    return bool(
        message.photo
        or (document and (document.mime_type or "").startswith("image/"))
    )


def _replied_text(message: Message) -> str:
    reply = message.reply_to_message
    if reply is None:
        return ""
    return (reply.text or reply.caption or "").strip()


def _replied_url(message: Message) -> str | None:
    return find_media_url(_replied_text(message))


def _match_trigger(text: str, words: tuple[str, ...]) -> str | None:
    lowered = text.lower()
    for word in words:
        if not lowered.startswith(word):
            continue
        rest = text[len(word) :]
        if rest and (rest[0].isalnum() or rest[0] == "_"):
            continue
        return rest.lstrip(" \t\n,.:;!?—–-")
    return None


async def _edit(message: Message, text: str) -> bool:
    try:
        await message.edit_text(text)
        return True
    except Exception:
        return False


async def _delete(message: Message) -> None:
    try:
        await message.delete()
    except Exception:
        pass


def _fail_text(exc: Exception) -> str:
    if isinstance(exc, AiError):
        logger.warning("модель не ответила: %s", exc)
        return f"⚠️ {exc}"
    logger.exception("ai request failed")
    return f"⚠️ Не получилось: {exc}"


def _split(text: str, limit: int = _TEXT_LIMIT) -> list[str]:
    """Режем длинный ответ по абзацам — Telegram больше 4096 символов не примет."""
    text = text.strip()
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    chunks.append(text)
    return chunks


def _addressed(message: Message, username: str | None) -> bool:
    """В личке отвечаем всегда, в группе — на упоминание или ответ боту."""
    if message.chat.type == "private":
        return True
    reply = message.reply_to_message
    if reply is not None and reply.from_user is not None and reply.from_user.is_bot:
        if username is None or reply.from_user.username == username:
            return True
    if username:
        text = message.text or message.caption or ""
        return f"@{username}".lower() in text.lower()
    return False


def _strip_mention(text: str, username: str | None) -> str:
    if not username:
        return text
    pattern = rf"@{re.escape(username)}\b[ \t]*"
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()


def _greeting(ai: AiResponder, settings: Settings) -> str:
    lines = [
        "Кидайте ссылку на TikTok, Instagram (reel/пост/фото) или YouTube — "
        "пришлю медиа без вотермарки и анонимно.",
    ]
    if ai.can_draw:
        lines.append("🎨 /img <описание> — нарисовать картинку")
    if ai.can_edit:
        lines.append(
            "🖌 /edit <что поменять> — правка фото: подписью к фото или ответом на него"
        )
    if ai.can_chat:
        lines.append("💭 /ask <вопрос> — спросить модель, /reset — забыть контекст")
    if ai.can_see:
        lines.append("👀 фото с вопросом в подписи — расскажу, что на нём")
    if ai.can_hear:
        lines.append("🎤 голосовое — расшифрую и отвечу")
    if ai.can_sum:
        lines.append("📝 /sum <ссылка> — перескажу содержание ролика")
    if ai.can_speak:
        lines.append("🔊 /say <текст> — озвучу")
    if ai.can_tune:
        lines.append("⚙️ /set — модель и промпт для этого чата")
    if not ai.enabled:
        lines.append("(AI-функции выключены — заполните секцию AI в .env)")

    words = []
    if ai.can_draw and settings.image_triggers:
        words += [w.capitalize() for w in settings.image_triggers]
    if ai.can_chat and settings.chat_triggers:
        words += [w.capitalize() for w in settings.chat_triggers]
    if words:
        lines.append(
            "Команды можно не писать — хватит первого слова: "
            + ", ".join(f"«{w} …»" for w in words)
            + ". С приложенным фото слово для картинок означает правку."
        )
    return "\n".join(lines)


_DRAW_WHAT = "Генерация картинок"
_EDIT_WHAT = "Редактирование картинок"
_CHAT_WHAT = "Обычная модель"

_SUM_WHAT = "Пересказ роликов"
_SAY_WHAT = "Озвучка"
_TUNE_WHAT = "Настройки чата"

_DRAW_HINT = "Опишите картинку: /img рыжий кот в скафандре"
_EDIT_HINT = "Опишите правку: /edit добавь снег — подписью к фото или ответом на него"
_ASK_HINT = "Спросите что-нибудь: /ask как варить борщ"
_SUM_HINT = "Дайте ссылку: /sum https://youtu.be/… — или ответьте /sum на сообщение с ней"
_SAY_HINT = "Дайте текст: /say привет — или ответьте /say на сообщение"


@dataclass(frozen=True)
class _Job:
    """Одно действие модели — за ним одинаково ходят и команда, и триггерное слово."""

    ready: Callable[[], bool]
    run: Callable[[Message, str], Awaitable[None]]
    what: str
    hint: str


def build_router(
    settings: Settings,
    sender: VideoSender,
    ai: AiResponder,
    bot_username: str | None = None,
) -> Router:
    router = Router()

    allowed = settings.allowed_chats
    if allowed:
        router.message.filter(F.chat.id.in_(allowed))
        router.callback_query.filter(F.message.chat.id.in_(allowed))

    chat_mode = ChatMode(settings.ai_chat_mode)

    async def _unavailable(message: Message, ready: bool, what: str) -> bool:
        if ready:
            return False
        await message.reply(f"{what} не настроено — заполните секцию AI в .env.")
        return True

    def _wanted(message: Message) -> bool:
        if chat_mode is ChatMode.OFF:
            return False
        if chat_mode is ChatMode.ALWAYS:
            return True
        return _addressed(message, bot_username)

    async def _handle_link(message: Message, url: str) -> None:
        if looks_like_profile_url(url):
            await message.reply(
                "Это ссылка на профиль, добавить новые профили можно в .env"
            )
            return

        status = await message.reply("⬇️ Скачиваем…")
        try:
            await sender.send_from_url(
                message.chat.id, url, reply_to=message.message_id
            )
            await _delete(status)
        except Exception as exc:
            logger.exception("failed to handle %s", url)
            await _edit(status, f"Не получилось скачать: {exc}")

    async def _dispatch(message: Message, prompt: str, job: _Job) -> None:
        if await _unavailable(message, job.ready(), job.what):
            return
        if not prompt:
            await message.reply(job.hint)
            return
        await job.run(message, prompt)

    draw = _Job(lambda: ai.can_draw, ai.draw, _DRAW_WHAT, _DRAW_HINT)
    edit = _Job(lambda: ai.can_edit, ai.redraw, _EDIT_WHAT, _EDIT_HINT)
    ask = _Job(lambda: ai.can_chat, ai.ask, _CHAT_WHAT, _ASK_HINT)

    async def _run_summary(message: Message, text: str) -> bool:
        """Отдельно и раньше остальных: тут ссылка — не повод скачивать, а что пересказать."""
        prompt = _match_trigger(text, settings.summary_triggers)
        if prompt is None:
            return False
        if await _unavailable(message, ai.can_sum, _SUM_WHAT):
            return True
        url = find_media_url(prompt) or _replied_url(message)
        if not url:
            await message.reply(_SUM_HINT)
            return True
        await ai.summarize(message, url)
        return True

    async def _run_trigger(message: Message, text: str) -> bool:
        """Триггерные слова заменяют команды — в группе тоже работают без упоминания."""
        prompt = _match_trigger(text, settings.image_triggers)
        if prompt is not None:
            attached = _has_image(message) or _has_image(message.reply_to_message)
            await _dispatch(message, prompt, edit if attached else draw)
            return True

        prompt = _match_trigger(text, settings.edit_triggers)
        if prompt is not None:
            await _dispatch(message, prompt, edit)
            return True

        prompt = _match_trigger(text, settings.chat_triggers)
        if prompt is not None:
            await _dispatch(message, prompt, ask)
            return True

        return False

    @router.message(CommandStart())
    @router.message(Command("help"))
    async def on_start(message: Message) -> None:
        await message.answer(_greeting(ai, settings))

    @router.message(Command("img", "image", "gen"))
    async def on_img(message: Message, command: CommandObject) -> None:
        await _dispatch(message, (command.args or "").strip(), draw)

    @router.message(Command("edit", "redraw"))
    async def on_edit(message: Message, command: CommandObject) -> None:
        await _dispatch(message, (command.args or "").strip(), edit)

    @router.message(Command("ask", "chat"))
    async def on_ask(message: Message, command: CommandObject) -> None:
        quoted = ""
        if message.reply_to_message is not None:
            quoted = (
                message.reply_to_message.text or message.reply_to_message.caption or ""
            ).strip()
        prompt = "\n\n".join(p for p in ((command.args or "").strip(), quoted) if p)
        await _dispatch(message, prompt, ask)

    @router.message(Command("reset", "forget"))
    async def on_reset(message: Message) -> None:
        if await _unavailable(message, ai.can_chat, _CHAT_WHAT):
            return
        dropped = ai.forget(message)
        await message.reply("Контекст очищен." if dropped else "Контекст и так пуст.")

    @router.message(Command("sum", "summary", "tldr"))
    async def on_sum(message: Message, command: CommandObject) -> None:
        if await _unavailable(message, ai.can_sum, _SUM_WHAT):
            return
        url = find_media_url(command.args) or _replied_url(message)
        if not url:
            await message.reply(_SUM_HINT)
            return
        await ai.summarize(message, url)

    @router.message(Command("say", "voice", "tts"))
    async def on_say(message: Message, command: CommandObject) -> None:
        if await _unavailable(message, ai.can_speak, _SAY_WHAT):
            return
        text = (command.args or "").strip() or _replied_text(message)
        if not text:
            await message.reply(_SAY_HINT)
            return
        await ai.say(message, text)

    @router.message(Command("set", "settings"))
    async def on_set(message: Message, command: CommandObject) -> None:
        if await _unavailable(message, ai.can_tune, _TUNE_WHAT):
            return
        args = (command.args or "").strip()
        if args:
            await ai.set_pref(message, args)
        else:
            await ai.show_prefs(message)

    @router.message(F.voice | F.video_note | F.audio)
    async def on_voice(message: Message) -> None:
        if not ai.can_hear or not _wanted(message):
            return
        await ai.listen(message)

    @router.message(F.text | F.caption)
    async def on_message(message: Message) -> None:
        raw = (message.text or message.caption or "").strip()
        text = _strip_mention(raw, bot_username)
        if await _run_summary(message, text):
            return

        url = find_media_url(raw)
        if url:
            await _handle_link(message, url)
            return
        if raw.startswith("/"):
            return

        if await _run_trigger(message, text):
            return
        if not text or not _wanted(message):
            return

        has_photo = _has_image(message) or _has_image(message.reply_to_message)
        if has_photo and ai.can_edit and not ai.can_see:
            await ai.redraw(message, text)
        elif ai.can_chat or (has_photo and ai.can_see):
            await ai.ask(message, text)

    @router.callback_query(F.data.startswith(f"{_AGAIN}:"))
    async def on_again(callback: CallbackQuery) -> None:
        if not (ai.can_draw or ai.can_edit):
            await callback.answer("Генерация картинок выключена.", show_alert=True)
            return
        await ai.again(callback, (callback.data or "").split(":", 1)[1])

    return router
