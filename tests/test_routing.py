"""Куда уходит сообщение: настоящие апдейты через настоящий диспетчер aiogram."""

from __future__ import annotations

import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import Update
from conftest import PNG, ai_settings, dummy_client, message, photo, run, voice
import conftest

import httpx

from ttbot.ai import Image, build_ai
from ttbot.bot import AiResponder, VideoSender, _has_image, build_router
from ttbot.media import build_downloader

CALLS: list[tuple[str, str]] = []


class SpyAi(AiResponder):
    """Считаем, какое действие выбрал роутер, до похода в сеть."""

    async def answer(self, message, prompt, images=None):
        CALLS.append(("vision" if images else "chat", prompt))

    async def ask(self, message, prompt):
        images = await self.images_of(message) if self.can_see else []
        await self.answer(message, prompt, images or None)

    async def draw(self, message, prompt):
        CALLS.append(("draw", prompt))

    async def redraw(self, message, prompt):
        CALLS.append(("edit", prompt))

    async def listen(self, message):
        CALLS.append(("listen", ""))

    async def say(self, message, text):
        CALLS.append(("say", text))

    async def summarize(self, message, url):
        CALLS.append(("sum", url))

    async def set_pref(self, message, args):
        CALLS.append(("set", args))

    async def show_prefs(self, message):
        CALLS.append(("prefs", ""))

    async def images_of(self, message):
        sources = (message, message.reply_to_message)
        return [Image.of(PNG) for m in sources if _has_image(m)]


class SpySender(VideoSender):
    async def send_from_url(self, chat_id, url, reply_to=None):
        CALLS.append(("download", url))
        return True


@pytest.fixture(autouse=True)
def offline_bot(monkeypatch):
    """Ни один вызов Bot API наружу не уходит."""

    async def fake_call(self, method, request_timeout=None):
        return conftest.message(self, "…", from_bot=True)

    monkeypatch.setattr(Bot, "__call__", fake_call)


def dispatcher(bot: Bot, **overrides) -> Dispatcher:
    base = dict(
        allowed_chat_ids="",
        ai_transcribe_model="whisper",
        ai_tts_model="tts",
    )
    base.update(overrides)
    settings = ai_settings(**base)
    client = dummy_client()
    ai = SpyAi(bot, settings, build_ai(settings, client), state=object(), downloader=object())
    sender = SpySender(bot, settings, build_downloader(settings, client))
    dp = Dispatcher()
    dp.include_router(build_router(settings, sender, ai, "MyBot"))
    return dp


def route(dp: Dispatcher, bot: Bot, msg) -> list[tuple[str, str]]:
    CALLS.clear()
    run(dp.feed_update(bot, Update(update_id=1, message=msg)))
    return list(CALLS)


# --- скачивание остаётся главным ----------------------------------------


def test_link_downloads(bot):
    dp = dispatcher(bot)
    msg = message(bot, "глянь https://www.tiktok.com/@a/video/123")
    assert route(dp, bot, msg) == [("download", "https://www.tiktok.com/@a/video/123")]


def test_photo_with_link_caption_downloads(bot):
    dp = dispatcher(bot)
    msg = message(bot, caption="https://youtu.be/abc", photo_sizes=photo())
    assert route(dp, bot, msg) == [("download", "https://youtu.be/abc")]


def test_image_trigger_does_not_steal_a_link(bot):
    dp = dispatcher(bot)
    msg = message(bot, "Нарисуй https://youtu.be/abc")
    assert route(dp, bot, msg) == [("download", "https://youtu.be/abc")]


# --- команды -------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("/img рыжий кот", [("draw", "рыжий кот")]),
        ("/gen кот", [("draw", "кот")]),
        ("/image кот", [("draw", "кот")]),
        ("/img@MyBot кот", [("draw", "кот")]),
        ("/ask как варить борщ", [("chat", "как варить борщ")]),
        ("/say привет", [("say", "привет")]),
        ("/sum https://youtu.be/abc", [("sum", "https://youtu.be/abc")]),
        ("/set", [("prefs", "")]),
        ("/set prompt Ты пират", [("set", "prompt Ты пират")]),
        ("/погода завтра", []), 
    ],
)
def test_commands(bot, text, expected):
    dp = dispatcher(bot)
    assert route(dp, bot, message(bot, text)) == expected


def test_edit_command_on_photo(bot):
    dp = dispatcher(bot)
    msg = message(bot, caption="/edit добавь снег", photo_sizes=photo())
    assert route(dp, bot, msg) == [("edit", "добавь снег")]


def test_commands_use_replied_message(bot):
    dp = dispatcher(bot)
    link = message(bot, "https://youtu.be/xyz")
    assert route(dp, bot, message(bot, "/sum", reply=link)) == [("sum", "https://youtu.be/xyz")]
    quote = message(bot, "вот этот текст")
    assert route(dp, bot, message(bot, "/say", reply=quote)) == [("say", "вот этот текст")]


def test_commands_without_arguments_only_hint(bot):
    dp = dispatcher(bot)
    for text in ("/img", "/edit", "/ask", "/say", "/sum"):
        assert route(dp, bot, message(bot, text)) == [], text


# --- фото: вопрос против правки -----------------------------------------


def test_photo_with_plain_caption_is_a_question(bot):
    """Раньше это уходило в перерисовку — и «что тут?» молча правило картинку."""
    dp = dispatcher(bot)
    msg = message(bot, caption="что здесь изображено?", photo_sizes=photo())
    assert route(dp, bot, msg) == [("vision", "что здесь изображено?")]


def test_reply_to_photo_is_a_question_too(bot):
    """Ответом на фото — тот же вопрос, что и подписью к нему."""
    dp = dispatcher(bot)
    source = message(bot, photo_sizes=photo())
    msg = message(bot, "что здесь изображено?", reply=source)
    assert route(dp, bot, msg) == [("vision", "что здесь изображено?")]


def test_reply_to_bot_photo_in_group_is_a_question(bot):
    """В группе ответ боту — уже обращение, упоминание не нужно."""
    dp = dispatcher(bot)
    sent = message(bot, photo_sizes=photo(), chat_type="supergroup", from_bot=True)
    msg = message(bot, "а что на фоне?", chat_type="supergroup", reply=sent)
    assert route(dp, bot, msg) == [("vision", "а что на фоне?")]


def test_reply_to_someone_elses_photo_in_group_needs_a_mention(bot):
    """Чужое фото в группе бот не комментирует, пока его не позвали."""
    dp = dispatcher(bot)
    source = message(bot, photo_sizes=photo(), chat_type="supergroup")
    quiet = message(bot, "что здесь?", chat_type="supergroup", reply=source)
    assert route(dp, bot, quiet) == []
    called = message(bot, "@MyBot что здесь?", chat_type="supergroup", reply=source)
    assert route(dp, bot, called) == [("vision", "что здесь?")]


def test_ask_command_on_a_photo_sees_it(bot):
    """Раньше /ask шёл чистым текстом и модель отвечала «фото не вижу»."""
    dp = dispatcher(bot)
    msg = message(bot, caption="/ask что здесь?", photo_sizes=photo())
    assert route(dp, bot, msg) == [("vision", "что здесь?")]


def test_ask_command_replying_to_a_photo_sees_it(bot):
    dp = dispatcher(bot)
    source = message(bot, photo_sizes=photo())
    assert route(dp, bot, message(bot, "/ask что здесь?", reply=source)) == [
        ("vision", "что здесь?")
    ]


def test_ask_without_photo_is_plain_chat(bot):
    dp = dispatcher(bot)
    assert route(dp, bot, message(bot, "/ask как варить борщ")) == [("chat", "как варить борщ")]


def test_ask_on_photo_without_vision_stays_text(bot):
    """Без vision-модели команда не должна пытаться приложить картинку."""
    dp = dispatcher(bot, ai_vision_model="off")
    msg = message(bot, caption="/ask что здесь?", photo_sizes=photo())
    assert route(dp, bot, msg) == [("chat", "что здесь?")]


def test_reply_without_photo_stays_plain_chat(bot):
    dp = dispatcher(bot)
    source = message(bot, "просто текст")
    assert route(dp, bot, message(bot, "а подробнее?", reply=source)) == [("chat", "а подробнее?")]


def test_without_vision_reply_to_photo_falls_back_to_editing(bot):
    dp = dispatcher(bot, ai_chat_model="", ai_vision_model="")
    source = message(bot, photo_sizes=photo())
    assert route(dp, bot, message(bot, "сделай ярче", reply=source)) == [("edit", "сделай ярче")]


@pytest.mark.parametrize(
    "caption, expected",
    [
        ("Сгенерируй ему шляпу", "ему шляпу"),
        ("Отредактируй: убери фон", "убери фон"),
        ("Измени фон на синий", "фон на синий"),
    ],
)
def test_photo_with_image_trigger_still_edits(bot, caption, expected):
    dp = dispatcher(bot)
    msg = message(bot, caption=caption, photo_sizes=photo())
    assert route(dp, bot, msg) == [("edit", expected)]


def test_image_trigger_replying_to_photo_edits_it(bot):
    dp = dispatcher(bot)
    source = message(bot, photo_sizes=photo())
    msg = message(bot, "Сгенерируй в стиле аниме", reply=source)
    assert route(dp, bot, msg) == [("edit", "в стиле аниме")]


def test_without_vision_photo_falls_back_to_editing(bot):
    dp = dispatcher(bot, ai_chat_model="", ai_vision_model="")
    msg = message(bot, caption="сделай ярче", photo_sizes=photo())
    assert route(dp, bot, msg) == [("edit", "сделай ярче")]


# --- голос ---------------------------------------------------------------


def test_voice_is_transcribed(bot):
    dp = dispatcher(bot)
    assert route(dp, bot, message(bot, voice_note=voice())) == [("listen", "")]


def test_voice_in_group_is_ignored(bot):
    dp = dispatcher(bot)
    msg = message(bot, voice_note=voice(), chat_type="supergroup")
    assert route(dp, bot, msg) == []


def test_voice_ignored_without_transcribe_model(bot):
    dp = dispatcher(bot, ai_transcribe_model="")
    assert route(dp, bot, message(bot, voice_note=voice())) == []


# --- пересказ ------------------------------------------------------------


def test_summary_trigger_beats_the_downloader(bot):
    dp = dispatcher(bot)
    msg = message(bot, "Перескажи https://youtu.be/abc")
    assert route(dp, bot, msg) == [("sum", "https://youtu.be/abc")]


def test_summary_trigger_on_reply(bot):
    dp = dispatcher(bot)
    link = message(bot, "https://youtu.be/xyz")
    assert route(dp, bot, message(bot, "Перескажи", reply=link)) == [("sum", "https://youtu.be/xyz")]


def test_bare_summary_trigger_only_hints(bot):
    dp = dispatcher(bot)
    assert route(dp, bot, message(bot, "Перескажи")) == []


# --- триггеры ------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Ответь как варить борщ", [("chat", "как варить борщ")]),
        ("Ответь, что такое Python", [("chat", "что такое Python")]),
        ("Сгенерируй рыжего кота", [("draw", "рыжего кота")]),
        ("нарисуй кота", [("draw", "кота")]),
        ("Сгенерируй", []),   
        ("Ответьте пожалуйста", [("chat", "Ответьте пожалуйста")]),
    ],
)
def test_triggers(bot, text, expected):
    dp = dispatcher(bot)
    assert route(dp, bot, message(bot, text)) == expected


def test_trigger_works_in_group_without_mention(bot):
    dp = dispatcher(bot)
    msg = message(bot, "Нарисуй кота", chat_type="supergroup")
    assert route(dp, bot, msg) == [("draw", "кота")]


# --- режимы общения ------------------------------------------------------


def test_group_stays_quiet_by_default(bot):
    dp = dispatcher(bot)
    assert route(dp, bot, message(bot, "ребята привет", chat_type="supergroup")) == []


def test_group_answers_on_mention(bot):
    dp = dispatcher(bot)
    msg = message(bot, "@MyBot что по погоде", chat_type="supergroup")
    assert route(dp, bot, msg) == [("chat", "что по погоде")]


def test_group_answers_on_reply(bot):
    dp = dispatcher(bot)
    reply = message(bot, "прошлый ответ", chat_type="supergroup", from_bot=True)
    msg = message(bot, "а подробнее?", chat_type="supergroup", reply=reply)
    assert route(dp, bot, msg) == [("chat", "а подробнее?")]


def test_mode_off_keeps_commands_and_triggers(bot):
    dp = dispatcher(bot, ai_chat_mode="off")
    assert route(dp, bot, message(bot, "привет")) == []
    assert route(dp, bot, message(bot, "/ask привет")) == [("chat", "привет")]
    assert route(dp, bot, message(bot, "Ответь привет")) == [("chat", "привет")]


def test_mode_always_answers_everything(bot):
    dp = dispatcher(bot, ai_chat_mode="always")
    msg = message(bot, "привет", chat_type="supergroup")
    assert route(dp, bot, msg) == [("chat", "привет")]


# --- белый список --------------------------------------------------------


def test_allowlist_blocks_foreign_chats(bot):
    dp = dispatcher(bot, allowed_chat_ids="999")
    assert route(dp, bot, message(bot, "/img кот")) == []


def test_allowlist_lets_its_own_chat_through(bot):
    dp = dispatcher(bot, allowed_chat_ids=str(conftest.CHAT_ID))
    assert route(dp, bot, message(bot, "/img кот")) == [("draw", "кот")]
