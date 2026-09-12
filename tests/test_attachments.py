"""Мультимодальные вложения: как видео, аудио и документы уходят в chat/completions."""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from aiogram import Bot
from conftest import PNG, document, message, run, settings, video

from ttbot.ai import AiClient, AiError, Attachment, Image, _content_part

MP4 = b"\x00\x00\x00\x18ftypmp42"


# --- обёртки частей ------------------------------------------------------


def test_image_stays_image_url():
    part = _content_part(Image.of(PNG))
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_audio_goes_as_input_audio_with_format():
    part = _content_part(Attachment(b"OggS" * 8, "audio/ogg", "voice.ogg"))
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "ogg"
    assert base64.b64decode(part["input_audio"]["data"]) == b"OggS" * 8


def test_mp3_format_is_normalized():
    assert _content_part(Attachment(b"ID3", "audio/mpeg", "a.mp3"))["input_audio"]["format"] == "mp3"


def test_video_goes_as_input_audio_not_video_url():
    """video_url шлюз отвергает; видео доезжает только этой обёрткой."""
    part = _content_part(Attachment(MP4, "video/mp4", "v.mp4"))
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "mp4"
    assert base64.b64decode(part["input_audio"]["data"]) == MP4


@pytest.mark.parametrize(
    "mime, fmt",
    [
        ("video/quicktime", "mov"),
        ("video/webm", "webm"),
        ("video/x-matroska", "mkv"),
        ("video/mp4; codecs=avc1", "mp4"),
    ],
)
def test_video_formats(mime, fmt):
    assert _content_part(Attachment(MP4, mime, "v"))["input_audio"]["format"] == fmt


def test_document_goes_as_file_with_name():
    part = _content_part(Attachment(b"%PDF-1.4", "application/pdf", "doc.pdf"))
    assert part["type"] == "file"
    assert part["file"]["filename"] == "doc.pdf"
    assert part["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_unknown_mime_falls_back_to_octet_stream():
    part = _content_part(Attachment(b"x", "", "blob"))
    assert part["file"]["file_data"].startswith("data:application/octet-stream;base64,")


@pytest.mark.parametrize("mime", ["text/plain", "text/csv", "application/json"])
def test_text_files_are_inlined_into_the_prompt(mime):
    part = _content_part(Attachment("привет, мир".encode(), mime, "note.txt"))
    assert part["type"] == "text"
    assert "привет, мир" in part["text"]
    assert "note.txt" in part["text"]


def test_huge_text_file_is_trimmed():
    part = _content_part(Attachment(b"a" * 500_000, "text/plain", "big.txt"))
    assert len(part["text"]) < 100_000


def test_broken_encoding_does_not_crash():
    part = _content_part(Attachment(b"\xff\xfe\x00 hi", "text/plain", "bad.txt"))
    assert "hi" in part["text"]


# --- какая модель получает вложение --------------------------------------


def spy_client(seen: dict, **kwargs) -> AiClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ок"}}]})

    defaults = dict(
        base_url="https://gw.example",
        api_key="k",
        chat_model="chat-model",
        vision_model="vision-model",
        media_model="media-model",
        retries=0,
    )
    defaults.update(kwargs)
    return AiClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)), **defaults)


def test_video_goes_to_the_media_model():
    seen: dict = {}
    run(spy_client(seen).chat("что тут?", files=[Attachment(MP4, "video/mp4", "v.mp4")]))
    assert seen["model"] == "media-model"


def test_audio_goes_to_the_media_model():
    seen: dict = {}
    run(spy_client(seen).chat("?", files=[Attachment(b"o", "audio/ogg", "v.ogg")]))
    assert seen["model"] == "media-model"


def test_picture_still_goes_to_the_vision_model():
    seen: dict = {}
    run(spy_client(seen).chat("?", images=[Image.of(PNG)]))
    assert seen["model"] == "vision-model"


def test_pdf_goes_to_the_vision_model():
    seen: dict = {}
    run(spy_client(seen).chat("?", files=[Attachment(b"%PDF", "application/pdf", "d.pdf")]))
    assert seen["model"] == "vision-model"


def test_text_file_needs_no_multimodal_model():
    seen: dict = {}
    run(spy_client(seen).chat("?", files=[Attachment(b"hi", "text/plain", "n.txt")]))
    assert seen["model"] == "chat-model"


def test_media_model_defaults_to_vision_then_chat():
    seen: dict = {}
    run(spy_client(seen, media_model="", vision_model="").chat(
        "?", files=[Attachment(MP4, "video/mp4", "v.mp4")]
    ))
    assert seen["model"] == "chat-model"


def test_media_model_can_be_switched_off_separately():
    ai = spy_client({}, media_model="off")
    assert ai.can_see and not ai.can_watch


# --- отказ шлюза ---------------------------------------------------------


def refusing_client() -> AiClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Invalid parameter value"}})

    return AiClient(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        base_url="https://gw.example",
        api_key="k",
        chat_model="chat-model",
        media_model="text-only-model",
        retries=0,
    )


def test_rejected_video_explains_which_setting_to_change():
    with pytest.raises(AiError) as err:
        run(refusing_client().chat("?", files=[Attachment(MP4, "video/mp4", "v.mp4")]))
    text = str(err.value)
    assert "AI_MEDIA_MODEL" in text and "видео" in text
    assert "Invalid parameter value" in text


def test_plain_text_failure_keeps_its_original_wording():
    with pytest.raises(AiError) as err:
        run(refusing_client().chat("привет"))
    assert "AI_MEDIA_MODEL" not in str(err.value)


# --- путь от сообщения Telegram до запроса в шлюз ------------------------


class Sent(list):
    """Что бот отправил в чат: текст reply и последующих правок."""

    @property
    def last(self) -> str:
        return self[-1] if self else ""


@pytest.fixture
def responder(monkeypatch, bot):
    """AiResponder на живом Bot: сеть Telegram и шлюз подменены."""
    from ttbot.bot import AiResponder

    sent = Sent()
    seen: dict = {}

    async def fake_call(self, method, request_timeout=None):
        text = getattr(method, "text", None)
        if text is not None:
            sent.append(text)
        return message(self, "…", from_bot=True)

    async def fake_download(self, target, destination, **kwargs):
        destination.write(MP4 if getattr(target, "width", None) else b"hello file")

    monkeypatch.setattr(Bot, "__call__", fake_call)
    monkeypatch.setattr(Bot, "download", fake_download)

    def build(**overrides):
        base = dict(
            ai_api_base="https://gw.example",
            ai_api_key="k",
            ai_chat_model="chat-model",
            ai_media_model="media-model",
        )
        base.update(overrides)
        conf = settings(bot_token="t", **base)
        return AiResponder(bot, conf, spy_client(seen, chat_model="chat-model",
                                                 vision_model="chat-model",
                                                 media_model=conf.ai_media_model or "chat-model"))

    return build, sent, seen


def test_video_with_caption_reaches_the_model(responder):
    build, sent, seen = responder
    ai = build()
    msg = message(ai._bot, caption="что тут?", video_file=video())
    run(ai.ask(msg, "что тут?"))

    parts = seen["messages"][-1]["content"]
    assert seen["model"] == "media-model"
    assert parts[0]["text"] == "что тут?"
    assert parts[1]["type"] == "input_audio"
    assert base64.b64decode(parts[1]["input_audio"]["data"]) == MP4


def test_video_in_a_reply_reaches_the_model(responder):
    build, sent, seen = responder
    ai = build()
    src = message(ai._bot, video_file=video())
    msg = message(ai._bot, "а что на видео?", reply=src)
    run(ai.ask(msg, "а что на видео?"))
    assert seen["messages"][-1]["content"][1]["type"] == "input_audio"


def test_text_document_is_inlined(responder):
    build, sent, seen = responder
    ai = build()
    msg = message(ai._bot, caption="перескажи", doc=document())
    run(ai.ask(msg, "перескажи"))
    parts = seen["messages"][-1]["content"]
    assert parts[1]["type"] == "text" and "hello file" in parts[1]["text"]


def test_oversized_attachment_tells_the_user(responder):
    build, sent, seen = responder
    ai = build(ai_attachment_max_mb=1)
    msg = message(ai._bot, caption="что тут?", video_file=video(size=50 * 1024 * 1024))
    run(ai.ask(msg, "что тут?"))

    assert not seen, "к шлюзу ходить не за чем"
    assert "AI_ATTACHMENT_MAX_MB" in sent.last and "50" in sent.last


def test_attachments_can_be_switched_off(responder):
    build, sent, seen = responder
    ai = build(ai_attachments=False)
    msg = message(ai._bot, caption="что тут?", video_file=video())
    run(ai.ask(msg, "что тут?"))
    assert seen["messages"][-1]["content"] == "что тут?"


# --- AI_DOC_PART: документ частью file или сырым data-URI ----------------


def test_document_stays_a_file_part_by_default():
    part = _content_part(Attachment(b"%PDF", "application/pdf", "d.pdf"))
    assert part["type"] == "file"


def test_inline_mode_sends_document_as_data_uri():
    part = _content_part(Attachment(b"%PDF", "application/pdf", "d.pdf"), doc_inline=True)
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:application/pdf;base64,")


def test_inline_mode_does_not_touch_video_or_text():
    assert _content_part(Attachment(MP4, "video/mp4", "v"), doc_inline=True)["type"] == "input_audio"
    assert _content_part(Attachment(b"hi", "text/plain", "n"), doc_inline=True)["type"] == "text"


def test_inline_document_goes_to_the_media_model():
    seen: dict = {}
    run(spy_client(seen, doc_inline=True).chat(
        "?", files=[Attachment(b"%PDF", "application/pdf", "d.pdf")]
    ))
    assert seen["model"] == "media-model"


def test_doc_part_is_validated_at_startup():
    from conftest import settings as conf

    with pytest.raises(ValueError, match="AI_DOC_PART"):
        conf(ai_doc_part="bogus")
    assert conf(ai_doc_part="inline").ai_doc_part == "inline"
