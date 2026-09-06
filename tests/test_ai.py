"""Клиент к шлюзу: разбор ответов, повторы, память диалога, включение функций."""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest
from conftest import PNG, ai_settings, dummy_client, run, settings

from ttbot.ai import (
    AiClient,
    AiError,
    ChatMemory,
    Image,
    ImageApi,
    Overrides,
    _normalize_base,
    _scan_images,
    _text_from_choices,
    build_ai,
)

PNG_B64 = base64.b64encode(PNG).decode()


# --- адрес шлюза ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://api.openai.com", "https://api.openai.com/v1"),
        ("https://api.openai.com/", "https://api.openai.com/v1"),
        ("https://gw.example/api/v1/", "https://gw.example/api/v1"),
        ("  ", ""),
    ],
)
def test_base_url_normalisation(raw, expected):
    assert _normalize_base(raw) == expected


# --- разбор ответов ------------------------------------------------------


def test_scan_images_openai_shape():
    refs = []
    _scan_images({"data": [{"b64_json": PNG_B64}, {"url": "https://x/y.png"}]}, refs)
    assert [r.b64 for r in refs] == [PNG_B64, None]
    assert refs[1].url == "https://x/y.png"


def test_scan_images_chat_shape():
    """Так картинку отдают Gemini-образные модели через OpenRouter."""
    refs = []
    _scan_images(
        {
            "choices": [
                {
                    "message": {
                        "content": "готово",
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
                            }
                        ],
                    }
                }
            ]
        },
        refs,
    )
    assert [r.b64 for r in refs] == [PNG_B64]


def test_scan_images_finds_data_uri_in_markdown():
    refs = []
    _scan_images(
        {"choices": [{"message": {"content": f"вот: ![](data:image/png;base64,{PNG_B64})"}}]},
        refs,
    )
    assert len(refs) == 1


def test_scan_images_ignores_unrelated_payload():
    refs = []
    _scan_images({"id": "req_1", "model": "gpt", "usage": {"total_tokens": 10}}, refs)
    assert refs == []


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"choices": [{"message": {"content": " hi "}}]}, "hi"),
        (
            {
                "choices": [
                    {"message": {"content": [{"type": "text", "text": "a"},
                                             {"type": "text", "text": "b"}]}}
                ]
            },
            "ab",
        ),
        ({"choices": []}, ""),
        ({}, ""),
    ],
)
def test_text_from_choices(payload, expected):
    assert _text_from_choices(payload) == expected


def test_image_sniffs_mime_from_bytes():
    assert Image.of(PNG).mime == "image/png"
    assert Image.of(b"\xff\xd8\xff" + b"\x00" * 10).mime == "image/jpeg"
    assert Image.of(PNG).filename == "image.png"


# --- память диалога ------------------------------------------------------


def test_memory_keeps_last_turns():
    mem = ChatMemory(max_turns=2, ttl_seconds=60)
    for i in range(3):
        mem.remember((1, 1), f"q{i}", f"a{i}")
    history = mem.history((1, 1))
    assert len(history) == 4
    assert history[0]["content"] == "q1"


def test_memory_is_per_user_and_forgettable():
    mem = ChatMemory(max_turns=2, ttl_seconds=60)
    mem.remember((1, 1), "q", "a")
    assert mem.history((1, 2)) == []
    assert mem.forget((1, 1)) is True
    assert mem.history((1, 1)) == []
    assert mem.forget((1, 1)) is False


def test_memory_zero_turns_stores_nothing():
    mem = ChatMemory(max_turns=0, ttl_seconds=60)
    mem.remember((1, 1), "q", "a")
    assert mem.history((1, 1)) == []


# --- какие функции включены ---------------------------------------------


def test_nothing_enabled_without_key():
    ai = build_ai(settings(), dummy_client())
    assert not any([ai.enabled, ai.can_chat, ai.can_draw, ai.can_edit, ai.can_see])


def test_each_model_enables_its_own_feature():
    ai = build_ai(ai_settings(ai_image_model=""), dummy_client())
    assert ai.can_chat and not ai.can_draw and not ai.can_edit

    ai = build_ai(ai_settings(ai_chat_model=""), dummy_client())
    assert ai.can_draw and ai.can_edit and not ai.can_chat


def test_fallback_models():
    """Пустые «дочерние» модели наследуют основные."""
    ai = build_ai(ai_settings(), dummy_client())
    assert ai.can_edit
    assert ai.can_see 


def test_voice_needs_its_own_models():
    ai = build_ai(ai_settings(), dummy_client())
    assert not ai.can_hear and not ai.can_speak
    ai = build_ai(
        ai_settings(ai_transcribe_model="whisper", ai_tts_model="tts"), dummy_client()
    )
    assert ai.can_hear and ai.can_speak


def test_vision_can_be_switched_off_while_chat_stays():
    """Для чат-модели без поддержки картинок нужен способ отключить зрение."""
    ai = build_ai(ai_settings(ai_vision_model="off"), dummy_client())
    assert ai.can_chat and not ai.can_see


def test_edit_api_inherits_image_api():
    assert ai_settings(ai_image_api="chat").ai_edit_api == "chat"
    assert ai_settings(ai_image_api="chat", ai_image_edit_api="images").ai_edit_api == "images"


def test_bad_enum_values_are_rejected_at_startup():
    with pytest.raises(ValueError, match="AI_CHAT_MODE"):
        settings(ai_chat_mode="bogus")
    with pytest.raises(ValueError, match="AI_IMAGE_API"):
        settings(ai_image_api="bogus")


# --- обращения к шлюзу ---------------------------------------------------


def gateway(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def client(handler, **kwargs) -> AiClient:
    defaults = dict(
        base_url="https://gw.example",
        api_key="k",
        chat_model="chat-model",
        image_model="image-model",
        transcribe_model="whisper",
        tts_model="tts",
        retries=0,
    )
    defaults.update(kwargs)
    return AiClient(gateway(handler), **defaults)


def ok_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if "/missing" in path:
        return httpx.Response(404, json={"error": {"message": "нет такого"}})
    if path.endswith("/chat/completions"):
        assert request.headers["authorization"] == "Bearer k"
        body = json.loads(request.content)
        if body["model"] == "vision-model":
            return httpx.Response(200, json={"choices": [{"message": {
                "content": "",
                "images": [{"image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}]}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ответ"}}]})
    if path.endswith("/images/generations"):
        return httpx.Response(200, json={"data": [{"b64_json": PNG_B64}]})
    if path.endswith("/images/edits"):
        assert b'name="image"' in request.content, "одна картинка идёт в поле image"
        return httpx.Response(200, json={"data": [{"url": "https://cdn.example/out.png"}]})
    if path.endswith("/audio/transcriptions"):
        return httpx.Response(200, json={"text": " привет мир "})
    if path.endswith("/audio/speech"):
        return httpx.Response(200, content=b"OggS-audio")
    if request.url.host == "cdn.example":
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})
    return httpx.Response(404, json={"error": {"message": "нет такого"}})


def test_chat_round_trip():
    assert run(client(ok_handler).chat("привет")) == "ответ"


def test_draw_decodes_base64():
    images = run(client(ok_handler).draw("кот"))
    assert len(images) == 1 and images[0].data == PNG


def test_edit_follows_remote_url():
    images = run(client(ok_handler).redraw("добавь снег", [Image.of(PNG)]))
    assert len(images) == 1 and images[0].data == PNG


def test_edit_without_source_is_rejected():
    with pytest.raises(AiError):
        run(client(ok_handler).redraw("что-то", []))


def test_chat_mode_image_generation():
    ai = client(ok_handler, image_model="vision-model", image_api=ImageApi.CHAT)
    images = run(ai.draw("кот"))
    assert len(images) == 1 and images[0].data == PNG


def test_transcribe_and_speak():
    ai = client(ok_handler)
    assert run(ai.transcribe(b"audio-bytes", "voice.ogg")) == "привет мир"
    assert run(ai.speak("привет")) == b"OggS-audio"


def test_vision_sends_image_part():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "кот"}}]})

    ai = client(handler, vision_model="vision-model")
    assert run(ai.chat("что тут?", None, Overrides(), [Image.of(PNG)])) == "кот"
    assert seen["model"] == "vision-model"
    parts = seen["messages"][-1]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_overrides_win_over_env():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ок"}}]})

    ai = client(handler, system_prompt="из .env")
    run(ai.chat("вопрос", None, Overrides(chat_model="свой", system_prompt="свой промпт")))
    assert seen["model"] == "свой"
    assert seen["messages"][0] == {"role": "system", "content": "свой промпт"}


def test_gateway_error_reaches_the_user():
    ai = client(ok_handler, base_url="https://gw.example/missing")
    with pytest.raises(AiError, match="404"):
        run(ai.chat("привет"))


# --- повторы -------------------------------------------------------------


def no_sleep(monkeypatch) -> list[float]:
    delays: list[float] = []

    async def fake(seconds):
        delays.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake)
    return delays


def flaky(fail_times: int, status: int = 429):
    state = {"left": fail_times}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["left"] > 0:
            state["left"] -= 1
            return httpx.Response(status, json={"error": {"message": "перегрузка"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ответ"}}]})

    return handler


def test_retries_recover_from_429(monkeypatch):
    delays = no_sleep(monkeypatch)
    assert run(client(flaky(2), retries=2).chat("привет")) == "ответ"
    assert len(delays) == 2


def test_retries_give_up_and_report(monkeypatch):
    no_sleep(monkeypatch)
    with pytest.raises(AiError, match="429"):
        run(client(flaky(5), retries=2).chat("привет"))


def test_no_retry_without_budget(monkeypatch):
    delays = no_sleep(monkeypatch)
    with pytest.raises(AiError, match="429"):
        run(client(flaky(1), retries=0).chat("привет"))
    assert delays == []


def test_client_errors_are_not_retried(monkeypatch):
    """Неверный ключ или модель повторять бессмысленно — сразу наверх."""
    delays = no_sleep(monkeypatch)
    with pytest.raises(AiError, match="401"):
        run(client(flaky(5, status=401), retries=3).chat("привет"))
    assert delays == []


def test_retry_after_header_is_respected(monkeypatch):
    delays = no_sleep(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if not delays:
            return httpx.Response(429, headers={"retry-after": "7"}, json={})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ответ"}}]})

    assert run(client(handler, retries=1).chat("привет")) == "ответ"
    assert delays == [7.0]


def test_network_errors_are_retried(monkeypatch):
    delays = no_sleep(monkeypatch)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("сеть отвалилась")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ответ"}}]})

    assert run(client(handler, retries=1).chat("привет")) == "ответ"
    assert len(delays) == 1
