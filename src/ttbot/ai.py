from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum

import httpx

logger = logging.getLogger(__name__)

_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_RETRY_CAP_SECONDS = 30.0

_OFF = frozenset({"off", "-", "none", "нет"})

_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

_DATA_URI_RE = re.compile(r"data:(image/[\w.+-]+);base64,([A-Za-z0-9+/=\s]{32,})")


class AiError(RuntimeError):
    """Ошибка модели или шлюза — текст показываем пользователю как есть."""


class ChatMode(str, Enum):
    """Когда обычная модель отвечает на простой текст."""

    OFF = "off"  
    MENTION = "mention"  
    ALWAYS = "always"  


class ImageApi(str, Enum):
    """Каким эндпоинтом шлюз умеет в картинки."""

    IMAGES = "images" 
    CHAT = "chat"   


def _sniff_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"GIF8"):
        return "image/gif"
    return "image/png"


@dataclass
class Image:
    data: bytes
    mime: str = "image/png"

    @classmethod
    def of(cls, data: bytes) -> Image:
        return cls(data, _sniff_mime(data))

    @property
    def filename(self) -> str:
        return f"image{_MIME_EXT.get(self.mime, '.png')}"

    @property
    def kind(self) -> str:
        return "image"

    @property
    def size_mb(self) -> float:
        return len(self.data) / (1024 * 1024)


@dataclass
class _ImageRef:
    """Картинка в ответе шлюза: либо base64, либо ссылка, которую надо забрать."""

    b64: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class Overrides:
    """Настройки конкретного чата поверх .env. Пустое поле -> берём глобальное."""

    chat_model: str = ""
    image_model: str = ""
    system_prompt: str = ""
    image_size: str = ""

    @property
    def empty(self) -> bool:
        return not (
            self.chat_model or self.image_model or self.system_prompt or self.image_size
        )


_NO_OVERRIDES = Overrides()


@dataclass
class _Dialogue:
    messages: list[dict] = field(default_factory=list)
    touched_at: float = 0.0


class ChatMemory:
    """Короткая история диалога на пару (чат, пользователь). Живёт в памяти процесса."""

    def __init__(self, max_turns: int, ttl_seconds: int) -> None:
        self._max_messages = max(0, max_turns) * 2
        self._ttl = max(0, ttl_seconds)
        self._store: dict[tuple[int, int], _Dialogue] = {}

    def _fresh(self, dialogue: _Dialogue) -> bool:
        return not self._ttl or (time.monotonic() - dialogue.touched_at) < self._ttl

    def history(self, key: tuple[int, int]) -> list[dict]:
        dialogue = self._store.get(key)
        if dialogue is None:
            return []
        if not self._fresh(dialogue):
            del self._store[key]
            return []
        return list(dialogue.messages)

    def remember(self, key: tuple[int, int], question: str, answer: str) -> None:
        if not self._max_messages:
            return
        self._evict()
        dialogue = self._store.setdefault(key, _Dialogue())
        dialogue.messages.append({"role": "user", "content": question})
        dialogue.messages.append({"role": "assistant", "content": answer})
        del dialogue.messages[: -self._max_messages]
        dialogue.touched_at = time.monotonic()

    def forget(self, key: tuple[int, int]) -> bool:
        return self._store.pop(key, None) is not None

    def _evict(self) -> None:
        for key in [k for k, v in self._store.items() if not self._fresh(v)]:
            del self._store[key]


def _normalize_base(base: str) -> str:
    """Голый хост дополняем до /v1; путь, заданный руками, не трогаем."""
    base = base.strip().rstrip("/")
    if not base:
        return ""
    if httpx.URL(base).path in ("", "/"):
        base = f"{base}/v1"
    return base


def _error_text(resp: httpx.Response) -> str:
    detail = ""
    try:
        payload = resp.json()
    except ValueError:
        detail = resp.text.strip()
    else:
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            detail = str(error.get("message") or error.get("type") or "")
        elif isinstance(error, str):
            detail = error
        if not detail:
            detail = str(payload)
    return f"HTTP {resp.status_code}: {detail[:400] or 'без подробностей'}"


def _text_from_choices(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return "".join(parts).strip()
    return ""


def _backoff(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(float(retry_after), _RETRY_CAP_SECONDS)
        except ValueError:
            pass
    return min(2.0**attempt + random.uniform(0, 0.5), _RETRY_CAP_SECONDS)


_AUDIO_FORMATS = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/aac": "aac",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/flac": "flac",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/opus": "ogg",
}

_VIDEO_FORMATS = {
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "video/webm": "webm",
    "video/x-matroska": "mkv",
    "video/mpeg": "mpeg",
    "video/3gpp": "3gp",
    "video/x-msvideo": "avi",
}

_TEXT_MIMES = frozenset(
    {
        "application/json",
        "application/xml",
        "application/javascript",
        "application/x-yaml",
        "application/yaml",
        "application/x-sh",
        "application/sql",
        "application/csv",
    }
)
_TEXT_CHARS = 60_000

IMAGE, AUDIO, VIDEO, TEXT, FILE = "image", "audio", "video", "text", "file"
_RICH = frozenset({IMAGE, AUDIO, VIDEO, FILE})
_HEAVY = frozenset({AUDIO, VIDEO})


def _kind_of(mime: str) -> str:
    mime = (mime or "").split(";", 1)[0].strip().lower()
    for family in (IMAGE, AUDIO, VIDEO):
        if mime.startswith(f"{family}/"):
            return family
    if mime.startswith("text/") or mime in _TEXT_MIMES:
        return TEXT
    return FILE


@dataclass
class Attachment:

    data: bytes
    mime: str
    filename: str

    @property
    def kind(self) -> str:
        return _kind_of(self.mime)

    @property
    def size_mb(self) -> float:
        return len(self.data) / (1024 * 1024)


def _media_format(mime: str, table: dict[str, str]) -> str:
    mime = (mime or "").split(";", 1)[0].strip().lower()
    known = table.get(mime)
    if known:
        return known
    tail = mime.split("/", 1)[-1] if "/" in mime else mime
    return tail.removeprefix("x-") or "bin"


def _as_text(item: Attachment) -> str:
    body = item.data.decode("utf-8", "replace")[:_TEXT_CHARS].strip()
    return f"Содержимое файла «{item.filename}»:\n\n{body}"


def _content_part(item: Image | Attachment, doc_inline: bool = False) -> dict:
    """Часть user-сообщения в формате OpenAI-совместимого chat/completions."""
    mime = item.mime or "application/octet-stream"
    kind = item.kind
    if kind == TEXT:
        return {"type": "text", "text": _as_text(item)}
    encoded = base64.b64encode(item.data).decode("ascii")
    if kind == IMAGE or (kind == FILE and doc_inline):
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
    if kind in _HEAVY:
        table = _AUDIO_FORMATS if kind == AUDIO else _VIDEO_FORMATS
        return {
            "type": "input_audio",
            "input_audio": {"data": encoded, "format": _media_format(mime, table)},
        }
    return {
        "type": "file",
        "file": {
            "filename": item.filename,
            "file_data": f"data:{mime};base64,{encoded}",
        },
    }


def _image_part(image: Image) -> dict:
    return _content_part(image)


def _ref_from_url(value: str) -> _ImageRef | None:
    value = value.strip()
    match = _DATA_URI_RE.search(value)
    if match:
        return _ImageRef(b64=match.group(2))
    if value.startswith(("http://", "https://")):
        return _ImageRef(url=value)
    return None


def _scan_images(node: object, out: list[_ImageRef]) -> None:
    """Шлюзы кладут картинки в разные места — обходим ответ целиком."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "b64_json" and isinstance(value, str) and value.strip():
                out.append(_ImageRef(b64=value))
            elif key in ("url", "image_url") and isinstance(value, str):
                ref = _ref_from_url(value)
                if ref is not None:
                    out.append(ref)
            else:
                _scan_images(value, out)
    elif isinstance(node, list):
        for item in node:
            _scan_images(item, out)
    elif isinstance(node, str):
        out.extend(_ImageRef(b64=m.group(2)) for m in _DATA_URI_RE.finditer(node))


class AiClient:

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str,
        api_key: str,
        chat_model: str = "",
        image_model: str = "",
        image_edit_model: str = "",
        vision_model: str = "",
        media_model: str = "",
        transcribe_model: str = "",
        tts_model: str = "",
        tts_voice: str = "alloy",
        doc_inline: bool = False,
        image_api: ImageApi = ImageApi.IMAGES,
        image_edit_api: ImageApi = ImageApi.IMAGES,
        system_prompt: str = "",
        max_tokens: int = 0,
        temperature: float | None = None,
        image_size: str = "",
        image_count: int = 1,
        timeout: int = 180,
        retries: int = 2,
    ) -> None:
        self._client = client
        self._base = _normalize_base(base_url)
        self._key = api_key.strip()
        self._chat_model = chat_model.strip()
        self._image_model = image_model.strip()
        self._edit_model = image_edit_model.strip() or self._image_model
        vision = vision_model.strip()
        self._vision_model = "" if vision.lower() in _OFF else (vision or self._chat_model)
        media = media_model.strip()
        self._media_model = "" if media.lower() in _OFF else (media or self._vision_model)
        self._transcribe_model = transcribe_model.strip()
        self._tts_model = tts_model.strip()
        self._tts_voice = tts_voice.strip() or "alloy"
        self._doc_inline = doc_inline
        self._image_api = image_api
        self._edit_api = image_edit_api
        self._system_prompt = system_prompt.strip()
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._image_size = image_size.strip()
        self._image_count = max(1, image_count)
        self._timeout = timeout
        self._retries = max(0, retries)

    @property
    def enabled(self) -> bool:
        return bool(self._base and self._key)

    @property
    def can_chat(self) -> bool:
        return self.enabled and bool(self._chat_model)

    @property
    def can_draw(self) -> bool:
        return self.enabled and bool(self._image_model)

    @property
    def can_edit(self) -> bool:
        return self.enabled and bool(self._edit_model)

    @property
    def can_see(self) -> bool:
        return self.enabled and bool(self._vision_model)

    @property
    def can_watch(self) -> bool:
        """Есть кому отдать видео и аудио файлом, без расшифровки."""
        return self.enabled and bool(self._media_model)

    @property
    def can_hear(self) -> bool:
        return self.enabled and bool(self._transcribe_model)

    @property
    def can_speak(self) -> bool:
        return self.enabled and bool(self._tts_model)

    async def chat(
        self,
        prompt: str,
        history: list[dict] | None = None,
        over: Overrides = _NO_OVERRIDES,
        images: list[Image] | None = None,
        files: list[Attachment] | None = None,
    ) -> str:
        media: list[Image | Attachment] = [*(images or []), *(files or [])]
        kinds = {m.kind for m in media}
        model = over.chat_model or self._model_for(kinds)

        content: object = prompt
        if media:
            content = [
                {"type": "text", "text": prompt},
                *(_content_part(m, self._doc_inline) for m in media),
            ]

        messages: list[dict] = []
        system = over.system_prompt or self._system_prompt
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(history or [])
        messages.append({"role": "user", "content": content})

        try:
            payload = await self._post(
                "/chat/completions", json=self._chat_body(model, messages)
            )
        except AiError as exc:
            raise self._media_hint(exc, model, kinds) from exc
        text = _text_from_choices(payload)
        if not text:
            raise AiError("модель вернула пустой ответ")
        return text

    def _model_for(self, kinds: set[str]) -> str:
        heavy = _HEAVY | {FILE} if self._doc_inline else _HEAVY
        if kinds & heavy:
            return self._media_model or self._vision_model or self._chat_model
        if kinds & _RICH:
            return self._vision_model or self._chat_model
        return self._chat_model

    @staticmethod
    def _media_hint(exc: AiError, model: str, kinds: set[str]) -> AiError:
        heavy = sorted(kinds & _HEAVY)
        if not heavy:
            return exc
        what = " и ".join("видео" if k == VIDEO else "аудио" for k in heavy)
        return AiError(
            f"{exc}\n\nПохоже, модель {model} не принимает {what}. "
            "Укажите в AI_MEDIA_MODEL мультимодальную модель "
            "(например, google/gemini-2.5-flash)."
        )

    async def draw(self, prompt: str, over: Overrides = _NO_OVERRIDES) -> list[Image]:
        model = over.image_model or self._image_model
        size = over.image_size or self._image_size
        if self._image_api is ImageApi.CHAT:
            return await self._chat_images(model, prompt, [])
        body = {"model": model, "prompt": prompt, "n": self._image_count}
        if size:
            body["size"] = size
        payload = await self._post("/images/generations", json=body)
        return self._require_images(await self._images_from(payload))

    async def redraw(
        self, prompt: str, sources: list[Image], over: Overrides = _NO_OVERRIDES
    ) -> list[Image]:
        if not sources:
            raise AiError("нечего редактировать — нужна исходная картинка")
        model = over.image_model or self._edit_model
        size = over.image_size or self._image_size
        if self._edit_api is ImageApi.CHAT:
            return await self._chat_images(model, prompt, sources)

        data = {"model": model, "prompt": prompt, "n": str(self._image_count)}
        if size:
            data["size"] = size
        field_name = "image" if len(sources) == 1 else "image[]"
        files = [
            (field_name, (img.filename, img.data, img.mime)) for img in sources
        ]
        payload = await self._post("/images/edits", data=data, files=files)
        return self._require_images(await self._images_from(payload))

    async def transcribe(self, audio: bytes, filename: str) -> str:
        payload = await self._post(
            "/audio/transcriptions",
            data={"model": self._transcribe_model},
            files=[("file", (filename, audio, "application/octet-stream"))],
        )
        text = str(payload.get("text") or "").strip()
        if not text:
            raise AiError("в аудио не нашлось речи")
        return text

    async def speak(self, text: str) -> bytes:
        audio = await self._post_raw(
            "/audio/speech",
            json={
                "model": self._tts_model,
                "input": text,
                "voice": self._tts_voice,
                "response_format": "opus",
            },
        )
        if not audio:
            raise AiError("шлюз вернул пустое аудио")
        return audio

    def _chat_body(self, model: str, messages: list[dict]) -> dict:
        body: dict = {"model": model, "messages": messages}
        if self._max_tokens:
            body["max_tokens"] = self._max_tokens
        if self._temperature is not None:
            body["temperature"] = self._temperature
        return body

    async def _chat_images(
        self, model: str, prompt: str, sources: list[Image]
    ) -> list[Image]:
        content: list[dict] = [{"type": "text", "text": prompt}]
        content.extend(_image_part(img) for img in sources)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["image", "text"],
        }
        payload = await self._post("/chat/completions", json=body)
        images = await self._images_from(payload)
        if not images:
            raise AiError(_text_from_choices(payload) or "модель не вернула картинку")
        return images

    @staticmethod
    def _require_images(images: list[Image]) -> list[Image]:
        if not images:
            raise AiError("модель не вернула картинку")
        return images

    async def _post(
        self,
        path: str,
        *,
        json: dict | None = None,
        data: dict | None = None,
        files: list | None = None,
    ) -> dict:
        resp = await self._request(path, json=json, data=data, files=files)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise AiError("шлюз вернул не JSON") from exc
        if not isinstance(payload, dict):
            raise AiError("шлюз вернул неожиданный ответ")
        return payload

    async def _post_raw(self, path: str, *, json: dict) -> bytes:
        """Для эндпоинтов, отвечающих файлом, а не JSON (например, TTS)."""
        resp = await self._request(path, json=json)
        return resp.content

    async def _request(
        self,
        path: str,
        *,
        json: dict | None = None,
        data: dict | None = None,
        files: list | None = None,
    ) -> httpx.Response:
        if not self.enabled:
            raise AiError("не заданы AI_API_BASE / AI_API_KEY")
        url = f"{self._base}{path}"
        last = ""
        for attempt in range(self._retries + 1):
            retry_after: str | None = None
            try:
                resp = await self._client.post(
                    url,
                    headers={"Authorization": f"Bearer {self._key}"},
                    json=json,
                    data=data,
                    files=files,
                    timeout=self._timeout,
                )
            except httpx.HTTPError as exc:
                last = f"не достучались до {self._base}: {exc}"
            else:
                if resp.status_code < 400:
                    return resp
                last = _error_text(resp)
                if resp.status_code not in _RETRY_STATUS:
                    raise AiError(last)
                retry_after = resp.headers.get("retry-after")

            if attempt == self._retries:
                break
            delay = _backoff(attempt, retry_after)
            logger.warning(
                "%s: %s — повтор через %.1f с (%d/%d)",
                path, last, delay, attempt + 1, self._retries,
            )
            await asyncio.sleep(delay)
        raise AiError(last or "запрос не удался")

    async def _images_from(self, payload: dict) -> list[Image]:
        refs: list[_ImageRef] = []
        _scan_images(payload, refs)
        images: list[Image] = []
        seen: set[str] = set()
        for ref in refs:
            token = ref.b64 or ref.url or ""
            if not token or token in seen:
                continue
            seen.add(token)
            image = await self._materialize(ref)
            if image is not None:
                images.append(image)
        return images

    async def _materialize(self, ref: _ImageRef) -> Image | None:
        if ref.b64 is not None:
            try:
                data = base64.b64decode(ref.b64, validate=False)
            except (binascii.Error, ValueError):
                return None
            return Image.of(data) if data else None

        assert ref.url
        try:
            resp = await self._client.get(ref.url, timeout=self._timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("не забрали картинку %s: %s", ref.url, exc)
            return None
        if not resp.headers.get("content-type", "").startswith("image/"):
            return None
        return Image.of(resp.content)


def build_ai(settings, client: httpx.AsyncClient) -> AiClient:
    return AiClient(
        client,
        base_url=settings.ai_api_base,
        api_key=settings.ai_api_key,
        chat_model=settings.ai_chat_model,
        image_model=settings.ai_image_model,
        image_edit_model=settings.ai_image_edit_model,
        vision_model=settings.ai_vision_model,
        media_model=settings.ai_media_model,
        transcribe_model=settings.ai_transcribe_model,
        tts_model=settings.ai_tts_model,
        tts_voice=settings.ai_tts_voice,
        doc_inline=settings.ai_doc_part == "inline",
        image_api=ImageApi(settings.ai_image_api),
        image_edit_api=ImageApi(settings.ai_edit_api),
        system_prompt=settings.ai_system_prompt,
        max_tokens=settings.ai_max_tokens,
        temperature=settings.ai_temperature,
        image_size=settings.ai_image_size,
        image_count=settings.ai_image_count,
        timeout=settings.ai_timeout,
        retries=settings.ai_retries,
    )
