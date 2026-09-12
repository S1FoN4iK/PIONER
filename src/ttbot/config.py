from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_CHAT_MODES = ("off", "mention", "always")
_IMAGE_APIS = ("images", "chat")
_REPORT_MODES = ("off", "fail", "always")
_DOC_PARTS = ("file", "inline")
_PROXY_ON = ("on", "true", "yes", "1")
_PROXY_OFF = ("off", "false", "no", "0", "")


def _one_of(value: object, allowed: tuple[str, ...], field: str, blank_ok: bool = False) -> object:
    if not isinstance(value, str):
        return value
    v = value.strip().lower()
    if not v and blank_ok:
        return ""
    if v not in allowed:
        raise ValueError(f"{field}: ожидается одно из {', '.join(allowed)}, получено {value!r}")
    return v


def _triggers(raw: str) -> tuple[str, ...]:
    words = {w.strip().lower() for w in raw.split(",") if w.strip()}
    return tuple(sorted(words, key=len, reverse=True))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Telegram ---
    bot_token: str
    target_chat_id: int | None = None
    allowed_chat_ids: str = ""

    @field_validator("target_chat_id", "ai_temperature", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    # --- Polling ---
    tiktok_profiles: str = ""
    watch_profiles: str = ""
    poll_interval_seconds: int = 300

    # --- Download ---
    provider_order: str = "tikwm,ytdlp"
    tikwm_api_base: str = "https://www.tikwm.com"
    tikwm_min_interval: float = 1.1
    max_file_size_mb: int = 49
    request_timeout: int = 60
    send_caption: bool = True
    youtube_max_height: int = 720

    # --- Network ---
    proxy_url: str = ""
    proxy_telegram: str = "on"
    proxy_downloads: str = "on"
    proxy_ai: str = "off"

    @field_validator("proxy_telegram", "proxy_downloads", "proxy_ai", mode="before")
    @classmethod
    def _check_proxy(cls, v: object) -> object:
        if not isinstance(v, str):
            return v
        raw = v.strip()
        if raw.lower() in _PROXY_ON or raw.lower() in _PROXY_OFF or "://" in raw:
            return raw
        raise ValueError(
            f"прокси: ожидается on, off или URL вида scheme://host:port, получено {v!r}"
        )

    def _resolve_proxy(self, raw: str) -> str | None:
        v = raw.strip()
        if v.lower() in _PROXY_ON:
            return self.proxy_url.strip() or None
        if v.lower() in _PROXY_OFF:
            return None
        return v

    @property
    def telegram_proxy(self) -> str | None:
        return self._resolve_proxy(self.proxy_telegram)

    @property
    def download_proxy(self) -> str | None:
        return self._resolve_proxy(self.proxy_downloads)

    @property
    def ai_proxy(self) -> str | None:
        return self._resolve_proxy(self.proxy_ai)

    # --- Auth (optional, mostly for Instagram / age-gated YouTube) ---
    cookies_file: str = ""
    cookies_from_browser: str = ""

    # --- AI (any OpenAI-compatible endpoint: chat + images) ---
    ai_api_base: str = ""
    ai_api_key: str = ""
    ai_chat_model: str = ""
    ai_image_model: str = ""
    ai_image_edit_model: str = ""
    ai_image_api: str = "images"
    ai_image_edit_api: str = ""
    ai_system_prompt: str = ""
    ai_chat_mode: str = "mention"
    ai_chat_triggers: str = "Ответь,Объясни,Расскажи"
    ai_image_triggers: str = "Сгенерируй,Нарисуй"
    ai_edit_triggers: str = "Отредактируй,Измени,Переделай"
    ai_history_turns: int = 64
    ai_history_ttl_minutes: int = 60
    ai_max_tokens: int = 0
    ai_temperature: float | None = None
    ai_image_size: str = "" 
    ai_image_count: int = 1
    ai_timeout: int = 180
    ai_retries: int = 2

    # --- AI: vision, voice, retelling ---
    ai_vision_model: str = ""
    ai_media_model: str = ""
    ai_attachments: bool = True
    ai_doc_part: str = "file"
    ai_attachment_max_mb: int = 20
    ai_transcribe_model: str = ""
    ai_tts_model: str = ""
    ai_tts_voice: str = "alloy"
    ai_voice_reply: bool = False
    ai_transcribe_max_mb: int = 25
    ai_summary_prompt: str = (
        "Перескажи содержание кратко и по делу, на русском, "
        "списком ключевых мыслей. Без вступлений."
    )
    ai_summary_triggers: str = "Перескажи,Саммари"
    ai_regen_button: bool = True
    ai_chat_settings: bool = True

    @field_validator("ai_chat_mode", mode="before")
    @classmethod
    def _check_chat_mode(cls, v: object) -> object:
        return _one_of(v, _CHAT_MODES, "AI_CHAT_MODE")

    @field_validator("ai_image_api", mode="before")
    @classmethod
    def _check_image_api(cls, v: object) -> object:
        return _one_of(v, _IMAGE_APIS, "AI_IMAGE_API")

    @field_validator("ai_doc_part", mode="before")
    @classmethod
    def _check_doc_part(cls, v: object) -> object:
        return _one_of(v, _DOC_PARTS, "AI_DOC_PART")

    @field_validator("ai_image_edit_api", mode="before")
    @classmethod
    def _check_image_edit_api(cls, v: object) -> object:
        return _one_of(v, _IMAGE_APIS, "AI_IMAGE_EDIT_API", blank_ok=True)

    # --- Self-check ---
    selfcheck_enabled: bool = True
    selfcheck_interval_hours: float = 12
    selfcheck_report: str = "fail"

    @field_validator("selfcheck_report", mode="before")
    @classmethod
    def _check_report(cls, v: object) -> object:
        return _one_of(v, _REPORT_MODES, "SELFCHECK_REPORT")

    # --- Storage ---
    state_file: str = "state.db"

    @property
    def profiles(self) -> list[str]:
        return [p.strip() for p in self.tiktok_profiles.split(",") if p.strip()]

    @property
    def providers(self) -> list[str]:
        return [p.strip().lower() for p in self.provider_order.split(",") if p.strip()]

    @property
    def allowed_chats(self) -> set[int]:
        out: set[int] = set()
        for raw in self.allowed_chat_ids.split(","):
            raw = raw.strip()
            if raw:
                out.add(int(raw))
        return out

    @property
    def ai_edit_api(self) -> str:
        return self.ai_image_edit_api or self.ai_image_api

    @property
    def chat_triggers(self) -> tuple[str, ...]:
        return _triggers(self.ai_chat_triggers)

    @property
    def image_triggers(self) -> tuple[str, ...]:
        return _triggers(self.ai_image_triggers)

    @property
    def edit_triggers(self) -> tuple[str, ...]:
        return _triggers(self.ai_edit_triggers)

    @property
    def summary_triggers(self) -> tuple[str, ...]:
        return _triggers(self.ai_summary_triggers)

    @property
    def watches(self) -> list[str]:
        raw = f"{self.tiktok_profiles},{self.watch_profiles}"
        seen: dict[str, None] = {}
        for item in raw.split(","):
            item = item.strip()
            if item:
                seen.setdefault(item, None)
        return list(seen)
