from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class Platform(str, Enum):
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"


TIKTOK_URL_RE = re.compile(r"https?://(?:[\w-]+\.)?tiktok\.com/[^\s]+", re.IGNORECASE)
INSTAGRAM_URL_RE = re.compile(
    r"https?://(?:www\.)?instagram\.com/"
    r"(?:stories/[\w.\-]+/\d+"  
    r"|(?:[\w.\-]+/)?(?:p|reel|reels|tv|share)/[\w\-]+)" 
    r"[^\s]*",
    re.IGNORECASE,
)
YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.|m\.|music\.)?(?:youtube\.com/(?:watch\?|shorts/|live/|embed/|v/)"
    r"|youtu\.be/)[^\s]+",
    re.IGNORECASE,
)

_PLATFORM_RES: list[tuple[Platform, re.Pattern[str]]] = [
    (Platform.TIKTOK, TIKTOK_URL_RE),
    (Platform.INSTAGRAM, INSTAGRAM_URL_RE),
    (Platform.YOUTUBE, YOUTUBE_URL_RE),
]

_VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif"}
_AUDIO_EXT = {".mp3", ".m4a", ".aac", ".opus", ".ogg", ".wav"}

_TIKWM_RETRIES = 2 
_FEED_ATTEMPTS = 2    
_FEED_RETRY_PAUSE = 3.0  


@dataclass
class Media:
    platform: Platform
    video_id: str
    author: str
    description: str
    page_url: str
    direct_url: str | None = None
    images: list[str] | None = None
    music_url: str | None = None
    width: int | None = None
    height: int | None = None
    duration: float | None = None

    @property
    def is_photo(self) -> bool:
        return bool(self.images)


@dataclass
class DownloadResult:
    media: Media
    provider: str
    videos: list[Path] = field(default_factory=list)
    images: list[Path] = field(default_factory=list)
    audio: Path | None = None

    @property
    def is_photo(self) -> bool:
        return not self.videos and bool(self.images)

    @property
    def is_single_video(self) -> bool:
        return len(self.videos) == 1 and not self.images


class ProviderError(RuntimeError):
    pass


class DownloadError(RuntimeError):
    pass


def detect_platform(url: str) -> Platform | None:
    for platform, pattern in _PLATFORM_RES:
        if pattern.match(url):
            return platform
    return None


def find_media_url(text: str | None) -> str | None:
    if not text:
        return None
    best: tuple[int, str] | None = None
    for _platform, pattern in _PLATFORM_RES:
        m = pattern.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), m.group(0))
    return best[1] if best else None


def looks_like_profile_url(url: str) -> bool:
    return bool(re.search(r"tiktok\.com/@[\w.\-]+/?$", url, re.IGNORECASE))


@dataclass(frozen=True)
class Watch:

    key: str  
    url: str 
    platform: Platform | None


_CHANNEL_RES: list[tuple[Platform, re.Pattern[str]]] = [
    (
        Platform.YOUTUBE,
        re.compile(
            r"https?://(?:www\.|m\.)?youtube\.com/(?:@[\w.\-]+|(?:c|user|channel)/[\w.\-]+)",
            re.IGNORECASE,
        ),
    ),
    (
        Platform.INSTAGRAM,
        re.compile(r"https?://(?:www\.)?instagram\.com/([\w.\-]+)/?$", re.IGNORECASE),
    ),
    (
        Platform.TIKTOK,
        re.compile(r"https?://(?:www\.)?tiktok\.com/@([\w.\-]+)", re.IGNORECASE),
    ),
]


def parse_watch(raw: str) -> Watch:
    item = raw.strip().rstrip("/")
    if not item:
        raise ValueError("пустой источник")

    if not item.startswith("http"):
        username = item.lstrip("@")
        return Watch(username, f"https://www.tiktok.com/@{username}", Platform.TIKTOK)

    for platform, pattern in _CHANNEL_RES:
        match = pattern.match(item)
        if not match:
            continue
        if platform is Platform.YOUTUBE:
            url = item if item.endswith("/videos") else f"{item}/videos"
            return Watch(item, url, platform)
        return Watch(match.group(1), item, platform)

    return Watch(item, item, None)


def page_url_for(username: str, video_id: str) -> str:
    return f"https://www.tiktok.com/@{username}/video/{video_id}"


def _as_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _short(exc: Exception, limit: int = 160) -> str:
    """yt-dlp вываливает в ошибку простыню про «report this issue» — режем."""
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _flat_entries(info: dict | None) -> list[dict]:
    """YouTube-каналы иногда отдают вкладки — записи лежат на уровень глубже."""
    out: list[dict] = []
    for entry in (info or {}).get("entries") or []:
        if not isinstance(entry, dict):
            continue
        nested = entry.get("entries")
        if isinstance(nested, list):
            out.extend(e for e in nested if isinstance(e, dict))
        else:
            out.append(entry)
    return out


def _classify(files: list[Path]) -> tuple[list[Path], list[Path], Path | None]:
    videos = [p for p in files if p.suffix.lower() in _VIDEO_EXT]
    images = [p for p in files if p.suffix.lower() in _IMAGE_EXT]
    audio = next((p for p in files if p.suffix.lower() in _AUDIO_EXT), None)
    return videos, images, audio


class TikwmProvider:
    """tikwm.com — explicit watermark-free stream (HD) and photo-post images. TikTok only."""

    name = "tikwm"
    platforms = frozenset({Platform.TIKTOK})

    def __init__(
        self,
        base: str,
        client: httpx.AsyncClient,
        timeout: int,
        min_interval: float = 1.1,
    ) -> None:
        self._base = base.rstrip("/")
        self._client = client
        self._timeout = timeout
        self._min_interval = max(0.0, min_interval)
        self._gate = asyncio.Lock()
        self._next_at = 0.0

    def _absolute(self, url: str) -> str:
        return f"{self._base}{url}" if url.startswith("/") else url

    async def _api(self, path: str, params: dict) -> dict:
        """Запрос к API с выдержкой между вызовами и повтором на рейт-лимит."""
        last = "bad response"
        for attempt in range(_TIKWM_RETRIES + 1):
            async with self._gate:
                pause = self._next_at - time.monotonic()
                if pause > 0:
                    await asyncio.sleep(pause)
                try:
                    resp = await self._client.get(
                        f"{self._base}{path}", params=params, timeout=self._timeout
                    )
                finally:
                    self._next_at = time.monotonic() + self._min_interval

            resp.raise_for_status()
            payload = resp.json()
            if payload.get("code") == 0 and payload.get("data"):
                return payload

            last = str(payload.get("msg") or "bad response")
            if "limit" not in last.lower() or attempt == _TIKWM_RETRIES:
                break
            logger.info("tikwm: %s — ждём и пробуем ещё раз", last)
        raise ProviderError(f"tikwm: {last}")

    async def resolve(self, url: str) -> Media:
        payload = await self._api("/api/", {"url": url, "hd": 1})
        d = payload["data"]

        images = d.get("images") or None
        music = None
        if images:
            music = d.get("music") or (d.get("music_info") or {}).get("play")
            direct = None
        else:
            direct = d.get("hdplay") or d.get("play") or d.get("wmplay")
            if not direct:
                raise ProviderError("tikwm: no playable url in response")

        author = (d.get("author") or {}).get("unique_id") or ""
        return Media(
            platform=Platform.TIKTOK,
            video_id=str(d.get("id") or ""),
            author=author,
            description=d.get("title") or "",
            page_url=url,
            direct_url=self._absolute(direct) if direct else None,
            images=[self._absolute(u) for u in images] if images else None,
            music_url=self._absolute(music) if music else None,
            duration=_as_float(d.get("duration")),
        )

    async def list_feed(self, watch: Watch, count: int) -> list[Media]:
        username = watch.key
        payload = await self._api(
            "/api/user/posts", {"unique_id": username, "count": count}
        )
        items = (payload.get("data") or {}).get("videos") or []
        videos: list[Media] = []
        for it in items:
            vid = str(it.get("video_id") or it.get("id") or "")
            if not vid:
                continue
            videos.append(
                Media(
                    platform=Platform.TIKTOK,
                    video_id=vid,
                    author=username,
                    description=it.get("title") or "",
                    page_url=page_url_for(username, vid),
                    duration=_as_float(it.get("duration")),
                )
            )
        return videos

    async def _stream_to(self, url: str, dest: Path) -> None:
        headers = {"User-Agent": _UA, "Referer": "https://www.tiktok.com/"}
        async with self._client.stream(
            "GET", url, headers=headers, timeout=self._timeout
        ) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                async for chunk in resp.aiter_bytes(64 * 1024):
                    fh.write(chunk)

    async def download(self, media: Media, dest_dir: Path) -> DownloadResult:
        if media.direct_url is None and not media.images:
            media = await self.resolve(media.page_url)

        if media.images:
            return await self._download_photo(media, dest_dir)

        assert media.direct_url
        dest = dest_dir / f"{media.video_id or 'video'}_{self.name}.mp4"
        await self._stream_to(media.direct_url, dest)
        return DownloadResult(media=media, provider=self.name, videos=[dest])

    async def _download_photo(self, media: Media, dest_dir: Path) -> DownloadResult:
        image_paths: list[Path] = []
        for i, img_url in enumerate(media.images or []):
            dest = dest_dir / f"{media.video_id or 'photo'}_{i:02d}.jpg"
            try:
                await self._stream_to(img_url, dest)
                image_paths.append(dest)
            except Exception as exc:
                logger.warning("tikwm: image %d download failed: %s", i, exc)

        audio_path: Path | None = None
        if media.music_url:
            a = dest_dir / f"{media.video_id or 'photo'}.mp3"
            try:
                await self._stream_to(media.music_url, a)
                audio_path = a
            except Exception as exc:
                logger.warning("tikwm: music download failed: %s", exc)

        return DownloadResult(
            media=media,
            provider=self.name,
            images=image_paths,
            audio=audio_path,
        )


class YtDlpProvider:
    """yt-dlp — handles TikTok, Instagram (incl. photo carousels) and YouTube."""

    name = "ytdlp"
    platforms = frozenset({Platform.TIKTOK, Platform.INSTAGRAM, Platform.YOUTUBE})

    def __init__(
        self,
        timeout: int,
        proxy: str | None = None,
        cookies_file: str | None = None,
        cookies_from_browser: str | None = None,
        youtube_max_height: int = 720,
    ) -> None:
        self._timeout = timeout
        self._proxy = proxy or None
        self._cookies_file = cookies_file or None
        self._cookies_from_browser = cookies_from_browser or None
        self._youtube_max_height = youtube_max_height

    @property
    def _has_auth(self) -> bool:
        return bool(self._cookies_file or self._cookies_from_browser)

    def _base_opts(self) -> dict:
        opts: dict = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": self._timeout,
            "ignoreerrors": False,
            "retries": 3,
            "fragment_retries": 3,
        }
        if self._proxy:
            opts["proxy"] = self._proxy
        if self._cookies_file:
            opts["cookiefile"] = self._cookies_file
        if self._cookies_from_browser:
            opts["cookiesfrombrowser"] = (self._cookies_from_browser,)
        return opts

    def _format_for(self, platform: Platform | None) -> str:
        if platform is Platform.YOUTUBE:
            h = self._youtube_max_height
            return f"best[height<=?{h}][ext=mp4]/best[height<=?{h}]/best"
        return "best"

    def _extract(
        self,
        url: str,
        download: bool,
        outtmpl: str | None = None,
        platform: Platform | None = None,
    ) -> dict:
        import yt_dlp

        opts = self._base_opts()
        opts["skip_download"] = not download
        if platform is Platform.INSTAGRAM:
            opts["noplaylist"] = False
        if download:
            opts["outtmpl"] = outtmpl
            opts["format"] = self._format_for(platform)
            opts["restrictfilenames"] = True
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=download)

    def _extract_flat(self, url: str, count: int) -> dict:
        import yt_dlp

        opts = self._base_opts()
        opts.pop("noplaylist", None)
        opts["extract_flat"] = True
        opts["playlistend"] = count
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)

    async def resolve(self, url: str) -> Media:
        platform = detect_platform(url) or Platform.TIKTOK
        info = await asyncio.to_thread(self._extract, url, False, None, platform)
        return _media_from_ytdlp(info, url, platform)

    async def list_feed(self, watch: Watch, count: int) -> list[Media]:
        info = await asyncio.to_thread(self._extract_flat, watch.url, count)
        platform = watch.platform or Platform.TIKTOK
        out: list[Media] = []
        for e in _flat_entries(info)[:count]:
            vid = str(e.get("id") or "")
            if not vid:
                continue
            page = e.get("url") or e.get("webpage_url")
            if not page and platform is Platform.TIKTOK:
                page = page_url_for(watch.key, vid)
            if not page:
                continue
            out.append(
                Media(
                    platform=platform,
                    video_id=vid,
                    author=e.get("uploader_id") or watch.key,
                    description=e.get("title") or "",
                    page_url=page,
                )
            )
        return out

    async def download(self, media: Media, dest_dir: Path) -> DownloadResult:
        outtmpl = str(dest_dir / f"{self.name}_%(id)s.%(ext)s")
        info = await asyncio.to_thread(
            self._extract, media.page_url, True, outtmpl, media.platform
        )
        files = sorted(p for p in dest_dir.glob(f"{self.name}_*") if p.is_file())
        videos, images, audio = _classify(files)
        if not videos and not images:
            raise ProviderError(self._empty_reason(media, info))
        return DownloadResult(
            media=media, provider=self.name, videos=videos, images=images, audio=audio
        )

    def _empty_reason(self, media: Media, info: dict | None) -> str:
        entries = (info or {}).get("entries")
        empty_playlist = isinstance(entries, list) and not entries
        if media.platform is Platform.INSTAGRAM and empty_playlist and not self._has_auth:
            return (
                "Instagram отдал 0 файлов — для постов и сторис нужен вход. "
                "Задайте COOKIES_FILE (cookies.txt) или COOKIES_FROM_BROWSER в .env."
            )
        return "ytdlp: no output files produced"


def _media_from_ytdlp(info: dict, url: str, platform: Platform) -> Media:
    return Media(
        platform=platform,
        video_id=str(info.get("id") or ""),
        author=info.get("uploader_id") or info.get("uploader") or "",
        description=info.get("description") or info.get("title") or "",
        page_url=info.get("webpage_url") or url,
        width=info.get("width"),
        height=info.get("height"),
        duration=_as_float(info.get("duration")),
    )


class Downloader:
    def __init__(self, providers: list) -> None:
        if not providers:
            raise ValueError("no providers configured")
        self._providers = providers

    @staticmethod
    def _ensure_nonempty(result: DownloadResult) -> None:
        result.videos = [p for p in result.videos if p.exists() and p.stat().st_size > 0]
        result.images = [p for p in result.images if p.exists() and p.stat().st_size > 0]
        if result.audio is not None and (
            not result.audio.exists() or result.audio.stat().st_size == 0
        ):
            result.audio = None
        if not result.videos and not result.images:
            raise ProviderError("nothing usable was downloaded")

    def _providers_for(self, platform: Platform | None) -> list:
        if platform is None:
            return list(self._providers)
        return [p for p in self._providers if platform in getattr(p, "platforms", ())]

    async def download(self, url: str, dest_dir: Path) -> DownloadResult:
        dest_dir.mkdir(parents=True, exist_ok=True)
        platform = detect_platform(url)
        providers = self._providers_for(platform)
        if not providers:
            raise DownloadError(f"no provider supports this link: {url}")
        last: Exception | None = None
        for provider in providers:
            try:
                media = await provider.resolve(url)
                result = await provider.download(media, dest_dir)
                self._ensure_nonempty(result)
                kind = "photo" if result.is_photo else "video"
                logger.info("downloaded %s (%s) via %s", url, kind, provider.name)
                return result
            except Exception as exc:
                last = exc
                logger.warning("provider %s failed for %s: %s", provider.name, url, exc)
        raise DownloadError(f"all providers failed for {url}: {last}")

    async def list_feed(self, watch: Watch, count: int) -> list[Media]:
        """Ленту платформы отдают через раз, поэтому цепочку прогоняем несколько раз."""
        failed = False
        for attempt in range(_FEED_ATTEMPTS):
            for provider in self._providers_for(watch.platform):
                if not hasattr(provider, "list_feed"):
                    continue
                try:
                    videos = await provider.list_feed(watch, count)
                    if videos:
                        return videos
                except Exception as exc:
                    failed = True
                    logger.debug(
                        "provider %s list_feed failed for %s: %s",
                        provider.name, watch.key, _short(exc),
                    )
            if attempt + 1 < _FEED_ATTEMPTS:
                await asyncio.sleep(_FEED_RETRY_PAUSE)

        if failed:
            logger.warning("лента %s сейчас недоступна — повторим позже", watch.key)
        return []


def build_downloader(settings, client: httpx.AsyncClient) -> Downloader:
    proxy = getattr(settings, "proxy_url", "") or None
    cookies_file = getattr(settings, "cookies_file", "") or None
    cookies_browser = getattr(settings, "cookies_from_browser", "") or None
    yt_height = getattr(settings, "youtube_max_height", 720)

    def make_ytdlp() -> YtDlpProvider:
        return YtDlpProvider(
            settings.request_timeout,
            proxy=proxy,
            cookies_file=cookies_file,
            cookies_from_browser=cookies_browser,
            youtube_max_height=yt_height,
        )

    providers: list = []
    for name in settings.providers:
        if name == "tikwm":
            providers.append(
                TikwmProvider(
                    settings.tikwm_api_base,
                    client,
                    settings.request_timeout,
                    getattr(settings, "tikwm_min_interval", 1.1),
                )
            )
        elif name == "ytdlp":
            providers.append(make_ytdlp())
        else:
            logger.warning("unknown provider %r - skipped", name)

    if not any(isinstance(p, YtDlpProvider) for p in providers):
        providers.append(make_ytdlp())
    return Downloader(providers)
