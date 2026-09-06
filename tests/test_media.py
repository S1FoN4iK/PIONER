"""Разбор ссылок: что скачиваем, что считаем профилем, за чем следим."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from conftest import run, settings

from ttbot.media import (
    Downloader,
    Media,
    Platform,
    ProviderError,
    TikwmProvider,
    _flat_entries,
    _short,
    detect_platform,
    find_media_url,
    looks_like_profile_url,
    parse_watch,
)


@pytest.mark.parametrize(
    "url, platform",
    [
        ("https://www.tiktok.com/@user/video/123", Platform.TIKTOK),
        ("https://vm.tiktok.com/ZM123/", Platform.TIKTOK),
        ("https://www.instagram.com/reel/Abc123/", Platform.INSTAGRAM),
        ("https://www.instagram.com/p/Abc123/", Platform.INSTAGRAM),
        ("https://youtu.be/dQw4w9WgXcQ", Platform.YOUTUBE),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", Platform.YOUTUBE),
        ("https://www.youtube.com/shorts/abc", Platform.YOUTUBE),
        ("https://example.com/video.mp4", None),
    ],
)
def test_detect_platform(url, platform):
    assert detect_platform(url) is platform


def test_find_url_inside_text():
    text = "глянь что нашёл https://youtu.be/abc круто же"
    assert find_media_url(text) == "https://youtu.be/abc"


def test_find_url_picks_the_first_one():
    text = "https://youtu.be/one и ещё https://www.tiktok.com/@a/video/2"
    assert find_media_url(text) == "https://youtu.be/one"


def test_find_url_without_links():
    assert find_media_url("просто текст") is None
    assert find_media_url(None) is None


def test_profile_link_is_not_a_post():
    assert looks_like_profile_url("https://www.tiktok.com/@user")
    assert not looks_like_profile_url("https://www.tiktok.com/@user/video/123")


# --- источники для поллера ----------------------------------------------


@pytest.mark.parametrize(
    "raw, key, platform",
    [
        ("user", "user", Platform.TIKTOK),
        ("@user", "user", Platform.TIKTOK),
        ("https://www.tiktok.com/@user", "user", Platform.TIKTOK),
        ("https://www.tiktok.com/@user/", "user", Platform.TIKTOK),
        ("https://www.instagram.com/user/", "user", Platform.INSTAGRAM),
        ("https://www.youtube.com/@channel", "https://www.youtube.com/@channel", Platform.YOUTUBE),
    ],
)
def test_parse_watch(raw, key, platform):
    watch = parse_watch(raw)
    assert watch.key == key
    assert watch.platform is platform


def test_bare_username_becomes_a_tiktok_link():
    assert parse_watch("user").url == "https://www.tiktok.com/@user"


def test_youtube_channel_gets_videos_tab():
    """Без /videos yt-dlp отдаёт вкладки канала вместо списка роликов."""
    assert parse_watch("https://www.youtube.com/@channel").url.endswith("/videos")
    already = "https://www.youtube.com/@channel/videos"
    assert parse_watch(already).url == already


def test_empty_source_is_rejected():
    with pytest.raises(ValueError):
        parse_watch("   ")


def test_watches_merge_both_settings_without_duplicates():
    merged = settings(
        tiktok_profiles="@a, b",
        watch_profiles="https://www.youtube.com/@c, b",
    ).watches
    assert merged == ["@a", "b", "https://www.youtube.com/@c"]


def test_watches_empty_by_default():
    assert settings().watches == []


# --- плоский разбор лент -------------------------------------------------


def test_flat_entries_unwraps_channel_tabs():
    info = {"entries": [{"entries": [{"id": "1"}, {"id": "2"}]}]}
    assert [e["id"] for e in _flat_entries(info)] == ["1", "2"]


def test_flat_entries_passes_plain_lists_through():
    assert [e["id"] for e in _flat_entries({"entries": [{"id": "1"}]})] == ["1"]


def test_flat_entries_survives_junk():
    assert _flat_entries(None) == []
    assert _flat_entries({}) == []
    assert _flat_entries({"entries": [None, "строка", {"id": "1"}]}) == [{"id": "1"}]


# --- лимит tikwm ---------------------------------------------------------


def tikwm(handler, min_interval: float = 0.0) -> TikwmProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TikwmProvider("https://tikwm.test", client, timeout=5, min_interval=min_interval)


def video_payload() -> dict:
    return {"code": 0, "data": {"id": "1", "hdplay": "/video.mp4",
                                "author": {"unique_id": "user"}, "title": "тест"}}


LIMIT = {"code": -1, "msg": "Free Api Limit: 1 request/second."}


def test_rate_limit_is_retried_not_surrendered():
    """Раньше «Free Api Limit» считался отказом и мы сразу падали на yt-dlp."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json=LIMIT if calls["n"] == 1 else video_payload())

    media = run(tikwm(handler).resolve("https://vt.tiktok.com/abc/"))
    assert media.video_id == "1"
    assert calls["n"] == 2


def test_rate_limit_gives_up_after_retries():
    def handler(request):
        return httpx.Response(200, json=LIMIT)

    with pytest.raises(ProviderError, match="Free Api Limit"):
        run(tikwm(handler).resolve("https://vt.tiktok.com/abc/"))


def test_other_errors_are_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"code": -1, "msg": "Url parsing is failed"})

    with pytest.raises(ProviderError, match="Url parsing"):
        run(tikwm(handler).resolve("https://vt.tiktok.com/abc/"))
    assert calls["n"] == 1


def test_http_errors_still_surface():
    def handler(request):
        return httpx.Response(403, text="Forbidden")

    with pytest.raises(httpx.HTTPStatusError):
        run(tikwm(handler).list_feed(parse_watch("user"), 15))


def test_api_calls_are_spaced_out():
    """Три ссылки подряд не должны бить в API одновременно."""
    stamps: list[float] = []

    def handler(request):
        stamps.append(time.monotonic())
        return httpx.Response(200, json=video_payload())

    provider = tikwm(handler, min_interval=0.05)

    async def three_at_once():
        await asyncio.gather(*(provider.resolve(f"https://vt.tiktok.com/{i}/") for i in range(3)))

    run(three_at_once())
    assert len(stamps) == 3
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert all(gap >= 0.04 for gap in gaps), gaps


def test_feed_retries_a_flaky_platform(no_feed_pause):
    """TikTok отдаёт ленту через раз — вторая попытка внутри тика должна спасать."""
    calls = {"n": 0}

    class Flaky:
        name = "flaky"
        platforms = frozenset({Platform.TIKTOK})

        async def list_feed(self, watch, count):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Unable to extract secondary user ID")
            return [Media(Platform.TIKTOK, "1", "user", "", "https://t/1")]

    posts = run(Downloader([Flaky()]).list_feed(parse_watch("user"), 5))
    assert [p.video_id for p in posts] == ["1"]
    assert calls["n"] == 2


def test_feed_returns_empty_when_everything_fails(no_feed_pause):
    class Dead:
        name = "dead"
        platforms = frozenset({Platform.TIKTOK})

        async def list_feed(self, watch, count):
            raise RuntimeError("нет")

    assert run(Downloader([Dead()]).list_feed(parse_watch("user"), 5)) == []


def test_long_ytdlp_errors_are_shortened():
    noise = RuntimeError("ERROR: [TikTok] 123: Unexpected response; " + "please report " * 40)
    assert len(_short(noise)) <= 161 and _short(noise).endswith("…")
