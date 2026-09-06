from __future__ import annotations

import asyncio
import logging

from .bot import VideoSender
from .config import Settings
from .media import Downloader, Watch, parse_watch
from .state import State

logger = logging.getLogger(__name__)

_FEED_WINDOW = 15
_BETWEEN_SENDS = 2


class Poller:
    def __init__(
        self,
        settings: Settings,
        state: State,
        downloader: Downloader,
        sender: VideoSender,
    ) -> None:
        self._settings = settings
        self._state = state
        self._downloader = downloader
        self._sender = sender
        self._watches = _parse_all(settings.watches)

    async def run(self) -> None:
        logger.info(
            "poller started: %d source(s), every %ds -> chat %s",
            len(self._watches),
            self._settings.poll_interval_seconds,
            self._settings.target_chat_id,
        )
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("poll tick failed")
            await asyncio.sleep(self._settings.poll_interval_seconds)

    async def _tick(self) -> None:
        target = self._settings.target_chat_id
        if target is None:
            return
        for watch in self._watches:
            videos = await self._downloader.list_feed(watch, _FEED_WINDOW)
            if not videos:
                continue
            ids = [v.video_id for v in videos]

            if not await self._state.is_bootstrapped(watch.key):
                await self._state.seed(watch.key, ids)
                logger.info("bootstrapped %s with %d existing posts", watch.key, len(ids))
                continue

            fresh = [
                v for v in videos if not await self._state.is_seen(watch.key, v.video_id)
            ]
            for video in reversed(fresh):
                try:
                    await self._sender.send_from_url(target, video.page_url)
                    await self._state.mark_seen(watch.key, video.video_id)
                    logger.info("posted new post %s from %s", video.video_id, watch.key)
                    await asyncio.sleep(_BETWEEN_SENDS)
                except Exception:
                    logger.exception(
                        "failed to post %s from %s", video.video_id, watch.key
                    )


def _parse_all(sources: list[str]) -> list[Watch]:
    out: list[Watch] = []
    for raw in sources:
        try:
            out.append(parse_watch(raw))
        except ValueError as exc:
            logger.warning("не разобрали источник %r: %s", raw, exc)
    return out
