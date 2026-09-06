from __future__ import annotations

import asyncio
import logging

import httpx
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramUnauthorizedError
from aiogram.types import BotCommand

from . import selfcheck
from .ai import build_ai
from .bot import AiResponder, VideoSender, build_router
from .config import Settings
from .media import build_downloader
from .poller import Poller
from .state import State

logger = logging.getLogger(__name__)


def _selfcheck_reporter(bot: Bot, settings: Settings):
    mode = settings.selfcheck_report
    chat_id = settings.target_chat_id
    if mode == "off" or chat_id is None:
        return None

    async def report(result: selfcheck.Result) -> None:
        if mode == "fail" and result.ok:
            return
        try:
            await bot.send_message(chat_id, result.report[:4096])
        except Exception as exc:
            logger.warning("не отправили отчёт самопроверки: %s", exc)

    return report


def _commands(ai: AiResponder) -> list[BotCommand]:
    items = [BotCommand(command="help", description="что умеет бот")]
    if ai.can_draw:
        items.append(BotCommand(command="img", description="нарисовать картинку"))
    if ai.can_edit:
        items.append(BotCommand(command="edit", description="отредактировать фото"))
    if ai.can_chat:
        items.append(BotCommand(command="ask", description="спросить модель"))
    if ai.can_sum:
        items.append(BotCommand(command="sum", description="пересказать ролик"))
    if ai.can_speak:
        items.append(BotCommand(command="say", description="озвучить текст"))
    if ai.can_tune:
        items.append(BotCommand(command="set", description="модель и промпт чата"))
    if ai.can_chat:
        items.append(BotCommand(command="reset", description="забыть контекст"))
    return items


async def _describe_bot(bot: Bot, ai: AiResponder) -> str | None:
    try:
        me = await bot.get_me()
        await bot.set_my_commands(_commands(ai))
        return me.username
    except Exception as exc:
        logger.warning("не удалось получить профиль бота: %s", exc)
        return None


async def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    settings = Settings()

    proxy = settings.proxy_url or None
    if proxy:
        logger.info("routing Telegram + downloads through proxy (AI stays direct)")
    client = httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": "media-tg-bot"},
        proxy=proxy,
    )
    ai_client = httpx.AsyncClient(follow_redirects=True)
    downloader = build_downloader(settings, client)
    state = State(settings.state_file)

    session = AiohttpSession(proxy=proxy) if proxy else None
    bot = Bot(token=settings.bot_token, session=session)
    dp = Dispatcher()
    sender = VideoSender(bot, settings, downloader)
    ai = AiResponder(bot, settings, build_ai(settings, ai_client), state, downloader)
    if ai.enabled:
        logger.info(
            "ai enabled: chat=%s draw=%s edit=%s see=%s hear=%s speak=%s sum=%s (mode=%s)",
            ai.can_chat,
            ai.can_draw,
            ai.can_edit,
            ai.can_see,
            ai.can_hear,
            ai.can_speak,
            ai.can_sum,
            settings.ai_chat_mode,
        )
    else:
        logger.info("ai disabled (need both AI_API_BASE and AI_API_KEY)")

    username = await _describe_bot(bot, ai)
    dp.include_router(build_router(settings, sender, ai, username))

    tasks: list[asyncio.Task] = []
    if settings.target_chat_id is not None and settings.watches:
        tasks.append(asyncio.create_task(Poller(settings, state, downloader, sender).run()))
    else:
        logger.info(
            "polling disabled (need TARGET_CHAT_ID plus TIKTOK_PROFILES or WATCH_PROFILES)"
        )

    if settings.selfcheck_enabled:
        tasks.append(
            asyncio.create_task(
                selfcheck.run_forever(
                    settings.selfcheck_interval_hours,
                    _selfcheck_reporter(bot, settings),
                )
            )
        )

    try:
        while True:
            try:
                await dp.start_polling(bot)
            except asyncio.CancelledError:
                raise
            except TelegramUnauthorizedError:
                logger.error("invalid BOT_TOKEN — check your .env")
                raise
            except Exception:
                logger.exception("polling crashed — restarting in 5s")
                await asyncio.sleep(5)
            else:
                break
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await client.aclose()
        await ai_client.aclose()
        await bot.session.close()


def main() -> None:
    try:
        asyncio.run(run())
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    main()
