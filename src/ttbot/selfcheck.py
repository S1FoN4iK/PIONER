"""Самопроверка: гоняем тесты при старте и дальше по расписанию."""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_TAIL = 1500


@dataclass(frozen=True)
class Result:
    ok: bool
    summary: str
    details: str = ""

    @property
    def report(self) -> str:
        mark = "✅" if self.ok else "❌"
        text = f"{mark} Самопроверка: {self.summary}"
        return f"{text}\n\n{self.details}" if self.details else text


def _tests_dir() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    tests = root / "tests"
    return tests if tests.is_dir() else None


async def run_once() -> Result:
    tests = _tests_dir()
    if tests is None:
        return Result(False, "каталог tests не найден — нечего проверять")

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            str(tests),
            cwd=str(tests.parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as exc:
        return Result(False, f"не удалось запустить pytest: {exc}")

    raw, _ = await proc.communicate()
    output = raw.decode("utf-8", "replace").strip()
    last = output.splitlines()[-1] if output else "нет вывода"

    if proc.returncode == 0:
        return Result(True, last)
    if proc.returncode == 5:
        return Result(False, "тесты не найдены")
    if "No module named pytest" in output:
        return Result(False, "pytest не установлен (uv sync --group dev)")
    return Result(False, last, output[-_TAIL:])


async def run_forever(interval_hours: float, report) -> None:
    """Первый прогон сразу, дальше — раз в interval_hours."""
    delay = max(0.0, interval_hours) * 3600
    while True:
        result = await run_once()
        (logger.info if result.ok else logger.error)("самопроверка: %s", result.summary)
        if result.details:
            logger.error("вывод pytest:\n%s", result.details)
        if report is not None:
            await report(result)
        if delay <= 0:
            return
        await asyncio.sleep(delay)
