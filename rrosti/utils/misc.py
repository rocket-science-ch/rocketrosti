# Copyright (c) 2023 Rocket Science AG, Switzerland

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Miscellaneous utility functions and classes."""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Coroutine, Mapping, NoReturn, Protocol, TypeVar, no_type_check

import numpy as np
import numpy.typing as npt
from loguru import logger
from typing_extensions import Self

FloatArray = npt.NDArray[np.float32]

JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | Mapping[JSONPrimitive, "JSONValue"] | list["JSONValue"]


class _ProgramArgsBaseProtocol(Protocol):
    @classmethod
    def parse_args(cls) -> Self:
        """Parse program arguments and return an instance of this class."""
        ...


class ProgramArgsMixinProtocol(Protocol):
    @classmethod
    def _add_args(cls, parser: argparse.ArgumentParser) -> None:
        ...


class _ProgramArgsTypeProtocol(_ProgramArgsBaseProtocol, ProgramArgsMixinProtocol, Protocol):
    pass


_ProgramArgsType = TypeVar("_ProgramArgsType", bound=_ProgramArgsTypeProtocol)


class ProgramArgsBase:
    """
    A base class for parsing program arguments.

    The idea is that your main program can collect arguments from different modules like this:

    class ProgramArgs(ProgramArgsBase, module1.ProgramArgsMixin, module2.ProgramArgsMixin):
        main_program_arg: int

        @classmethod
        def _add_args(cls, parser: argparse.ArgumentParser) -> None:
            super()._add_args(parser)
            parser.add_argument("--main-program-arg", type=int, default=42)

    args = ProgramArgs.parse_args()

    The mixins look like this:

    class ProgramArgsMixin(ProgramArgsMixinProtocol):
        mixin_arg: int

        @classmethod
        def _add_args(cls, parser: argparse.ArgumentParser) -> None:
            parser.add_argument("--mixin-arg", type=int, default=42)
    """

    @classmethod
    def parse_args(cls: type[_ProgramArgsType]) -> _ProgramArgsType:  # noqa: PYI019 (cannot return Self here)
        parser = argparse.ArgumentParser()
        cls._add_args(parser)
        return parser.parse_args(namespace=cls())


class InterceptHandler(logging.Handler):
    @no_type_check
    def emit(self, record: Any) -> None:
        if logger is None:
            # Somehow we get called with this at the end of the program.
            return
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the frame that originated the log message.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str = "DEBUG") -> None:
    logging.basicConfig(handlers=[InterceptHandler()], level=level)


def indent(text: str, prefix: str | int) -> str:
    if isinstance(prefix, int):
        prefix = " " * prefix
    return "\n".join(prefix + line for line in text.splitlines())


def truncate_string(s: str, length: int = 68) -> str:
    """Truncate a string to a maximum length."""
    if len(s) <= length:
        return s
    return s[:length] + "..."


def truncate_json_strings(obj: JSONValue, length: int = 68) -> JSONValue:
    """Truncate all strings in a JSON object to a maximum length."""
    if isinstance(obj, str):
        return truncate_string(obj, length)
    if isinstance(obj, list):
        return [truncate_json_strings(item, length) for item in obj]
    if isinstance(obj, dict):
        return {key: truncate_json_strings(value, length) for key, value in obj.items()}
    return obj


_T = TypeVar("_T")


def file_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def async_once_blocking(
    func: Callable[[], Coroutine[NoReturn, NoReturn, _T]]
) -> Callable[[], Coroutine[NoReturn, NoReturn, _T]]:
    """
    Decorator for a coroutine that should only be run once, and then cached.

    The coroutine is run in the event loop, so it should be fast.
    """

    task: asyncio.Task[_T] | None = None

    @functools.wraps(func)
    async def wrapper() -> _T:
        nonlocal task
        if task is None:
            task = asyncio.create_task(func())
        return await task

    return wrapper


def async_once_in_thread(func: Callable[[], _T]) -> Callable[[], Coroutine[NoReturn, NoReturn, _T]]:
    """
    Decorator for a coroutine that should only be run once, and then cached.

    The coroutine is run in a thread, so it can be slow and/or I/O bound.
    """

    task: asyncio.Task[_T] | None = None

    @functools.wraps(func)
    async def wrapper() -> _T:
        nonlocal task
        if task is None:
            task = asyncio.create_task(asyncio.to_thread(func))
        return await task

    return wrapper


def is_import_in_progress() -> bool:
    return any("importlib._bootstrap" in frame_info.filename for frame_info in inspect.stack())
