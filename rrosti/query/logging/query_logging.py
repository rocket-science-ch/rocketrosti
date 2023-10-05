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

"""
Log queries and other pertinent information to a file.

The general loguru logger is used for general application logging, in the interests of software
development and debugging, often in a very ad hoc fashion.

The query logger, instead, is used for more planned and carefully specified logging of queries,
LLM executions, the times they took, user feedback (thumbs up/down), and so on.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import os
import platform
import signal
import sys
import threading
import uuid
from abc import ABC, abstractmethod
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import Any, BinaryIO, Callable, Iterator

import loguru
import orjson
import zstandard as zstd
from overrides import override
from typing_extensions import Self

# This is used just as a hint to make sure users don't instantiate some classes.
_PRIVATE_OBJECT = object()

# This is a silly chicken and egg problem. We globally start logging before we have read the config, so we cannot
# really use the config here.
_ROTATION_SIZE_MEGABYTES = 250


class _SystemAdapter:
    """Wrap some filesystem operations that we need to mock in tests."""

    @staticmethod
    def mkdir(path: Path) -> None:
        """Make a directory."""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def open_xb(path: Path) -> BinaryIO:
        """Open a file."""
        if path == Path(os.devnull):
            return open(path, "wb")  # noqa: SIM115 - cannot use context handler here
        return open(path, "xb")  # noqa: SIM115 - cannot use context handler here

    @staticmethod
    def close(f: BinaryIO) -> None:
        f.close()

    @staticmethod
    def rename(original_filename: Path, new_filename: Path) -> None:
        """Rename a File"""
        original_filename.rename(new_filename)

    @staticmethod
    def exists(path: Path) -> bool:
        """Check if a given file exists"""
        return os.path.exists(path)

    @staticmethod
    def getsize(path: Path) -> int:
        """Get the Size of a given File in bytes"""
        return os.path.getsize(path)


def _get_timestamp() -> str:
    """Get a timestamp for the current time in the format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class LoggingNotInitializedError(Exception):
    """Raised when logging is used before it's initialized."""


class _LoggingEnvironment:
    """
    Base class representing a logging environment. This one logs to /dev/null and can thus also
    be used directly in tests.

    There is (at most) one per execution. It contains things like the run UUID and the log file.

    For server runs, this is initialized by logging a ServerStartedEvent. For test and script runs,
    something else should usually be done in order to not pollute the server logs.
    """

    run_uuid = str(uuid.uuid4())
    _query_logger: loguru.Logger

    _log_file: RotatingLog
    _run_log_fname_stem: Path  # This Filename should not include the ending ".log.json.zst"

    LOG_PATH: str = os.devnull  # override in subclasses

    def _make_query_filename(cls, query_uuid: str) -> Path:
        """Make a query log file name."""
        if cls._run_log_fname_stem == Path(os.devnull):
            return Path(os.devnull)
        return cls._run_log_fname_stem.parent / "queries" / f"{_get_timestamp()}__{query_uuid}"

    @property
    def log_file(self) -> RotatingLog:
        """Get the log file."""
        return self._log_file

    @classmethod
    def _add_regular_logger_handlers(cls) -> None:
        """Add regular logger handlers."""

        Path(cls.LOG_PATH).mkdir(parents=True, exist_ok=True)

        rotation_size = f"{_ROTATION_SIZE_MEGABYTES} MB"

        # Log the regular logs as JSON...
        loguru.logger.add(
            Path(cls.LOG_PATH) / "debug.log.json",
            level="DEBUG",
            serialize=True,
            enqueue=True,
            rotation=rotation_size,
            compression="gz",
        )

        # ... and, for convenience, also as text.
        loguru.logger.add(
            Path(cls.LOG_PATH) / "debug.log.txt",
            level="DEBUG",
            enqueue=True,
            rotation=rotation_size,
            compression="gz",
        )

    def __init__(self) -> None:
        """Initialize the logging environment."""

        loguru.logger.info("Starting logging to '{}'", self.LOG_PATH)

        if self.LOG_PATH != os.devnull:
            loguru.logger.remove()
            loguru.logger.add(sys.stderr, level=os.environ.get("LOGURU_LEVEL", "INFO"))
            self._add_regular_logger_handlers()

            self._run_log_fname_stem = (
                Path(self.LOG_PATH) / f"{_get_timestamp()}__{_LoggingEnvironment.run_uuid}" / "run"
            )

            _SystemAdapter.mkdir(self._run_log_fname_stem.parent)
        else:
            self._run_log_fname_stem = Path(os.devnull)

        self._query_logger = loguru.logger.bind(logger="query_logger")
        self._query_logger.add(_handler, filter=lambda r: r["message"] == _RawQueryLogEntry._MAGIC)

        self._log_file = RotatingLog(self._run_log_fname_stem)


class _ServerLoggingEnvironment(_LoggingEnvironment):
    """A logging environment for server runs."""

    LOG_PATH = "logs"


_logging_environment: _LoggingEnvironment | None = None
_query_logger_context: ContextVar[_Context | None] = ContextVar("query_logger_context", default=None)


def _cleanup() -> None:
    """Reset the global state of the module. Used in test fixtures."""

    global _logging_environment
    global _query_logger_context
    _logging_environment = None
    _query_logger_context = ContextVar("query_logger_context", default=None)
    loguru.logger.remove()
    loguru.logger.add(sys.stderr, level=os.environ.get("LOGURU_LEVEL", "INFO"))


def _env() -> _LoggingEnvironment:
    """Get the logging environment."""
    if _logging_environment is None:
        raise LoggingNotInitializedError(
            "Logging environment not initialized by ServerStartedEvent (for servers) or similar"
        )
    return _logging_environment


@dataclass
class _Context:
    """
    A context for logging. This is used to keep track of the current logfile and sections we're in, in a stack-like
    fashion.
    """

    logfile: RotatingLog
    vars: dict[str, Any]
    outer: _Context | None = None
    depth: int = field(init=False)
    _old_context: _Context | None = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self.depth = 0 if self.outer is None else self.outer.depth + 1

    def get(self, key: str, default: Any = None) -> Any:
        if not isinstance(key, str):
            raise TypeError
        """Get a context variable."""
        if key in self.vars:
            return self.vars[key]
        if self.outer is not None:
            return self.outer.get(key, default)
        return default

    def as_dict(self) -> dict[str, Any]:
        """Convert the context to a dictionary."""
        return dict(depth=self.depth, vars=self.vars, outer=self.outer.as_dict() if self.outer else None)

    def nest(self, logfile: RotatingLog | None = None, **kwargs: Any) -> _Context:
        """Nest a new context."""
        if logfile is None:
            logfile = self.logfile
        return _Context(logfile=logfile, vars=kwargs, outer=self)

    def __enter__(self) -> Self:
        """Enter the context."""
        self._old_context = _query_logger_context.get()
        _query_logger_context.set(self)
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        """Exit the context."""
        _query_logger_context.set(self._old_context)


def _get_query_logger_context() -> _Context:
    """
    Get the current query logger context.

    For creating new contexts, prefer _context().
    """
    context = _query_logger_context.get()
    if context is None:
        context = _Context(logfile=_env().log_file, vars={})
        _query_logger_context.set(context)
    return context


@contextlib.contextmanager
def _context(**kwargs: Any) -> Iterator[_Context]:
    """
    Add contextual information to the query log.

    When creating a new context, this context manager takes care of the cleanup of the old context.
    """
    outer_context = _get_query_logger_context()
    with outer_context.nest(**kwargs) as inner_context:
        yield inner_context


@dataclass(frozen=True)
class _Metadata:
    """Metadata about a query log entry."""

    level: str
    time: datetime
    elapsed: timedelta
    run_uuid: str = _LoggingEnvironment.run_uuid

    @staticmethod
    def from_loguru_record(record: loguru.Record) -> _Metadata:
        assert isinstance(record["time"], datetime)
        assert isinstance(record["elapsed"], timedelta)
        return _Metadata(
            elapsed=record["elapsed"],
            level=record["level"].name,
            time=record["time"],
        )


@dataclass(frozen=True)
class _RawQueryLogEntry:
    """A single entry in the query log, encompassing metadata and log data."""

    rec: dict[str, Any]
    meta: _Metadata
    _MAGIC = "$$query_log_entry"

    @staticmethod
    def from_loguru_record(record: loguru.Record) -> _RawQueryLogEntry:
        if record["message"] != _RawQueryLogEntry._MAGIC:
            raise ValueError(f"Invalid log entry: {record}")
        extra = copy.copy(record["extra"])
        if "logger" in extra and extra["logger"] == "query_logger":
            del extra["logger"]
        return _RawQueryLogEntry(
            meta=_Metadata.from_loguru_record(record),
            rec=extra,
        )

    def as_dict(self) -> dict[str, Any]:
        """Convert the query log entry to a dictionary."""
        return dict(
            rec=self.rec,
            meta=dataclasses.asdict(self.meta),
        )


def _to_json(obj: Any) -> str | float:
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Type is not JSON serializable: {type(obj)}")


class RotatingLog:
    _file: BinaryIO
    _compressor: zstd.ZstdCompressionWriter
    filename: str
    filename_without_ending: str
    _file_ending: str
    _file_lock: threading.Lock

    _original_sigterm_handler: Callable[[int, FrameType | None], Any] | int | None
    _original_sigint_handler: Callable[[int, FrameType | None], Any] | int | None

    @property
    def max_size(self) -> int:
        return _ROTATION_SIZE_MEGABYTES * 1_000_000

    def __init__(self, filename: str | Path) -> None:
        if Path(filename) == Path(os.devnull):
            self._file_ending = ""
        else:
            self._file_ending = ".log.json.zst"
        self.filename_without_ending = str(filename)
        self.filename = self.filename_without_ending + self._file_ending
        self._open()
        self._file_lock = threading.Lock()

        self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)

        # Register signal handler for SIGTERM and SIGINT (Ctrl+C)
        signal.signal(signal.SIGTERM, self._handle_exit)
        signal.signal(signal.SIGINT, self._handle_exit)

    def __del__(self) -> None:
        # Close File (and flush compressor) if not already closed
        with suppress(ValueError):
            self._close()

    def _handle_exit(self, signum: int, frame: FrameType | None) -> None:
        # Close File (and flush compressor) if not already closed
        with suppress(ValueError):
            self._close()
        # Call the original signal handler to exit gracefully
        original_handler = self._original_sigterm_handler if signum == signal.SIGTERM else self._original_sigint_handler
        if original_handler is None:
            return
        if isinstance(original_handler, int):
            assert original_handler == signal.SIG_DFL or original_handler == signal.SIG_IGN, original_handler
            return
        original_handler(signum, frame)

    def _get_rotation_filename(self) -> Path:
        formatted_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        return Path(self.filename_without_ending + "." + formatted_datetime + self._file_ending)

    def __rotation_needed(self) -> bool:
        return (
            _SystemAdapter.exists(Path(self.filename)) and _SystemAdapter.getsize(Path(self.filename)) >= self.max_size
        )

    def __rotate_if_needed(self) -> None:
        if self.__rotation_needed():
            self._close()
            _SystemAdapter.rename(Path(self.filename), self._get_rotation_filename())
            self._open()

    def write(self, msg: "loguru.Message") -> None:  # noqa: UP037  (quotes necessary because loguru doesn't export)
        with self._file_lock:
            try:
                self.__rotate_if_needed()
                self._compressor.write(
                    orjson.dumps(_RawQueryLogEntry.from_loguru_record(msg.record).as_dict(), default=_to_json)
                )
                self._compressor.write(b"\n")
                self._compressor.flush(
                    zstd.FLUSH_FRAME
                )  # TODO: This is costly, so for the future we should try to move all the logging to a seperate thread
            except ValueError:
                loguru.logger.error(f"Trying to write to closed file {self.filename}.")

    def _open(self) -> None:
        self._file = _SystemAdapter.open_xb(Path(self.filename))
        self._compressor = zstd.ZstdCompressor().stream_writer(self._file)

    def _close(self) -> None:
        self._compressor.flush(zstd.FLUSH_FRAME)
        _SystemAdapter.close(self._file)


def _handler(msg: "loguru.Message") -> None:  # noqa: UP037  (quotes necessary because loguru doesn't export)
    """
    Handle the processing and writing of a log message to a log file.

    Converts the log message to JSON and writes it to the appropriate logfile.
    """
    _get_query_logger_context().logfile.write(msg)


def _get_platform_info() -> dict[str, Any]:
    """Get information about the current platform."""
    return dict(
        hostname=platform.node(),
        system=platform.system(),
        platform=platform.platform(),
        python_buildno=platform.python_build()[0],
        python_builddate=platform.python_build()[1],
        processor=platform.processor(),
        python_version=platform.python_version(),
        python_path=sys.executable,
    )


@contextlib.contextmanager
def _log_section(event: _LogEvent) -> Iterator[None]:
    """Add contextual information to the query log."""
    event_dict = event.to_dict()
    event_name = event_dict.pop("event")
    with _context(event=event_name, args=event_dict) as inner_context:
        _env()._query_logger.bind(
            event=event_name, args=event_dict, section=True, context=inner_context.as_dict()
        ).info(_RawQueryLogEntry._MAGIC)

        exception: str | None = None
        try:
            yield
        except Exception as e:
            try:
                exception = f"{type(e).__name__}: {e}"
            except Exception:
                exception = f"Exception of type {type(e)} that cannot be converted to string"
            raise
        finally:
            _env()._query_logger.bind(
                event="end_section", name=event_name, section=True, context=inner_context.as_dict(), exception=exception
            ).info(_RawQueryLogEntry._MAGIC)


class _LogEvent(ABC):
    """
    An abstract base class representing a log event.
    Classes derived from this class represent specific kinds of log

    There are two ways to use the events:

    1. query_logging.ServerStartedEvent.log(args=sys.argv)
    2. with query_logging.ServerStartedEvent.section(args=sys.argv): ...

    Most subclasses only support one or the other.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary."""


def _log(event: _LogEvent) -> None:
    """Log a query log entry."""
    d = event.to_dict()
    event_name = d.pop("event")

    _env()._query_logger.bind(
        event=event_name, args=d, section=False, context=_get_query_logger_context().as_dict()
    ).info(_RawQueryLogEntry._MAGIC)


@contextlib.contextmanager
def _switch_logfile(logfile: Path) -> Iterator[None]:
    """Switch to a new logfile."""
    with _LogfileSwitchEvent.section(logfile):
        old_context = _get_query_logger_context()
        old_logfile = old_context.logfile
        assert old_logfile is not None
        _SystemAdapter.mkdir(logfile.parent)
        with _context(logfile=RotatingLog(logfile), logfile_name=logfile):
            _LogfileSwitchedFromEvent.log(parent=str(old_logfile.filename))
            yield


## These are the only public API of this module.


def init_disabled_logging() -> None:
    """Initialize logging for a test run."""
    global _logging_environment
    _logging_environment = _LoggingEnvironment()


class ServerStartedEvent(_LogEvent):
    """A server started event."""

    args: list[str]

    def __init__(self, private_object: object, args: list[str]) -> None:
        """Private constructor. Use the log() method instead."""
        assert private_object is _PRIVATE_OBJECT, "Cannot instantiate ServerStartedEvent"
        self.args = args

    # We need to repeat to_dict and log in every subclass, sadly, to get mypy and type hinting
    # in IDEs to work well.
    @override
    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary."""
        return dict(event="server_started", args=self.args, platform=_get_platform_info())

    @staticmethod
    def log(args: list[str]) -> None:
        """Log the event. Also initializes the logging environment."""

        global _logging_environment
        if _logging_environment is None:
            _logging_environment = _ServerLoggingEnvironment()
        else:
            raise RuntimeError("ServerStartedEvent already logged. There should be only one.")

        _log(ServerStartedEvent(_PRIVATE_OBJECT, args=args))


class ConnectionEvent(_LogEvent):
    """A connection event."""

    id: str
    local_addr: str
    remote_addr: str

    def __init__(self, private_object: object, id: str, local_addr: str, remote_addr: str) -> None:
        """Private constructor. Use the section() method instead."""
        assert private_object is _PRIVATE_OBJECT, "Cannot instantiate ConnectionEvent"
        self.id = id
        self.local_addr = local_addr
        self.remote_addr = remote_addr

    @override
    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary."""
        return dict(event="connection", id=self.id, local_addr=self.local_addr, remote_addr=self.remote_addr)

    @staticmethod
    @contextlib.contextmanager
    def section(id: str, local_addr: str, remote_addr: str) -> Iterator[None]:
        with _log_section(ConnectionEvent(_PRIVATE_OBJECT, id=id, local_addr=local_addr, remote_addr=remote_addr)):
            yield


class QueryEvent(_LogEvent):
    """A query event."""

    text: str
    uuid: str
    prompt: str

    def __init__(self, private_object: object, text: str, uuid: str, prompt: str) -> None:
        """Private constructor. Use the section() method instead."""
        assert private_object is _PRIVATE_OBJECT, "Cannot instantiate QueryEvent"
        self.text = text
        self.uuid = uuid
        self.prompt = prompt

    @override
    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary."""
        return dict(event="query", text=self.text, uuid=self.uuid, prompt=self.prompt)

    @staticmethod
    @contextlib.contextmanager
    def section(text: str, uuid: str, prompt: str) -> Iterator[None]:
        query_fname = _env()._make_query_filename(uuid)
        with _log_section(QueryEvent(_PRIVATE_OBJECT, text=text, uuid=uuid, prompt=prompt)), _switch_logfile(
            query_fname
        ):
            yield


class ChatCompletionCallEvent(_LogEvent):
    """An LLM call event."""

    model: str
    text: str
    endpoint: str  # OPENAI_API_BASE or similar

    def __init__(self, private_object: object, model: str, text: str, endpoint: str) -> None:
        """Private constructor. Use the section() method instead."""
        assert private_object is _PRIVATE_OBJECT, "Cannot instantiate ChatCompletionCallEvent"
        self.model = model
        self.text = text
        self.endpoint = endpoint

    @override
    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary."""
        return dict(event="chat_completion_call", model=self.model, text=self.text, endpoint=self.endpoint)

    @staticmethod
    @contextlib.contextmanager
    def section(model: str, text: list[Any], endpoint: str) -> Iterator[None]:
        with _log_section(ChatCompletionCallEvent(_PRIVATE_OBJECT, model=model, text=str(text), endpoint=endpoint)):
            yield


class _LogfileSwitchEvent(_LogEvent):
    """A logfile switch event."""

    logfile: Path

    def __init__(self, private_object: object, logfile: Path) -> None:
        """Private constructor. Use the section() method instead."""
        assert private_object is _PRIVATE_OBJECT, "Cannot instantiate LogfileSwitchEvent"
        self.logfile = logfile

    @override
    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary."""
        return dict(event="switch_to_logfile", logfile=str(self.logfile))

    @staticmethod
    @contextlib.contextmanager
    def section(logfile: Path) -> Iterator[None]:
        with _log_section(_LogfileSwitchEvent(_PRIVATE_OBJECT, logfile=logfile)):
            yield


class _LogfileSwitchedFromEvent(_LogEvent):
    """A logfile switch event."""

    parent: str

    def __init__(self, private_object: object, parent: str) -> None:
        """Private constructor. Use the log() method instead."""
        assert private_object is _PRIVATE_OBJECT, "Cannot instantiate LogfileSwitchedFromEvent"
        self.parent = parent

    @override
    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary."""
        return dict(event="switched_from_logfile", parent=self.parent)

    @staticmethod
    def log(parent: str) -> None:
        """Log the event."""
        _log(_LogfileSwitchedFromEvent(_PRIVATE_OBJECT, parent=parent))


class WaitForUserInputEvent(_LogEvent):
    """A wait for user input event."""

    def __init__(self, private_object: object) -> None:
        """Private constructor. Use the section() method instead."""
        assert private_object is _PRIVATE_OBJECT, "Cannot instantiate WaitForUserInputEvent"

    @override
    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary."""
        return dict(event="wait_for_user_input")

    @staticmethod
    @contextlib.contextmanager
    def section() -> Iterator[None]:
        with _log_section(WaitForUserInputEvent(_PRIVATE_OBJECT)):
            yield


class UserInputReceivedEvent(_LogEvent):
    """A user input event."""

    text: str
    uuid: str

    def __init__(self, private_object: object, text: str, uuid: str) -> None:
        """Private constructor. Use the log() method instead."""
        assert private_object is _PRIVATE_OBJECT, "Cannot instantiate UserInputReceivedEvent"
        self.text = text
        self.uuid = uuid

    @override
    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary."""
        return dict(event="user_input_received", text=self.text, uuid=self.uuid)

    @staticmethod
    def log(text: str, uuid: str) -> None:
        """Log the event."""
        _log(UserInputReceivedEvent(_PRIVATE_OBJECT, text=text, uuid=uuid))


class SendMessageToFrontendEvent(_LogEvent):
    """A message sent event."""

    uuid: str
    msg: str
    time: float | None
    cost: float | None

    def __init__(self, private_object: object, uuid: str, msg: str, time: float | None, cost: float | None) -> None:
        """Private constructor. Use the log() method instead."""
        assert private_object is _PRIVATE_OBJECT, "Cannot instantiate SendMessageToFrontendEvent"
        self.uuid = uuid
        self.msg = msg
        self.time = time
        self.cost = cost

    @override
    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary."""
        return dict(event="send_message_frontend", uuid=self.uuid, msg=self.msg, time_used=self.time, cost=self.cost)

    @staticmethod
    def log(uuid: str, msg: str, time_used: float | None, cost: float | None) -> None:
        """Log the event."""
        _log(SendMessageToFrontendEvent(_PRIVATE_OBJECT, uuid=uuid, msg=msg, time=time_used, cost=cost))


class ReceivedMessageFromFrontendEvent(_LogEvent):
    """A message received event."""

    uuid: str
    msg: str

    def __init__(self, private_object: object, uuid: str, msg: str) -> None:
        """Private constructor. Use the log() method instead."""
        assert private_object is _PRIVATE_OBJECT, "Cannot instantiate ReceivedMessageFromFrontendEvent"
        self.uuid = uuid
        self.msg = msg

    @override
    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary."""
        return dict(event="received_message_frontend", uuid=self.uuid, msg=self.msg)

    @staticmethod
    def log(uuid: str, msg: str) -> None:
        """Log the event."""
        _log(ReceivedMessageFromFrontendEvent(_PRIVATE_OBJECT, uuid=uuid, msg=msg))


class UserLikedMessageEvent(_LogEvent):
    """A user liked message event."""

    uuid: str
    text: str | None
    ref_uuid: str

    def __init__(self, private_object: object, uuid: str, text: str | None, ref_uuid: str) -> None:
        """Private constructor. Use the log() method instead."""
        assert private_object is _PRIVATE_OBJECT, "Cannot instantiate UserLikedMessageEvent"
        self.uuid = uuid
        self.text = text
        self.ref_uuid = ref_uuid

    @override
    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary."""
        return dict(event="user_liked_message", uuid=self.uuid, text=self.text, ref_uuid=self.ref_uuid)

    @staticmethod
    def log(uuid: str, text: str | None, ref_uuid: str) -> None:
        """Log the event."""
        _log(UserLikedMessageEvent(_PRIVATE_OBJECT, uuid=uuid, text=text, ref_uuid=ref_uuid))


class UserDislikedMessageEvent(_LogEvent):
    """A user disliked message event."""

    uuid: str
    text: str | None
    ref_uuid: str

    def __init__(self, private_object: object, uuid: str, text: str | None, ref_uuid: str) -> None:
        """Private constructor. Use the log() method instead."""
        assert private_object is _PRIVATE_OBJECT, "Cannot instantiate UserDislikedMessageEvent"
        self.uuid = uuid
        self.text = text
        self.ref_uuid = ref_uuid

    @override
    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a dictionary."""
        return dict(event="user_disliked_message", uuid=self.uuid, text=self.text, ref_uuid=self.ref_uuid)

    @staticmethod
    def log(uuid: str, text: str | None, ref_uuid: str) -> None:
        """Log the event."""
        _log(UserDislikedMessageEvent(_PRIVATE_OBJECT, uuid=uuid, text=text, ref_uuid=ref_uuid))
