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

import io
import os
import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable
from unittest.mock import ANY, _Call, call

import orjson
import overrides
import pytest
import zstandard as zstd
from loguru import logger
from pytest_mock import MockerFixture

import rrosti.query.logging.query_logging as qlog
from tests import testing_utils

FAKE_RUN_UUID = "123"
FAKE_TIME = "20230101_123456"

LOG_DIR = Path("logs")
RUN_LOG_DIR = LOG_DIR / f"{FAKE_TIME}__{FAKE_RUN_UUID}"


FileDict = dict[Path, io.BytesIO]


@pytest.fixture
def query_logging_cleanup(mocker: MockerFixture) -> Iterable[FileDict]:
    """Reset the query_logging module state and mock out the filesystem operations."""

    global qlog

    # Reload the module as a guard in addition to calling _cleanup() (which removes the loguru
    # handlers and is necessary in any case.)
    (qlog,) = testing_utils.reload_modules("rrosti.query.logging.query_logging")

    files: FileDict = {}
    rotation_number: int = 0

    def mock_open_xb(path: Path) -> io.BytesIO:
        assert path not in files, path
        f = io.BytesIO()
        files[path] = f
        return f

    def mock_close(f: BinaryIO) -> None:
        pass

    def mock_exists(path: Path) -> bool:
        return path in files

    def mock_getsize(path: Path) -> int:
        assert path in files, path
        return len(files[path].getbuffer())

    def mock_rename(original_filename: Path, new_filename: Path) -> None:
        assert new_filename not in files, new_filename
        assert original_filename in files, original_filename
        files[new_filename] = files[original_filename]
        del files[original_filename]

    class MockRotatingLog(qlog.RotatingLog):
        rotation_number: int

        def __init__(self, filename: str | Path) -> None:
            super().__init__(filename)
            self.rotation_number = 0

        @overrides.override
        def _get_rotation_filename(self) -> Path:
            nonlocal rotation_number
            rotation_number += 1
            return Path(self.filename_without_ending + "." + str(rotation_number) + self._file_ending)

    mocker.patch.object(qlog, "_SystemAdapter", autospec=True)
    mocker.patch.object(qlog._LoggingEnvironment, "_add_regular_logger_handlers")
    mocker.patch.object(qlog._LoggingEnvironment, "run_uuid", FAKE_RUN_UUID)
    mocker.patch.object(qlog, "_get_timestamp", return_value=FAKE_TIME)

    qlog.RotatingLog = MockRotatingLog  # type: ignore[misc]

    qlog._SystemAdapter.open_xb.side_effect = mock_open_xb  # type: ignore[attr-defined]
    qlog._SystemAdapter.exists.side_effect = mock_exists  # type: ignore[attr-defined]
    qlog._SystemAdapter.getsize.side_effect = mock_getsize  # type: ignore[attr-defined]
    qlog._SystemAdapter.rename.side_effect = mock_rename  # type: ignore[attr-defined]
    qlog._SystemAdapter.close.side_effect = mock_close  # type: ignore[attr-defined]

    yield files
    qlog._cleanup()
    del qlog


@pytest.fixture
def query_logging_cleanup_always_rotate(query_logging_cleanup: MockerFixture, mocker: MockerFixture) -> MockerFixture:
    mocker.patch.object(qlog.RotatingLog, "_RotatingLog__rotation_needed", return_value=True)
    return query_logging_cleanup


def decompress_log_contents(contents: bytes) -> list[bytes]:
    cctx = zstd.ZstdDecompressor()
    decompressed_data = b""
    with cctx.stream_reader(contents) as reader:
        while True:
            chunk = reader.read(1024)
            if not chunk:
                break
            decompressed_data += chunk
    return decompressed_data.splitlines()


def test_no_logging_without_server_start_event(query_logging_cleanup: FileDict) -> None:
    with pytest.raises(qlog.LoggingNotInitializedError):
        qlog.UserInputReceivedEvent.log(text="Hello", uuid=FAKE_RUN_UUID)
    assert not qlog._SystemAdapter.open_xb.called  # type: ignore[attr-defined]
    assert not qlog._SystemAdapter.mkdir.called  # type: ignore[attr-defined]
    assert not qlog._LoggingEnvironment._add_regular_logger_handlers.called  # type: ignore[attr-defined]


def test_logging_with_server_start_event(query_logging_cleanup: FileDict) -> None:
    qlog.ServerStartedEvent.log(args=["--fake"])
    qlog.UserInputReceivedEvent.log(text="Hello", uuid=FAKE_RUN_UUID)

    assert qlog._SystemAdapter.open_xb.called  # type: ignore[attr-defined]
    assert qlog._SystemAdapter.mkdir.called  # type: ignore[attr-defined]
    assert qlog._LoggingEnvironment._add_regular_logger_handlers.called  # type: ignore[attr-defined]

    logger.info(qlog._SystemAdapter.open_xb.call_args_list)  # type: ignore[attr-defined]
    qlog._SystemAdapter.mkdir.assert_called_once_with(RUN_LOG_DIR)  # type: ignore[attr-defined]
    qlog._SystemAdapter.open_xb.assert_called_once_with(RUN_LOG_DIR / "run.log.json.zst")  # type: ignore[attr-defined]

    logf = query_logging_cleanup[RUN_LOG_DIR / "run.log.json.zst"]

    log_contents = logf.getvalue()

    # We could do various checks here
    lines = decompress_log_contents(log_contents)
    assert len(lines) == 2
    assert orjson.loads(lines[0])["rec"]["event"] == "server_started"
    assert orjson.loads(lines[1])["rec"]["event"] == "user_input_received"


def test_no_fs_ops_with_disabled_logging(query_logging_cleanup: FileDict) -> None:
    qlog.init_disabled_logging()
    qlog.UserInputReceivedEvent.log(text="Hello", uuid=FAKE_RUN_UUID)
    # Any opens should be of os.devnull
    assert all(
        args == (Path(os.devnull),)
        for args, _ in qlog._SystemAdapter.open_xb.call_args_list  # type: ignore[attr-defined]
    )
    assert not qlog._SystemAdapter.mkdir.called  # type: ignore[attr-defined]


def test_log_chat_completion_call(query_logging_cleanup: FileDict) -> None:
    qlog.ServerStartedEvent.log(args=["--fake"])
    with qlog.ChatCompletionCallEvent.section(model="fake_model", text=["hello, world"], endpoint="fake_endpoint"):
        pass

    qlog._SystemAdapter.open_xb.assert_called_once_with(RUN_LOG_DIR / "run.log.json.zst")  # type: ignore[attr-defined]
    logf = query_logging_cleanup[RUN_LOG_DIR / "run.log.json.zst"]
    log_contents = logf.getvalue()
    recs = [orjson.loads(line) for line in decompress_log_contents(log_contents)]
    logger.info(pprint.pformat(recs))
    assert len(recs) == 3
    assert recs[0]["rec"]["event"] == "server_started"
    assert recs[1]["rec"] == dict(
        event="chat_completion_call",
        args=dict(endpoint="fake_endpoint", model="fake_model", text="['hello, world']"),
        section=True,
        context=ANY,
    )
    assert recs[2]["rec"] == dict(
        event="end_section", name="chat_completion_call", section=True, context=ANY, exception=None
    )


def test_log_chat_completion_call_with_exception(query_logging_cleanup: FileDict) -> None:
    qlog.ServerStartedEvent.log(args=["--fake"])

    class DummyException(Exception):
        pass

    with pytest.raises(DummyException), qlog.ChatCompletionCallEvent.section(
        model="fake_model", text=["hello, world"], endpoint="fake_endpoint"
    ):
        raise DummyException("oops")

    qlog._SystemAdapter.open_xb.assert_called_once_with(RUN_LOG_DIR / "run.log.json.zst")  # type: ignore[attr-defined]
    logf = query_logging_cleanup[RUN_LOG_DIR / "run.log.json.zst"]
    log_contents = logf.getvalue()
    recs = [orjson.loads(line) for line in decompress_log_contents(log_contents)]
    logger.info(pprint.pformat(recs))
    assert len(recs) == 3, "Did not get all of the expected log lines"
    assert recs[0]["rec"]["event"] == "server_started"
    assert recs[1]["rec"] == dict(
        event="chat_completion_call",
        args=dict(endpoint="fake_endpoint", model="fake_model", text="['hello, world']"),
        section=True,
        context=ANY,
    )
    assert recs[2]["rec"] == dict(
        event="end_section",
        name="chat_completion_call",
        section=True,
        context=ANY,
        exception="DummyException: oops",
    )


class NotPicklableException(Exception):
    def __init__(self) -> None:
        super().__init__("Do not pickle me!")

    def __getstate__(self) -> None:
        raise RuntimeError("Do not pickle me!")


def test_log_chat_completion_call_with_unpicklable_exception(query_logging_cleanup: FileDict) -> None:
    qlog.ServerStartedEvent.log(args=["--fake"])
    with pytest.raises(NotPicklableException), qlog.ChatCompletionCallEvent.section(
        model="fake_model", text=["hello, world"], endpoint="fake_endpoint"
    ):
        raise NotPicklableException

    qlog._SystemAdapter.open_xb.assert_called_once_with(RUN_LOG_DIR / "run.log.json.zst")  # type: ignore[attr-defined]
    logf = query_logging_cleanup[RUN_LOG_DIR / "run.log.json.zst"]
    log_contents = logf.getvalue()
    recs = [orjson.loads(line) for line in decompress_log_contents(log_contents)]
    logger.info(pprint.pformat(recs))
    assert len(recs) == 3, "Did not get all of the expected log lines"
    assert recs[0]["rec"]["event"] == "server_started"
    assert recs[1]["rec"] == dict(
        event="chat_completion_call",
        args=dict(endpoint="fake_endpoint", model="fake_model", text="['hello, world']"),
        section=True,
        context=ANY,
    )
    assert recs[2]["rec"] == dict(
        event="end_section",
        name="chat_completion_call",
        section=True,
        context=ANY,
        exception="NotPicklableException: Do not pickle me!",
    )


# TODO: More of the test logic could be in this class and not duplicated in the tests that use this
@dataclass
class ExpectedLog:
    event_name: str
    message: str
    args: dict[str, Any]
    calls: list[_Call]
    run_log_rows: list[dict[str, Any]]

    def compare_rows(self, rows: list[dict[str, Any]]) -> None:
        recs = [row["rec"] for row in rows]
        for rec, expected_rec in zip(recs, self.run_log_rows):
            assert rec == expected_rec
        assert recs == self.run_log_rows


@pytest.mark.parametrize(
    "expected",
    [
        ExpectedLog(
            event_name="_LogfileSwitchedFromEvent",
            message="switched_from_logfile",
            args={"parent": "parent_log"},
            calls=[
                call(RUN_LOG_DIR / "run.log.json.zst"),
            ],
            run_log_rows=[
                dict(event="server_started", args=ANY, context=ANY, section=False),
                dict(
                    event="switched_from_logfile",
                    args=dict(parent="parent_log"),
                    section=False,
                    context=ANY,
                ),
            ],
        ),
        ExpectedLog(
            event_name="UserInputReceivedEvent",
            message="user_input_received",
            args={"uuid": "007", "text": "user_input"},
            calls=[
                call(RUN_LOG_DIR / "run.log.json.zst"),
            ],
            run_log_rows=[
                dict(event="server_started", args=ANY, context=ANY, section=False),
                dict(
                    event="user_input_received",
                    args=dict(uuid="007", text="user_input"),
                    section=False,
                    context=ANY,
                ),
            ],
        ),
        ExpectedLog(
            event_name="SendMessageToFrontendEvent",
            message="send_message_frontend",
            args={"uuid": "007", "msg": "ai_message", "time_used": 0.5, "cost": 0.15},
            calls=[
                call(RUN_LOG_DIR / "run.log.json.zst"),
            ],
            run_log_rows=[
                dict(event="server_started", args=ANY, context=ANY, section=False),
                dict(
                    event="send_message_frontend",
                    args=dict(uuid="007", msg="ai_message", time_used=0.5, cost=0.15),
                    section=False,
                    context=ANY,
                ),
            ],
        ),
        ExpectedLog(
            event_name="ReceivedMessageFromFrontendEvent",
            message="received_message_frontend",
            args={"uuid": "007", "msg": "user_message"},
            calls=[
                call(RUN_LOG_DIR / "run.log.json.zst"),
            ],
            run_log_rows=[
                dict(event="server_started", args=ANY, context=ANY, section=False),
                dict(
                    event="received_message_frontend",
                    args=dict(uuid="007", msg="user_message"),
                    section=False,
                    context=ANY,
                ),
            ],
        ),
        ExpectedLog(
            event_name="UserLikedMessageEvent",
            message="user_liked_message",
            args={"uuid": "007", "text": None, "ref_uuid": "008"},
            calls=[
                call(RUN_LOG_DIR / "run.log.json.zst"),
            ],
            run_log_rows=[
                dict(event="server_started", args=ANY, context=ANY, section=False),
                dict(
                    event="user_liked_message",
                    args=dict(uuid="007", text=None, ref_uuid="008"),
                    section=False,
                    context=ANY,
                ),
            ],
        ),
        ExpectedLog(
            event_name="UserDislikedMessageEvent",
            message="user_disliked_message",
            args={"uuid": "007", "text": None, "ref_uuid": "008"},
            calls=[
                call(RUN_LOG_DIR / "run.log.json.zst"),
            ],
            run_log_rows=[
                dict(event="server_started", args=ANY, context=ANY, section=False),
                dict(
                    event="user_disliked_message",
                    args=dict(uuid="007", text=None, ref_uuid="008"),
                    section=False,
                    context=ANY,
                ),
            ],
        ),
    ],
)
def test_log_events(query_logging_cleanup: FileDict, expected: ExpectedLog) -> None:
    qlog.ServerStartedEvent.log(args=["--fake"])
    getattr(qlog, expected.event_name).log(**expected.args)

    assert qlog._SystemAdapter.open_xb.call_args_list == expected.calls  # type: ignore[attr-defined]
    logf = query_logging_cleanup[RUN_LOG_DIR / "run.log.json.zst"]
    log_contents = logf.getvalue()
    recs = [orjson.loads(line) for line in decompress_log_contents(log_contents)]
    logger.info(pprint.pformat(recs))
    expected.compare_rows(recs)


@pytest.mark.parametrize(
    "expected",
    [
        ExpectedLog(
            event_name="ConnectionEvent",
            message="connection",
            args={"id": "007", "local_addr": "127.0.0.1", "remote_addr": "127.0.0.1"},
            calls=[
                call(RUN_LOG_DIR / "run.log.json.zst"),
            ],
            run_log_rows=[
                dict(event="server_started", args=ANY, context=ANY, section=False),
                dict(
                    event="connection",
                    args=dict(
                        id="007",
                        local_addr="127.0.0.1",
                        remote_addr="127.0.0.1",
                    ),
                    section=True,
                    context=ANY,
                ),
                dict(event="end_section", name="connection", section=True, context=ANY, exception=None),
            ],
        ),
        ExpectedLog(
            event_name="QueryEvent",
            message="query",
            args={"text": "text", "uuid": "007", "prompt": "FAKE_PROMPT"},
            calls=[
                call(RUN_LOG_DIR / "run.log.json.zst"),
                call(RUN_LOG_DIR / "queries" / f"{FAKE_TIME}__007.log.json.zst"),
            ],
            run_log_rows=[
                dict(event="server_started", args=ANY, context=ANY, section=False),
                dict(
                    event="query",
                    args=dict(text="text", uuid="007", prompt="FAKE_PROMPT"),
                    section=True,
                    context=ANY,
                ),
                dict(
                    event="switch_to_logfile",
                    args=dict(logfile=str(Path("logs/20230101_123456__123/queries/20230101_123456__007"))),
                    section=True,
                    context=ANY,
                ),
                dict(event="end_section", name="switch_to_logfile", section=True, context=ANY, exception=None),
                dict(event="end_section", name="query", section=True, context=ANY, exception=None),
            ],
        ),
        ExpectedLog(
            event_name="ChatCompletionCallEvent",
            message="chat_completion_call",
            args={"model": "gpt4", "text": "text", "endpoint": "OPENAI_API_BASE"},
            calls=[
                call(RUN_LOG_DIR / "run.log.json.zst"),
            ],
            run_log_rows=[
                dict(event="server_started", args=ANY, context=ANY, section=False),
                dict(
                    event="chat_completion_call",
                    args=dict(model="gpt4", text="text", endpoint="OPENAI_API_BASE"),
                    section=True,
                    context=ANY,
                ),
                dict(event="end_section", name="chat_completion_call", section=True, context=ANY, exception=None),
            ],
        ),
        ExpectedLog(
            event_name="_LogfileSwitchEvent",
            message="switch_to_logfile",
            args={"logfile": "./query/query.log.json.zst"},
            calls=[
                call(RUN_LOG_DIR / "run.log.json.zst"),
            ],
            run_log_rows=[
                dict(event="server_started", args=ANY, context=ANY, section=False),
                dict(
                    event="switch_to_logfile",
                    args=dict(logfile="./query/query.log.json.zst"),
                    section=True,
                    context=ANY,
                ),
                dict(event="end_section", name="switch_to_logfile", section=True, context=ANY, exception=None),
            ],
        ),
        ExpectedLog(
            event_name="WaitForUserInputEvent",
            message="wait_for_user_input",
            args={},
            calls=[
                call(RUN_LOG_DIR / "run.log.json.zst"),
            ],
            run_log_rows=[
                dict(event="server_started", args=ANY, context=ANY, section=False),
                dict(event="wait_for_user_input", args={}, section=True, context=ANY),
                dict(event="end_section", name="wait_for_user_input", section=True, context=ANY, exception=None),
            ],
        ),
    ],
)
def test_log_section_events(
    query_logging_cleanup: FileDict,
    expected: ExpectedLog,
) -> None:
    # TODO: Also check the query log file
    qlog.ServerStartedEvent.log(args=["--fake"])
    with getattr(qlog, expected.event_name).section(**expected.args):
        pass

    assert qlog._SystemAdapter.open_xb.call_args_list == expected.calls  # type: ignore[attr-defined]
    logf = query_logging_cleanup[RUN_LOG_DIR / "run.log.json.zst"]
    log_contents = logf.getvalue()
    rows = [orjson.loads(line) for line in decompress_log_contents(log_contents)]
    logger.info(pprint.pformat(rows))
    expected.compare_rows(rows)


@pytest.mark.usefixtures("query_logging_cleanup_always_rotate")
def test_log_rotation(query_logging_cleanup_always_rotate: FileDict) -> None:
    file_ending = ".log.json.zst"
    expected_result_file_dict: dict[Path, Any] = {
        RUN_LOG_DIR / ("run.1" + file_ending): [],
        RUN_LOG_DIR / ("run.2" + file_ending): {"event": "server_started"},
        RUN_LOG_DIR / ("run.3" + file_ending): {"event": "user_input_received", "args": {"text": "Hello the first"}},
        RUN_LOG_DIR / ("run" + file_ending): {"event": "user_input_received", "args": {"text": "Hello the second"}},
    }

    qlog.ServerStartedEvent.log(args=["--fake"])
    qlog.UserInputReceivedEvent.log(text="Hello the first", uuid=FAKE_RUN_UUID)
    qlog.UserInputReceivedEvent.log(text="Hello the second", uuid=FAKE_RUN_UUID)

    for key, value in expected_result_file_dict.items():
        assert key in query_logging_cleanup_always_rotate
        logf = query_logging_cleanup_always_rotate[key]
        log_contents = logf.getvalue()
        recs = [orjson.loads(line) for line in decompress_log_contents(log_contents)]
        if value:
            assert value["event"] == recs[0]["rec"]["event"]
            if recs[0]["rec"]["event"] == "user_input_received":
                assert value["args"]["text"] == recs[0]["rec"]["args"]["text"]
        else:
            assert recs == []
