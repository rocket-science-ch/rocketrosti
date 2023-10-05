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

"""Implementation of a websocket query server."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import websockets
from loguru import logger
from overrides import override

from rrosti.chat.chat_session import Message
from rrosti.query import logging as qlog
from rrosti.snippets.snippet import Snippet
from rrosti.utils.config import config
from rrosti.utils.misc import ProgramArgsMixinProtocol

_FAKE_DELAY = 1.5

_ALLOWED_CHARACTERS_RE = re.compile(
    r"[^a-zA-ZäÄëËïÏöÖüÜâÂêÊîÎôÔûÛéÉèÈàÀùÙñÑßçÇ0-9" + re.escape(r",.?!\-_: ();<>=+*°") + "]"
)


class ProgramArgsMixin(ProgramArgsMixinProtocol):
    port: int
    public: bool
    debug_send_intermediates: bool
    debug_asyncio: bool

    @classmethod
    def _add_args(cls, parser: argparse.ArgumentParser) -> None:
        super()._add_args(parser)
        parser.add_argument("--port", type=int, default=config.backend.listen_port, help="Port to listen on")
        parser.add_argument("--public", action="store_true", help="Listen on all interfaces")
        parser.add_argument("--debug-asyncio", action="store_true", help="Debug coroutine execution")
        parser.add_argument(
            "--debug-send-intermediates",
            action="store_true",
            help="Send intermediate messages and more detailed output to the frontend",
        )


@dataclass(frozen=True)
class UserInputMessage:
    content: str
    uuid: str


class MessageType(Enum):
    question = auto()
    like = auto()
    dislike = auto()
    new_question = auto()
    unknown = auto()


class UnknownMessageType(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class Frontend(ABC):
    _peek_buffer: UserInputMessage | None = None

    @abstractmethod
    async def _get_user_input_impl(self) -> UserInputMessage:
        ...

    async def get_user_input(self) -> UserInputMessage:
        msg = await self.peek_user_input()
        self._peek_buffer = None
        return msg

    async def peek_user_input(self) -> UserInputMessage:
        if self._peek_buffer is None:
            self._peek_buffer = await self._get_user_input_impl()
        return self._peek_buffer

    @abstractmethod
    async def send_message(self, msg: Message | str) -> None:
        ...

    @abstractmethod
    async def read_message(self, message: str) -> MessageType:
        ...

    @abstractmethod
    def handle_python_output(self, python_output: PythonItem) -> None:
        ...

    @abstractmethod
    def handle_rtfm_output(self, snippets: list[Snippet]) -> int:
        ...


class WebFrontend(Frontend):
    __user_message_queue: asyncio.Queue[UserInputMessage]
    answer: Answer
    __debug_send_intermediates: bool
    __websocket: websockets.WebSocketServerProtocol  # type: ignore[name-defined]

    def __init__(
        self,
        websocket: websockets.WebSocketServerProtocol,  # type: ignore[name-defined]
        debug_send_intermediates: bool = False,
    ) -> None:
        self.__websocket = websocket
        self.__debug_send_intermediates = debug_send_intermediates
        self.__user_message_queue = asyncio.Queue()
        self.answer = Answer()

    @staticmethod
    def _sanitize_str(s: str) -> str:
        return _ALLOWED_CHARACTERS_RE.sub("", s)

    @override
    async def _get_user_input_impl(self) -> UserInputMessage:
        with qlog.WaitForUserInputEvent.section():
            if not self.answer.is_empty():
                await self.__send_msg(
                    dict(
                        id=str(uuid.uuid4()),
                        type="answer",
                        content=self.answer.to_message(self.__debug_send_intermediates),
                    ),
                    time=None,
                    cost=None,
                )
            await self.__send_msg(dict(id=str(uuid.uuid4()), type="user_turn"), time=None, cost=None)
            reply = await self.__user_message_queue.get()
            qlog.UserInputReceivedEvent.log(text=reply.content, uuid=reply.uuid)
        return reply

    @override
    async def send_message(self, msg: Message | str) -> None:
        self.answer.append_answer(msg)
        if isinstance(msg, Message):
            await self.__send_msg(
                dict(id=str(uuid.uuid4()), type="intermediate", content=msg.text.replace("\n", "<br>")),
                time=msg.time_used,
                cost=msg.cost,
            )
        else:
            await self.__send_msg(
                dict(id=str(uuid.uuid4()), type="intermediate", content=msg.replace("\n", "<br>")),
                time=None,
                cost=None,
            )

    @override
    async def read_message(self, message: str) -> MessageType:
        message_data = json.loads(message)
        logger.info("Received query: {}", message_data)
        message_content = self._sanitize_str(message_data["content"])
        if message_data["type"] == "question":
            await self.__user_message_queue.put(UserInputMessage(content=message_content, uuid=message_data["id"]))
            return MessageType.question
        if message_data["type"] == "like":
            qlog.UserLikedMessageEvent.log(
                uuid=message_data["id"], text=message_content, ref_uuid=message_data["refId"]
            )
            return MessageType.like
        if message_data["type"] == "dislike":
            qlog.UserDislikedMessageEvent.log(
                uuid=message_data["id"], text=message_content, ref_uuid=message_data["refId"]
            )
            return MessageType.dislike
        if message_data["type"] == "new_question":
            return MessageType.new_question
        return MessageType.unknown

    @override
    def handle_python_output(self, python_output: PythonItem) -> None:
        self.answer.append_python(python_output._code, python_output._output)

    @override
    def handle_rtfm_output(self, snippets: list[Snippet]) -> int:
        start_index = self.answer._n_excerpts
        for snippet in snippets:
            if not snippet.page_start:
                page_range = ""
            elif snippet.page_start == snippet.page_end:
                page_range = f", p. {snippet.page_start}"
            else:
                page_range = f", p. {snippet.page_start}-{snippet.page_end}"
            link_path = str(Path("source") / Path(snippet.source_filename).name)
            if snippet.page_start:
                link_path += "#page=" + str(snippet.page_start)
            self.answer.append_rtfm(
                content=snippet.text,
                title=f"{Path(snippet.source_filename).stem}{page_range}",
                link=link_path,
            )
        return start_index

    async def __send_msg(self, msg: dict[str, Any], time: float | None, cost: float | None) -> None:
        """
        Make sure that a message is logged (even if not sent), and then send it to the frontend (over the open
        websocket), if either it is not just an intermediate message or if the server is run with
        --debug-send-intermediates.
        """
        msg_str = json.dumps(msg)
        # FIXME: We very much should record these at the level of the messages that were shown to the user,
        # not at the level of the messages that were sent to the frontend.
        qlog.SendMessageToFrontendEvent.log(uuid=msg["id"], msg=msg_str, time_used=time, cost=cost)
        if msg["type"] != "intermediate" or self.__debug_send_intermediates:
            await self.__websocket.send(msg_str)


@dataclass
class InterpolableItem(ABC):
    """
    A base class for Items that should be sent together with an Answer and were previously generated from an
    Interpolable.

    For example either a PythonItem including python code and output of a piece of python code that was run from an
    interpolable.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        ...


@dataclass
class PythonItem(InterpolableItem):
    _code: str
    _output: str

    @override
    def to_dict(self) -> dict[str, str]:
        return {"code": self._code, "output": self._output}


@dataclass
class RTFMExcerpt(InterpolableItem):
    _index: int
    _content: str
    _title: str | None
    _link: str | None

    @override
    def to_dict(self, debug: bool = False) -> dict[str, str | int | None]:
        return_dict: dict[str, str | int | None] = {
            "title": self._title,
            "link": self._link,
        }
        if debug:
            return_dict["index"] = str(self._index)
            return_dict["content"] = self._content
        return return_dict


class Answer:
    """
    A class that collects all intermediate answers, rtfm excerpts and python code that was run to then be sent all
    together in a final answer to the frontend.
    """

    _intermediate_answers: list[Message | str]
    _rtfm_excerpts: list[RTFMExcerpt]
    _python: list[PythonItem]
    _n_excerpts: int

    def __init__(self) -> None:
        self._intermediate_answers = []
        self._rtfm_excerpts = []
        self._python = []
        self._n_excerpts = 0

    def is_empty(self) -> bool:
        return len(self._intermediate_answers) < 1

    def append_answer(self, answer: Message | str) -> None:
        self._intermediate_answers.append(answer)

    def append_rtfm(self, content: str, title: str | None = None, link: str | None = None) -> None:
        self._rtfm_excerpts.append(RTFMExcerpt(_index=self._n_excerpts, _content=content, _title=title, _link=link))
        self._n_excerpts += 1

    def append_python(self, code: str, output: str) -> None:
        self._python.append(PythonItem(_code=code, _output=output))

    def __intermediates_to_list_of_dicts(
        self,
    ) -> tuple[list[dict[str, str | float | None]], dict[str, str | float | None]]:
        messages = []
        final_answer = {}
        total_time: float = 0
        total_cost: float = 0
        for msg in self._intermediate_answers:
            if isinstance(msg, Message):
                messages.append(dict(content=msg.text.replace("\n", "<br>"), time_used=msg.time_used, cost=msg.cost))
                total_time += msg.time_used if isinstance(msg.time_used, float) else 0
                total_cost += msg.cost if isinstance(msg.cost, float) else 0
            else:
                messages.append(dict(content=msg, time_used=None, cost=None))
        final_answer["content"] = messages[-1]["content"]
        final_answer["time_used"] = total_time
        final_answer["cost"] = total_cost
        return messages, final_answer

    def to_message(self, debug: bool = False) -> dict[str, Any]:
        intermediates_dict, final_answer = self.__intermediates_to_list_of_dicts()
        data: dict[str, Any] = {"final_answer": final_answer}
        if debug:
            data["intermediate_answers"] = intermediates_dict
            data["python"] = [python.to_dict() for python in self._python]
        data["rtfm_excerpts"] = [excerpt.to_dict(debug) for excerpt in self._rtfm_excerpts]
        return data


class QueryEngineBase(ABC):
    """
    A base class for query engines.

    These need to be stateless; there is a single instance of an engine that is used to serve all
    user sessions.
    """

    async def ensure_loaded(self) -> None:  # noqa: B027 (intentional empty non-abstract method)
        """Called at initialization time before the first query is handled."""

    @abstractmethod
    async def arun(self, frontend: Frontend) -> None:
        """
        Starts a process that can send messages and request information/questions from the user. At the beginning
        get_user_input has to be called to initiate the process with the frontend.
        """

    @logger.catch
    async def aserve_forever(self, args: ProgramArgsMixin) -> None:
        asyncio.get_event_loop().slow_callback_duration = 0.05  # 50 ms

        # Used to create and close a query logging context.
        @logger.catch
        async def ahandle_query(
            websocket: websockets.WebSocketServerProtocol, path: str  # type: ignore[name-defined]
        ) -> None:
            nonlocal args

            async def awebsocket_message_handler(frontend: Frontend) -> None:
                try:
                    async for message in websocket:
                        message_type = await frontend.read_message(message)
                        if message_type is MessageType.new_question:
                            query_handler_task.cancel()
                            # by closing the websocket, the whole engine in the back is relaunched for a new question
                            # (not keeping the old questions)
                            await websocket.close()
                        elif message_type is MessageType.unknown:
                            raise UnknownMessageType(f'Error: Unknown message type "{message["type"]}".')
                except (
                    websockets.exceptions.ConnectionClosedError,  # type: ignore[attr-defined]
                    websockets.exceptions.ConnectionClosedOK,  # type: ignore[attr-defined]
                ):
                    logger.info("WebSocket connection closed")
                finally:
                    query_handler_task.cancel()
                    # by closing the websocket, the whole engine in the back is relaunched for a new question
                    # (not keeping the old questions)
                    await websocket.close()

            async def aquery_handler(frontend: Frontend) -> None:
                await self.arun(frontend)

            frontend = WebFrontend(websocket=websocket, debug_send_intermediates=args.debug_send_intermediates)

            try:
                remote_address: tuple[Any, ...] = (
                    websocket.request_headers.get("X-Real-IP"),
                    int(websocket.request_headers.get("X-Remote-Port")),
                )
            except TypeError:
                remote_address = websocket.remote_address
            if remote_address[0] is None:
                remote_address = websocket.remote_address
            with qlog.ConnectionEvent.section(
                id=websocket.id, local_addr=websocket.local_address, remote_addr=str(remote_address)
            ):
                # create tasks for the websocket message handler and the query handler
                query_handler_task = asyncio.create_task(aquery_handler(frontend))
                user_message_handler_task = asyncio.create_task(awebsocket_message_handler(frontend))

                # wait for both tasks to finish (they won't, under normal conditions)
                await asyncio.gather(user_message_handler_task, query_handler_task)

        logger.info("Loading all engine resources...")
        await self.ensure_loaded()
        logger.info("Starting websocket server...")

        try:
            async with websockets.serve(  # type: ignore[attr-defined]
                ahandle_query, "0.0.0.0" if args.public else "127.0.0.1", port=args.port
            ):
                await asyncio.Future()
        except websockets.exceptions.ConnectionClosedOK as e:  # type: ignore[attr-defined]
            logger.info("Connection closed gracefully:", e)

    def serve_forever(self, args: ProgramArgsMixin) -> None:
        asyncio.run(self.aserve_forever(args), debug=args.debug_asyncio)
