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

import asyncio
import json
from typing import Callable

import pytest
import websockets
from pytest_mock import MockerFixture

import rrosti.query.logging as qlog
from rrosti.servers import websocket_query_server as wqs
from rrosti.utils import config

config.load_test_config()

WebSocketMocker = Callable[[], websockets.WebSocketServerProtocol]  # type: ignore[name-defined]


@pytest.fixture
def websocket_mocker(mocker: MockerFixture) -> WebSocketMocker:
    def _run() -> websockets.WebSocketServerProtocol:  # type: ignore[name-defined]
        return mocker.Mock(spec=websockets.WebSocketServerProtocol)  # type: ignore[attr-defined]

    return _run


async def test_web_frontend_sessions_simple(websocket_mocker: WebSocketMocker, mocker: MockerFixture) -> None:
    qlog.init_disabled_logging()

    ws1 = websocket_mocker()
    ws2 = websocket_mocker()

    front1 = wqs.WebFrontend(ws1)
    front2 = wqs.WebFrontend(ws2)

    mocker.patch.object(front1, "_get_user_input_impl", return_value=wqs.UserInputMessage("Hello front1", "1"))

    _, _, front2_msg, front1_msg = await asyncio.gather(
        front1.read_message(json.dumps(dict(type="question", content="Hello front1", id="1"))),
        front2.read_message(json.dumps(dict(type="question", content="Hello front2", id="2"))),
        front2.get_user_input(),
        front1.get_user_input(),
    )

    assert front1_msg == wqs.UserInputMessage("Hello front1", "1")
    assert front2_msg == wqs.UserInputMessage("Hello front2", "2")
