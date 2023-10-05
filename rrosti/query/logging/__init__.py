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


from rrosti.query.logging.query_logging import (
    ChatCompletionCallEvent,
    ConnectionEvent,
    LoggingNotInitializedError,
    QueryEvent,
    ReceivedMessageFromFrontendEvent,
    SendMessageToFrontendEvent,
    ServerStartedEvent,
    UserDislikedMessageEvent,
    UserInputReceivedEvent,
    UserLikedMessageEvent,
    WaitForUserInputEvent,
    init_disabled_logging,
)

__all__ = [
    "ChatCompletionCallEvent",
    "ConnectionEvent",
    "LoggingNotInitializedError",
    "QueryEvent",
    "ReceivedMessageFromFrontendEvent",
    "SendMessageToFrontendEvent",
    "ServerStartedEvent",
    "UserDislikedMessageEvent",
    "UserInputReceivedEvent",
    "UserLikedMessageEvent",
    "WaitForUserInputEvent",
    "init_disabled_logging",
]
